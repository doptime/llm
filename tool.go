package llm

import (
	"encoding/json"
	"log"
	"reflect"
	"strings"

	"github.com/mitchellh/mapstructure"
	openai "github.com/sashabaranov/go-openai"

	genai "google.golang.org/genai"
)

type ToolInterface interface {
	HandleCallback(Param interface{}, CallMemory map[string]any) (err error)
	OaiTool() *openai.Tool
	GoogleGenaiTool() *genai.FunctionDeclaration
	Name() string
}

// Tool 是FuctionCall的逻辑实现。FunctionCall 是Tool的接口定义
type Tool[v any] struct {
	openai.Tool
	GoogleFunc genai.FunctionDeclaration
	Functions  []func(param v)
}

func (t *Tool[v]) OaiTool() *openai.Tool {
	return &t.Tool
}
func (t *Tool[v]) GoogleGenaiTool() *genai.FunctionDeclaration {
	return &t.GoogleFunc
}
func (t *Tool[v]) Name() string {
	return t.Tool.Function.Name
}

func (t *Tool[v]) WithFunction(f func(param v)) *Tool[v] {
	t.Functions = append(t.Functions, f)
	return t
}

func (t *Tool[v]) HandleCallback(Param interface{}, CallMemory map[string]any) (err error) {
	var parambytes []byte
	if str, ok := Param.(string); ok {
		parambytes = []byte(str)
	} else {
		parambytes, err = json.Marshal(Param)
		if err != nil {
			log.Printf("Error marshaling arguments for tool %s: %v", t.Tool.Function.Name, err)
			return err
		}
	}

	var val v
	err = json.Unmarshal(parambytes, &val) // 直接反序列化到 v 的地址
	if err != nil {
		log.Printf("Error parsing arguments for tool %s: %v。make sure type of v is a pointer to struct", t.Tool.Function.Name, err)
		return err
	}

	// Extract the memory cached key to destination struct
	if CallMemory != nil {
		if decErr := mapstructure.Decode(CallMemory, &val); decErr != nil {
			log.Printf("Warning: mapstructure decode failed for tool %s: %v", t.Tool.Function.Name, decErr)
		}
	}

	for _, f := range t.Functions {
		f(val)
	}

	if CallMemory != nil {
		// 确保传入的是一个 struct
		v := reflect.ValueOf(val)
		for v.Kind() == reflect.Ptr {
			if v.IsNil() {
				break // 阻断空指针解引用
			}
			v = v.Elem() // 如果是指针，获取其指向的值
		}
		if v.IsValid() && v.Kind() == reflect.Struct {
			tt := v.Type()

			// 遍历 struct 的所有字段
			for i := 0; i < v.NumField(); i++ {
				field := v.Field(i)
				if !field.CanInterface() {
					continue // 跳过未导出(小写字母开头)的私有字段
				}
				// 使用与 NewTool schema 生成一致的字段名（优先 json tag）
				fieldName := getFieldName(tt.Field(i))
				if fieldName == "-" {
					continue
				}

				// 将字段名和字段值添加到 map 中
				CallMemory[fieldName] = field.Interface()
			}
		}
	}

	return nil
}

// getFieldName 优先获取 json tag 中的名称，如果没有则使用字段名
func getFieldName(field reflect.StructField) string {
	jsonTag := field.Tag.Get("json")
	if jsonTag == "" {
		return field.Name
	}
	// json tag 格式可能是 "name,omitempty" 或 "-"
	parts := strings.Split(jsonTag, ",")
	name := strings.TrimSpace(parts[0])

	if name == "-" {
		return "-" // 明确表示忽略
	}
	if name == "" {
		return field.Name // 只有 omitempty 没有名字的情况
	}
	return name
}

// NewTool creates a new tool, correctly generating schemas for nested structs and slices.
func NewTool[v any](name string, description string, fs ...func(param v)) *Tool[v] {
	vType := reflect.TypeOf(new(v)).Elem()
	for vType.Kind() == reflect.Ptr {
		vType = vType.Elem()
	}

	oaiProperties := make(map[string]any)
	googleProperties := make(map[string]*genai.Schema)
	var requiredFields []string

	// Initialize the visited map to prevent infinite recursion
	visited := make(map[reflect.Type]bool)

	if vType.Kind() == reflect.Struct {
		for i := 0; i < vType.NumField(); i++ {
			field := vType.Field(i)
			desc := field.Tag.Get("description")
			if desc == "-" {
				continue
			}
			// 与 buildSchemaForType 保持一致：顶层也做 jsonschema tag 兜底
			if desc == "" {
				desc = field.Tag.Get("jsonschema")
			}

			// 获取字段名称（优先使用 json tag）
			paramName := getFieldName(field)
			if paramName == "-" {
				continue
			}

			// Generate the schema for each field's type using the recursive helper.
			fieldOAI, fieldGoogle := buildSchemaForType(field.Type, visited)

			// The description from the tag belongs to the property definition itself.
			fieldOAI["description"] = desc
			fieldGoogle.Description = desc

			oaiProperties[paramName] = fieldOAI
			googleProperties[paramName] = fieldGoogle

			// 非 omitempty 的字段视为 required
			jsonTag := field.Tag.Get("json")
			isOptional := strings.Contains(jsonTag, "omitempty") || field.Tag.Get("required") == "false"
			if !isOptional {
				requiredFields = append(requiredFields, paramName)
			}
		}
	} else {
		log.Printf("Warning: Tool %s is created with a non-struct parameter type. No parameters will be defined.", name)
	}

	// Construct the final top-level schema object that describes the tool's parameters.
	oaiParams := map[string]any{
		"type":       "object",
		"properties": oaiProperties,
	}
	if len(requiredFields) > 0 {
		oaiParams["required"] = requiredFields
	}

	googleSchema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: googleProperties,
	}

	a := &Tool[v]{
		Tool: openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        name,
				Description: description,
				Parameters:  oaiParams,
			},
		},
		GoogleFunc: genai.FunctionDeclaration{
			Name:        name,
			Description: description,
			Parameters:  googleSchema,
		},
		Functions: fs,
	}
	return a
}

// buildSchemaForType is the recursive helper function. It generates the schema for any given type.
// Added visited map to prevent Stack Overflow on recursive types.
func buildSchemaForType(t reflect.Type, visited map[reflect.Type]bool) (map[string]any, *genai.Schema) {
	// Dereference pointers until we reach the base type
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	oaiSchema := make(map[string]any)
	googleSchema := &genai.Schema{}

	// Check for recursion
	if visited[t] {
		oaiSchema["type"] = "object"
		googleSchema.Type = genai.TypeObject
		return oaiSchema, googleSchema
	}

	// Mark type as visited for the scope of this branch
	visited[t] = true
	defer func() { delete(visited, t) }() // Backtracking: allow same type in sibling branches

	oaiSchema["type"] = mapKindToDataType(t.Kind())
	googleSchema.Type = KindToJSONType(t.Kind())

	switch t.Kind() {
	case reflect.Struct:
		// When we encounter a nested struct, we must define its properties.
		oaiProperties := make(map[string]any)
		googleProperties := make(map[string]*genai.Schema)

		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			desc := field.Tag.Get("description")
			if desc == "-" {
				continue
			}
			if desc == "" {
				desc = field.Tag.Get("jsonschema")
			}

			// 获取字段名称（优先使用 json tag）
			paramName := getFieldName(field)
			if paramName == "-" {
				continue
			}

			// Recursive call for the nested struct's fields
			subOAI, subGoogle := buildSchemaForType(field.Type, visited)

			subOAI["description"] = desc
			subGoogle.Description = desc

			oaiProperties[paramName] = subOAI
			googleProperties[paramName] = subGoogle
		}
		oaiSchema["properties"] = oaiProperties
		googleSchema.Properties = googleProperties

	case reflect.Slice, reflect.Array:
		// For a slice, we define the schema of its items.
		elemType := t.Elem()
		itemsOAI, itemsGoogle := buildSchemaForType(elemType, visited) // Recursive call
		oaiSchema["items"] = itemsOAI
		googleSchema.Items = itemsGoogle
	}

	return oaiSchema, googleSchema
}

var _mapKindToDataType = map[reflect.Kind]string{
	reflect.Struct:  "object",
	reflect.Float32: "number", reflect.Float64: "number",
	reflect.Int: "integer", reflect.Int8: "integer", reflect.Int16: "integer", reflect.Int32: "integer", reflect.Int64: "integer",
	reflect.Uint: "integer", reflect.Uint8: "integer", reflect.Uint16: "integer", reflect.Uint32: "integer", reflect.Uint64: "integer",
	reflect.String:  "string",
	reflect.Slice:   "array",
	reflect.Array:   "array",
	reflect.Bool:    "boolean",
	reflect.Invalid: "null",
	reflect.Map:     "object",
}

func mapKindToDataType(kind reflect.Kind) string {
	_type, ok := _mapKindToDataType[kind]
	if !ok {
		return "type_unspecified"
	}
	return _type
}

// KindToJSONType 使用显式映射，避免依赖 strings.ToUpper 与 genai 枚举名称的巧合对齐
var _kindToGenaiType = map[reflect.Kind]genai.Type{
	reflect.Struct:  genai.TypeObject,
	reflect.Map:     genai.TypeObject,
	reflect.Float32: genai.TypeNumber, reflect.Float64: genai.TypeNumber,
	reflect.Int: genai.TypeInteger, reflect.Int8: genai.TypeInteger, reflect.Int16: genai.TypeInteger,
	reflect.Int32: genai.TypeInteger, reflect.Int64: genai.TypeInteger,
	reflect.Uint: genai.TypeInteger, reflect.Uint8: genai.TypeInteger, reflect.Uint16: genai.TypeInteger,
	reflect.Uint32: genai.TypeInteger, reflect.Uint64: genai.TypeInteger,
	reflect.String: genai.TypeString,
	reflect.Slice:  genai.TypeArray,
	reflect.Array:  genai.TypeArray,
	reflect.Bool:   genai.TypeBoolean,
}

func KindToJSONType(kind reflect.Kind) genai.Type {
	if t, ok := _kindToGenaiType[kind]; ok {
		return t
	}
	return genai.TypeUnspecified
}
