package llm

import (
	"encoding/json"
	"log"
	"reflect"
	"strings"

	"github.com/mitchellh/mapstructure"
	openai "github.com/openai/openai-go/v3"
	genai "google.golang.org/genai"
)

type ToolInterface interface {
	HandleCallback(Param interface{}, CallMemory map[string]any) (err error)
	OaiTool() openai.ChatCompletionToolUnionParam
	GoogleGenaiTool() *genai.FunctionDeclaration
	Name() string
}

// Tool 是 FuctionCall 的逻辑实现。FunctionCall 是 Tool 的接口定义。
//
// 在新版 openai-go 中，工具用 ChatCompletionToolUnionParam 表示，
// 通过 openai.ChatCompletionFunctionTool(FunctionDefinitionParam{...}) 构造。
type Tool[v any] struct {
	OaiToolParam openai.ChatCompletionToolUnionParam
	FuncName     string
	GoogleFunc   genai.FunctionDeclaration
	Functions    []func(param v)
}

func (t *Tool[v]) OaiTool() openai.ChatCompletionToolUnionParam {
	return t.OaiToolParam
}

func (t *Tool[v]) GoogleGenaiTool() *genai.FunctionDeclaration {
	return &t.GoogleFunc
}

func (t *Tool[v]) Name() string {
	return t.FuncName
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
			log.Printf("Error marshaling arguments for tool %s: %v", t.FuncName, err)
			return err
		}
	}

	var val v
	err = json.Unmarshal(parambytes, &val) // 直接反序列化到 v 的地址
	if err != nil {
		log.Printf("Error parsing arguments for tool %s: %v。make sure type of v is a pointer to struct", t.FuncName, err)
		return err
	}

	// Extract the memory cached key to destination struct
	if CallMemory != nil {
		if decErr := mapstructure.Decode(CallMemory, &val); decErr != nil {
			log.Printf("Warning: mapstructure decode failed for tool %s: %v", t.FuncName, decErr)
		}
	}

	for _, f := range t.Functions {
		f(val)
	}

	if CallMemory != nil {
		// 确保传入的是一个 struct
		rv := reflect.ValueOf(val)
		for rv.Kind() == reflect.Ptr {
			if rv.IsNil() {
				break // 阻断空指针解引用
			}
			rv = rv.Elem()
		}
		if rv.IsValid() && rv.Kind() == reflect.Struct {
			tt := rv.Type()
			for i := 0; i < rv.NumField(); i++ {
				field := rv.Field(i)
				if !field.CanInterface() {
					continue // 跳过未导出（小写字母开头）的私有字段
				}
				fieldName := getFieldName(tt.Field(i))
				if fieldName == "-" {
					continue
				}
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
	parts := strings.Split(jsonTag, ",")
	name := strings.TrimSpace(parts[0])

	if name == "-" {
		return "-"
	}
	if name == "" {
		return field.Name
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

	visited := make(map[reflect.Type]bool)

	if vType.Kind() == reflect.Struct {
		for i := 0; i < vType.NumField(); i++ {
			field := vType.Field(i)
			desc := field.Tag.Get("description")
			if desc == "-" {
				continue
			}
			if desc == "" {
				desc = field.Tag.Get("jsonschema")
			}

			paramName := getFieldName(field)
			if paramName == "-" {
				continue
			}

			fieldOAI, fieldGoogle := buildSchemaForType(field.Type, visited)

			fieldOAI["description"] = desc
			fieldGoogle.Description = desc

			oaiProperties[paramName] = fieldOAI
			googleProperties[paramName] = fieldGoogle

			jsonTag := field.Tag.Get("json")
			isOptional := strings.Contains(jsonTag, "omitempty") || field.Tag.Get("required") == "false"
			if !isOptional {
				requiredFields = append(requiredFields, paramName)
			}
		}
	} else {
		log.Printf("Warning: Tool %s is created with a non-struct parameter type. No parameters will be defined.", name)
	}

	// openai-go 用 FunctionParameters（其实就是 map[string]any）描述 schema
	oaiParams := openai.FunctionParameters{
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

	funcDef := openai.FunctionDefinitionParam{
		Name:        name,
		Description: openai.String(description),
		Parameters:  oaiParams,
	}

	a := &Tool[v]{
		OaiToolParam: openai.ChatCompletionFunctionTool(funcDef),
		FuncName:     name,
		GoogleFunc: genai.FunctionDeclaration{
			Name:        name,
			Description: description,
			Parameters:  googleSchema,
		},
		Functions: fs,
	}
	return a
}

// buildSchemaForType is the recursive helper. It generates the schema for any given type.
// Added visited map to prevent stack overflow on recursive types.
func buildSchemaForType(t reflect.Type, visited map[reflect.Type]bool) (map[string]any, *genai.Schema) {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	oaiSchema := make(map[string]any)
	googleSchema := &genai.Schema{}

	if visited[t] {
		oaiSchema["type"] = "object"
		googleSchema.Type = genai.TypeObject
		return oaiSchema, googleSchema
	}

	visited[t] = true
	defer func() { delete(visited, t) }() // Backtracking: allow same type in sibling branches

	oaiSchema["type"] = mapKindToDataType(t.Kind())
	googleSchema.Type = KindToJSONType(t.Kind())

	switch t.Kind() {
	case reflect.Struct:
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

			paramName := getFieldName(field)
			if paramName == "-" {
				continue
			}

			subOAI, subGoogle := buildSchemaForType(field.Type, visited)

			subOAI["description"] = desc
			subGoogle.Description = desc

			oaiProperties[paramName] = subOAI
			googleProperties[paramName] = subGoogle
		}
		oaiSchema["properties"] = oaiProperties
		googleSchema.Properties = googleProperties

	case reflect.Slice, reflect.Array:
		elemType := t.Elem()
		itemsOAI, itemsGoogle := buildSchemaForType(elemType, visited)
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
