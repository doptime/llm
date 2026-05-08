// 本文件实现 ToolInterface 的"运行时类型"版本。与 tool.go 中泛型驱动的
// Tool[v any] 不同,DynamicTool 接受一个运行时拼装的 reflect.Type —— 典型
// 来源:
//   - reflect.StructOf 拼出的 anonymous struct 类型
//   - go/parser 读取 .go 源码后,由调用方按 AST 还原出来的 struct 类型
//
// 服务于 dopharness 的 "flywheel skill" 体系:skill 文件是一段纯数据(type
// 声明 + 同名 SOP 常量),没有任何 callback 代码。框架把每个 skill 注册成
// 一个 toolcall,LLM 调它时框架渲染 SOP 模板、起一个 sub-agent 去执行。
// 这个机制的关键就是"在运行时把 reflect.Type 变成 ToolInterface"。
package llm

import (
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"strings"

	openai "github.com/openai/openai-go/v3"
	"github.com/mitchellh/mapstructure"
	genai "google.golang.org/genai"
)

// DynamicSinkFunc 是 NewToolFromType 的回调签名。
//
// 参数:
//   value      —— 一个 *T 形态的指针,T 是构造时传入的 vType 描述的 struct,
//                  实例已经从 LLM 的 JSON 参数反序列化好,并且已经从 CallMemory
//                  做过 mapstructure 二次填充(与 Tool[v] 的 HandleCallback 保持
//                  一致语义)。
//   callMemory —— 父 Agent 的会话级状态。回调可以读取(例如父级 Context),
//                  也可以写入(框架在回调结束后还会自动把 struct 的导出字段
//                  反向写回 callMemory)。允许为 nil。
//
// 返回 error 时,会被聚合到 Agent.Call 的最终错误里;调用方(如 dopharness 的
// Run loop)由此感知失败并发起 retry。
type DynamicSinkFunc func(value any, callMemory map[string]any) error

// DynamicTool 是 ToolInterface 的运行时构造实现。它与 Tool[v] 的对外行为一致:
//
//   - schema 生成(OAI / Google)走与 Tool[v] 相同的代码路径(getFieldName、
//     buildSchemaForType,见 tool.go)
//   - HandleCallback 的双向 CallMemory 流(读入 + 字段反向导出)与 Tool[v] 完全一致
//
// 唯一差异:Functions 字段被替换为单个 sink,因为运行时类型没法持有强类型
// 闭包的切片(那要求编译期已知 v)。
type DynamicTool struct {
	OaiToolParam openai.ChatCompletionToolUnionParam
	GoogleFunc   genai.FunctionDeclaration
	FuncName     string
	VType        reflect.Type // 必须是 Kind() == reflect.Struct
	Sink         DynamicSinkFunc
}

// 编译期接口检查
var _ ToolInterface = (*DynamicTool)(nil)

func (t *DynamicTool) Name() string                                 { return t.FuncName }
func (t *DynamicTool) OaiTool() openai.ChatCompletionToolUnionParam { return t.OaiToolParam }
func (t *DynamicTool) GoogleGenaiTool() *genai.FunctionDeclaration  { return &t.GoogleFunc }

// HandleCallback 与 Tool[v].HandleCallback 在行为上严格对齐,只是把 var val v
// 换成了 reflect.New(t.VType) 分配出来的 *T。
func (t *DynamicTool) HandleCallback(Param interface{}, CallMemory map[string]any) (err error) {
	if t.VType == nil {
		return fmt.Errorf("DynamicTool %q: VType is nil", t.FuncName)
	}
	if t.VType.Kind() != reflect.Struct {
		return fmt.Errorf("DynamicTool %q: VType must be struct, got %s",
			t.FuncName, t.VType.Kind())
	}

	// 1) 把 Param 序列化成字节流。Param 可能是:
	//    - string  (来自 OAI 标准 ToolCalls 字段的原始 JSON 字符串)
	//    - map[string]any 或别的 Go 结构 (来自自家 parser:XML/markdown/...)
	var parambytes []byte
	if str, ok := Param.(string); ok {
		parambytes = []byte(str)
	} else {
		parambytes, err = json.Marshal(Param)
		if err != nil {
			log.Printf("DynamicTool %s: marshal Param: %v", t.FuncName, err)
			return err
		}
	}

	// 2) 分配一个新的 *T 实例并把 JSON 反序列化进去。
	instance := reflect.New(t.VType) // Value of type *T
	if err := json.Unmarshal(parambytes, instance.Interface()); err != nil {
		log.Printf("DynamicTool %s: unmarshal into %s: %v",
			t.FuncName, t.VType.String(), err)
		return err
	}

	// 3) 用 CallMemory 二次填充 struct(与 Tool[v] 一致,允许父域注入额外上下文)。
	if CallMemory != nil {
		if decErr := mapstructure.Decode(CallMemory, instance.Interface()); decErr != nil {
			log.Printf("DynamicTool %s: mapstructure decode warning: %v",
				t.FuncName, decErr)
		}
	}

	// 4) 调 sink。
	if t.Sink != nil {
		if err := t.Sink(instance.Interface(), CallMemory); err != nil {
			return err
		}
	}

	// 5) 把 struct 的导出字段反向写回 CallMemory(与 Tool[v] 一致)。
	if CallMemory != nil {
		rv := instance.Elem()
		if rv.IsValid() && rv.Kind() == reflect.Struct {
			tt := rv.Type()
			for i := 0; i < rv.NumField(); i++ {
				field := rv.Field(i)
				if !field.CanInterface() {
					continue
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

// NewToolFromType 用一个运行时 reflect.Type 构造 ToolInterface。
//
// 用法示意(dopharness 的 skill 加载场景):
//
//	rt, _ := buildStructType(astStructType, structName) // skills/builder.go
//	tool := llm.NewToolFromType(skillName, skillDesc, rt,
//	    func(value any, callMem map[string]any) error {
//	        // value 是 *<动态 struct>,可用 reflect.ValueOf(value).Elem() 拿到字段
//	        return runSkillSubAgent(value, callMem)
//	    })
//	agent.UseTools(tool)
//
// vType 必须是(或解引用后是)struct;否则返回的 tool 的 OaiTool() 仍然合法,
// 但 HandleCallback 会立即报错。这是为了保持与 NewTool[v any] 一致的"非 struct
// 时仍能通过编译"的容错。
func NewToolFromType(name, description string, vType reflect.Type, sink DynamicSinkFunc) *DynamicTool {
	for vType != nil && vType.Kind() == reflect.Ptr {
		vType = vType.Elem()
	}

	oaiProperties := make(map[string]any)
	googleProperties := make(map[string]*genai.Schema)
	var requiredFields []string

	visited := make(map[reflect.Type]bool)

	if vType != nil && vType.Kind() == reflect.Struct {
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

			// 直接复用 tool.go 中的 buildSchemaForType,不在这里重复实现。
			fieldOAI, fieldGoogle := buildSchemaForType(field.Type, visited)
			fieldOAI["description"] = desc
			fieldGoogle.Description = desc

			oaiProperties[paramName] = fieldOAI
			googleProperties[paramName] = fieldGoogle

			jsonTag := field.Tag.Get("json")
			isOptional := strings.Contains(jsonTag, "omitempty") ||
				field.Tag.Get("required") == "false"
			if !isOptional {
				requiredFields = append(requiredFields, paramName)
			}
		}
	} else {
		log.Printf("Warning: NewToolFromType for %s called with non-struct type", name)
	}

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

	return &DynamicTool{
		OaiToolParam: openai.ChatCompletionFunctionTool(funcDef),
		FuncName:     name,
		GoogleFunc: genai.FunctionDeclaration{
			Name:        name,
			Description: description,
			Parameters:  googleSchema,
		},
		VType: vType,
		Sink:  sink,
	}
}

// NewToolFromValue 是 NewToolFromType 的便捷封装:从一个值实例推导出 vType。
// 适合手写测试或临时探索时用。
func NewToolFromValue(name, description string, valuePrototype any, sink DynamicSinkFunc) *DynamicTool {
	t := reflect.TypeOf(valuePrototype)
	return NewToolFromType(name, description, t, sink)
}
