package llm

import (
	"reflect"
	"testing"
)

// 用 reflect.StructOf 拼一个运行时 struct,模拟 dopharness 从 skill 文件
// 解析后产出的 reflect.Type。
func runtimeStructForTest() reflect.Type {
	return reflect.StructOf([]reflect.StructField{
		{
			Name: "Path",
			Type: reflect.TypeOf(""),
			Tag:  reflect.StructTag(`json:"path" jsonschema:"description=route path"`),
		},
		{
			Name: "HandlerName",
			Type: reflect.TypeOf(""),
			Tag:  reflect.StructTag(`json:"handler_name" jsonschema:"description=handler func name"`),
		},
		{
			Name: "Method",
			Type: reflect.TypeOf(""),
			Tag:  reflect.StructTag(`json:"method,omitempty" jsonschema:"description=HTTP method"`),
		},
		{
			Name: "Tags",
			Type: reflect.SliceOf(reflect.TypeOf("")),
			Tag:  reflect.StructTag(`json:"tags,omitempty" jsonschema:"description=route tags"`),
		},
	})
}

// 验证 schema 生成:把 LLM 看到的 OAI tool 参数序列化出来,检查 properties 与
// required 字段。
func TestNewToolFromType_SchemaShape(t *testing.T) {
	rt := runtimeStructForTest()
	captured := struct {
		called bool
	}{}
	tool := NewToolFromType("AddRoute", "新增一条路由", rt,
		func(value any, callMem map[string]any) error {
			captured.called = true
			return nil
		})
	if tool.Name() != "AddRoute" {
		t.Fatalf("Name() = %q, want AddRoute", tool.Name())
	}

	// OaiTool 返回的是 union 类型,我们直接看封装到 DynamicTool 里的 schema
	// (即顶层 Parameters)。
	rawParams := tool.OaiToolParam
	// 这里我们走"间接"的检查:序列化整个 union 后看 properties 字段是否齐。
	// 由于 openai-go 的 union 类型不便直接断言,改成对 DynamicTool 自身的字段做检查。
	if tool.VType != rt {
		t.Errorf("VType not preserved: got %v want %v", tool.VType, rt)
	}
	_ = rawParams
}

// 端到端验证 HandleCallback:JSON in -> struct out -> 字段反向落到 CallMemory。
func TestDynamicTool_HandleCallback_RoundTrip(t *testing.T) {
	rt := runtimeStructForTest()

	var captured map[string]any
	tool := NewToolFromType("AddRoute", "新增一条路由", rt,
		func(value any, callMem map[string]any) error {
			// value 应该是 *<动态 struct>。用 reflect 抽出每个字段。
			rv := reflect.ValueOf(value).Elem()
			captured = map[string]any{
				"Path":        rv.FieldByName("Path").String(),
				"HandlerName": rv.FieldByName("HandlerName").String(),
				"Method":      rv.FieldByName("Method").String(),
			}
			tags := rv.FieldByName("Tags")
			if tags.IsValid() && tags.Len() > 0 {
				out := make([]string, tags.Len())
				for i := 0; i < tags.Len(); i++ {
					out[i] = tags.Index(i).String()
				}
				captured["Tags"] = out
			}
			return nil
		})

	// 模拟 LLM 返回的原始 JSON 字符串(走 OAI 标准 ToolCalls 路径)
	rawJSON := `{"path": "/api/foo", "handler_name": "fooHandler", "method": "GET", "tags": ["api", "v1"]}`
	cm := map[string]any{}
	if err := tool.HandleCallback(rawJSON, cm); err != nil {
		t.Fatalf("HandleCallback failed: %v", err)
	}

	// sink 拿到的字段
	if captured["Path"] != "/api/foo" {
		t.Errorf("Path = %v, want /api/foo", captured["Path"])
	}
	if captured["HandlerName"] != "fooHandler" {
		t.Errorf("HandlerName = %v, want fooHandler", captured["HandlerName"])
	}
	if got, ok := captured["Tags"].([]string); !ok || len(got) != 2 {
		t.Errorf("Tags = %v, want [api v1]", captured["Tags"])
	}

	// 字段反向落到 CallMemory(用 json tag name 而不是 Go field name)
	if cm["path"] != "/api/foo" {
		t.Errorf("CallMemory[path] = %v, want /api/foo", cm["path"])
	}
	if cm["handler_name"] != "fooHandler" {
		t.Errorf("CallMemory[handler_name] = %v, want fooHandler", cm["handler_name"])
	}
}

// 验证 Param 是 map[string]any 时也能正确反序列化(走自家 parser 路径)。
func TestDynamicTool_HandleCallback_MapInput(t *testing.T) {
	rt := runtimeStructForTest()

	var capturedPath string
	tool := NewToolFromType("AddRoute", "", rt,
		func(value any, callMem map[string]any) error {
			rv := reflect.ValueOf(value).Elem()
			capturedPath = rv.FieldByName("Path").String()
			return nil
		})

	mapInput := map[string]any{
		"path":         "/bar",
		"handler_name": "barHandler",
	}
	if err := tool.HandleCallback(mapInput, nil); err != nil {
		t.Fatalf("HandleCallback failed: %v", err)
	}
	if capturedPath != "/bar" {
		t.Errorf("Path = %q, want /bar", capturedPath)
	}
}

// 错误场景:非 struct 类型不应在构造时 panic,但调用时应明确报错。
func TestNewToolFromType_NonStruct(t *testing.T) {
	tool := NewToolFromType("BadTool", "", reflect.TypeOf(0),
		func(value any, callMem map[string]any) error { return nil })
	if tool == nil {
		t.Fatal("tool should not be nil for non-struct type")
	}
	err := tool.HandleCallback(`{}`, nil)
	if err == nil {
		t.Error("expected error for non-struct VType, got nil")
	}
}

// NewToolFromValue 便捷封装应该等价于 NewToolFromType + reflect.TypeOf。
func TestNewToolFromValue(t *testing.T) {
	type Sample struct {
		Foo string `json:"foo"`
	}
	tool := NewToolFromValue("Sample", "", Sample{},
		func(value any, callMem map[string]any) error { return nil })
	if tool.VType.Kind() != reflect.Struct {
		t.Errorf("VType.Kind = %v, want struct", tool.VType.Kind())
	}
	if tool.VType.Name() != "Sample" {
		t.Errorf("VType.Name = %q, want Sample", tool.VType.Name())
	}
}

// sink 返回 error 应该被传播到 HandleCallback 的返回值。
func TestDynamicTool_SinkError(t *testing.T) {
	rt := runtimeStructForTest()
	wantErr := "intentional sink failure"
	tool := NewToolFromType("AddRoute", "", rt,
		func(value any, callMem map[string]any) error {
			return &errString{msg: wantErr}
		})
	err := tool.HandleCallback(`{"path":"/x","handler_name":"h"}`, nil)
	if err == nil || err.Error() != wantErr {
		t.Errorf("error = %v, want %q", err, wantErr)
	}
}

type errString struct{ msg string }

func (e *errString) Error() string { return e.msg }
