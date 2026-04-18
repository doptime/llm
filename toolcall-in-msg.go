package llm

import (
	"bytes"
	"encoding/json"
	"strings"
	"text/template"

	"github.com/openai/openai-go/v3"
	"github.com/samber/lo"
)

type ToolInPrompt struct {
	InSystemPrompt bool
	InUserPrompt   bool
}

var ToolCallMsgQwen, _ = template.New("ToolCallMsg").Parse(`
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{{range $ind, $val := .Tools}}
{{$val}}
{{end}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
`)

var ToolCallGlm45Air, _ = template.New("ToolCallMsg").Parse(`
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{{range $ind, $val := .Tools}}
{{$val}}
{{end}}
</tools>

For each function call, output the function name and arguments within the following XML format:
<function_calls>
<invoke name="{function-name}">
<parameter name="{arg-parameter-name-1}">{arg-parameter-value-1}</parameter>
<parameter name="{arg-parameter-name-2}">{arg-parameter-value-2}</parameter>
...
</invoke>
</function_calls>
`)

// WithToolcallSysMsg 把工具签名拼到 prompt 里（system 或 user），
// 然后把请求里原本的 Tools 字段清空，避免又通过标准 OAI tools 字段又通过 prompt 重复传递。
//
// 这一路径用于不支持原生 function calling 但支持长 system prompt 的本地模型
// （比如 GLM-4.5-Air、Qwen 蒸馏版等）。
func (toolInPrompt *ToolInPrompt) WithToolcallSysMsg(tools []openai.ChatCompletionToolUnionParam, req *openai.ChatCompletionNewParams) {
	if req == nil || len(tools) == 0 {
		return
	}

	ToolStr := []string{}
	for _, v := range tools {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(v); err != nil {
			continue // 不要 panic，工具序列化失败不该让整个 agent 崩溃
		}
		jsonStr := strings.TrimRight(buf.String(), "\n")
		ToolStr = append(ToolStr, jsonStr)
	}

	tmpl := lo.Ternary(strings.Contains(req.Model, "GLM-4.5-Air"), ToolCallGlm45Air, ToolCallMsgQwen)
	var promptBuffer bytes.Buffer
	if err := tmpl.Execute(&promptBuffer, map[string]any{"Tools": ToolStr}); err != nil {
		return
	}
	promptStr := promptBuffer.String()

	if toolInPrompt.InSystemPrompt {
		// 把工具说明作为 system message 插到最前面
		sysMsg := openai.SystemMessage(promptStr)
		req.Messages = append([]openai.ChatCompletionMessageParamUnion{sysMsg}, req.Messages...)
	} else if toolInPrompt.InUserPrompt {
		// 优先把工具说明前置到第一条 user message；找不到就单独插一条
		injected := false
		for i := range req.Messages {
			userParam := req.Messages[i].OfUser
			if userParam == nil {
				continue
			}
			// user 消息的 Content 是 union 类型，最常见是简单字符串。
			// 如果是字符串形态，前置拼上工具说明；否则降级新建一条 user 消息插到最前面。
			if userParam.Content.OfString.Valid() {
				origText := userParam.Content.OfString.Value
				userParam.Content.OfString = openai.String("\n" + promptStr + origText)
				injected = true
			}
			break
		}
		if !injected {
			userMsg := openai.UserMessage(promptStr)
			req.Messages = append([]openai.ChatCompletionMessageParamUnion{userMsg}, req.Messages...)
		}
	}

	// 关键：既然已经把工具签名拼进 prompt，就清掉 tools 字段，避免
	// 不支持 function calling 的后端报错 / 重复消耗 token。
	req.Tools = nil
}
