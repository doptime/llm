package llm

import (
	"bytes"
	"encoding/json"
	"strings"
	"text/template" // 修复：使用 text/template 而非 html/template，无需 template.HTML 包装

	"github.com/samber/lo"
	openai "github.com/sashabaranov/go-openai"
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

func (toolInPrompt *ToolInPrompt) WithToolcallSysMsg(tools []openai.Tool, req *openai.ChatCompletionRequest) {
	if req == nil {
		return
	}
	if len(tools) == 0 {
		return
	}

	// text/template 不会自动转义，直接使用 string 即可，无需 template.HTML 包装
	ToolStr := []string{}
	for _, v := range tools {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false) // 禁用 HTML 转义
		if err := enc.Encode(v); err != nil {
			continue // 不要 panic，工具序列化失败不该让整个 agent 崩溃
		}

		// 去掉末尾的换行符（Encode 会自动添加换行符）
		jsonStr := strings.TrimRight(buf.String(), "\n")
		ToolStr = append(ToolStr, jsonStr)
	}
	ToolCallMsg := lo.Ternary(strings.Contains(req.Model, "GLM-4.5-Air"), ToolCallGlm45Air, ToolCallMsgQwen)
	var promptBuffer bytes.Buffer
	if err := ToolCallMsg.Execute(&promptBuffer, map[string]any{"Tools": ToolStr}); err == nil {
		if toolInPrompt.InSystemPrompt {
			msgToolCall := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: promptBuffer.String()}
			req.Messages = append([]openai.ChatCompletionMessage{msgToolCall}, req.Messages...)
		} else if toolInPrompt.InUserPrompt {
			if len(req.Messages) > 0 && req.Messages[0].Role == openai.ChatMessageRoleUser {
				req.Messages[0].Content = "\n" + promptBuffer.String() + req.Messages[0].Content
			} else {
				msgToolCall := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: promptBuffer.String()}
				req.Messages = append([]openai.ChatCompletionMessage{msgToolCall}, req.Messages...)
			}
		}
	}
}
