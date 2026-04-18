package llm

import (
	"encoding/json"
	"regexp"
	"strings"

	"github.com/openai/openai-go/v3"
)

// FunctionCall 表示一次工具调用（已解析）。
// Arguments 既可能是 map[string]any（自己解析的 XML/JSON），
// 也可能是 string（从 OAI 标准 tool_calls 直传的原始 JSON 字符串）。
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments any    `json:"arguments,omitempty"`
}

func parseOneToolcall(toolcallString string) *FunctionCall {
	if len(toolcallString) == 0 {
		return nil
	}

	tryUnmarshal := func(s string) (FunctionCall, error) {
		t := FunctionCall{Arguments: map[string]any{}}
		e := json.Unmarshal([]byte(s), &t)
		return t, e
	}

	tool, err := tryUnmarshal(toolcallString)
	if err != nil {
		// 尝试补尾 "}"
		tool, err = tryUnmarshal(toolcallString + "}")
	}
	if err != nil && len(toolcallString) > 1 {
		// 尝试去尾字符
		tool, err = tryUnmarshal(toolcallString[:len(toolcallString)-1])
	}
	if err != nil {
		return nil
	}
	if tool.Name == "" {
		return nil
	}

	// 安全地读取 Arguments，不做裸类型断言
	args, ok := tool.Arguments.(map[string]any)
	if !ok || len(args) == 0 {
		// 兜底：把整个 string 当 arguments 再试一次
		var alt map[string]any
		if e := json.Unmarshal([]byte(toolcallString), &alt); e == nil && len(alt) > 0 {
			tool.Arguments = alt
			return &tool
		}
		return nil
	}
	return &tool
}

func ParseToolCallFromXlm(s string) (toolCalls *FunctionCall) {
	// Extract function name
	nameRe := regexp.MustCompile(`<invoke name="([^"]+)"`)
	nameMatches := nameRe.FindStringSubmatch(s)
	if len(nameMatches) < 2 {
		return nil
	}
	functionName := strings.TrimSpace(nameMatches[1])

	// Extract all parameters
	paramRe := regexp.MustCompile(`(?s)<parameter name="([^"]+)">(.+?)</parameter>`)
	paramMatches := paramRe.FindAllStringSubmatch(s, -1)

	args := make(map[string]interface{})
	for _, match := range paramMatches {
		if len(match) < 3 {
			continue
		}
		key := strings.TrimSpace(match[1])
		value := strings.TrimSpace(match[2])

		var jsonValue interface{}
		if err := json.Unmarshal([]byte(value), &jsonValue); err == nil {
			args[key] = jsonValue
		} else {
			args[key] = value
		}
	}

	return &FunctionCall{
		Name:      functionName,
		Arguments: args,
	}
}

// toolCallReplacer 使用预编译的 Replacer 进行单遍替换，
// 替代原来 10+ 次 strings.ReplaceAll 链，减少内存分配和 GC 压力。
var toolCallReplacer = strings.NewReplacer(
	"minimax:tool_call>", "tool_call>",
	"/function_calls>", "tool_call>",
	"function_calls>", "tool_call>",
	"tool_code>", "tool_call>",
	"<tool>", "<tool_call>",
	"</tools>", "<tool_call>",
	"</tool_call>", "<tool_call>",
	"```json\n", "<tool_call>",
	"```tool_call\n", "<tool_call>",
	"\n```", "<tool_call>",
	"```\n", "<tool_call>",
	"```tool_call>", "<tool_call>",
)

// ToolcallParserDefault 解析新版 openai-go 返回的 ChatCompletion 中的工具调用。
//
// 在 openai-go v1 里，每条 ToolCall 是一个 union 类型 ChatCompletionMessageToolCallUnion，
// 标准 function tool 的字段路径是 .Function.Name 和 .Function.Arguments（string，原始 JSON）。
// 当模型把工具调用塞在 Content 里（XML/markdown 包裹）时，回退到字符串解析路径。
func ToolcallParserDefault(resp *openai.ChatCompletion) (toolCalls []*FunctionCall) {
	if resp == nil {
		return nil
	}
	for _, choice := range resp.Choices {
		for _, tc := range choice.Message.ToolCalls {
			// Union 的 .Function 字段对 function 类型的 tool call 直接可用；
			// 对 custom tool call，Function.Name 会是空字符串，这里就跳过。
			name := tc.Function.Name
			if name == "" {
				continue
			}
			toolCalls = append(toolCalls, &FunctionCall{
				Name:      name,
				Arguments: tc.Function.Arguments, // 原始 JSON 字符串
			})
		}
	}

	if len(toolCalls) == 0 && len(resp.Choices) > 0 {
		rsp := resp.Choices[0].Message.Content
		ind, ind2 := strings.LastIndex(rsp, "tool_call>"), strings.LastIndex(rsp, "}")
		if ind > 0 && ind2 > ind {
			rsp = rsp[:ind2+1] + "</tool_call>"
		}

		rsp = toolCallReplacer.Replace(rsp)

		items := strings.Split(rsp, "<tool_call>")
		// case json only
		if len(items) > 3 {
			items = items[1 : len(items)-1]
		}
		for _, toolcallString := range items {
			if len(toolcallString) < 10 {
				continue
			}
			toolcallString = strings.TrimSpace(toolcallString)
			toolcall := ParseToolCallFromXlm(toolcallString)
			if toolcall == nil {
				if i := strings.Index(toolcallString, "{"); i > 0 {
					toolcallString = toolcallString[i:]
				}
				if i := strings.LastIndex(toolcallString, "}"); i > 0 {
					toolcallString = toolcallString[:i+1]
				}
				toolcall = parseOneToolcall(toolcallString)
			}
			if toolcall != nil {
				toolCalls = append(toolCalls, toolcall)
			}
		}
	}
	return toolCalls
}

// WithToolcallParser 注册一个工具调用解析器；nil 表示使用默认解析器。
func (a *Agent) WithToolcallParser(parse func(resp *openai.ChatCompletion) (toolCalls []*FunctionCall)) *Agent {
	if parse == nil {
		parse = ToolcallParserDefault
	}
	a.functioncallParsers = append(a.functioncallParsers, parse)
	return a
}
