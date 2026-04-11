package llm

import (
	"encoding/json"
	"regexp"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// Process each choice in the response
type FunctionCall struct {
	Name string `json:"name,omitempty"`
	// call function with arguments in JSON format
	Arguments any `json:"arguments,omitempty"`
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

		// Try to unmarshal as JSON, otherwise use as string
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

func ToolcallParserDefault(resp openai.ChatCompletionResponse) (toolCalls []*FunctionCall) {
	for _, choice := range resp.Choices {
		for _, toolcall := range choice.Message.ToolCalls {
			functioncall := &FunctionCall{
				Name:      toolcall.Function.Name,
				Arguments: toolcall.Function.Arguments,
			}
			toolCalls = append(toolCalls, functioncall)
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
		//case json only
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
func (a *Agent) WithToolcallParser(parse func(resp openai.ChatCompletionResponse) (toolCalls []*FunctionCall)) *Agent {
	if parse == nil {
		parse = ToolcallParserDefault
	}
	a.functioncallParsers = append(a.functioncallParsers, parse)
	return a
}
