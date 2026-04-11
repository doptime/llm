package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"text/template"
	"time"

	"github.com/samber/lo"
	openai "github.com/sashabaranov/go-openai"

	"golang.design/x/clipboard"
)

type FileToMem struct {
	File string `json:"file"`
	Mem  string `json:"mem"`
}

const (
	UseContentFromClipboard        string = "ContentFromClipboard"
	UseContentFromClipboardAsParam string = "ContentFromClipboardAsParam"
	UseContentToParam              string = "ContentToMemoryKey"
	UseCopyPromptOnly              string = "CopyPromptOnly"
	UseSharedMemory                string = "SharedMemory"
	UseModel                       string = "Model"
	UseTemplate                    string = "Template"
)

// Agent is responsible for proposing goals using an OpenAI model,
// handling function calls, and managing callbacks.
type Agent struct {
	Models                      []*Model
	PromptTemplate              *template.Template
	Tools                       []openai.Tool
	ToolInSystemPrompt          bool
	ToolInUserPrompt            bool
	toolsCallbacks              map[string]func(Param interface{}, CallMemory map[string]any) error
	functioncallParsers         []func(resp openai.ChatCompletionResponse) (toolCalls []*FunctionCall)
	CallBack                    func(ctx context.Context, inputs string) error
	CheckToolCallsBeforeCalling func(toolCalls []*FunctionCall) error

	ToolCallRunningMutex *sync.Mutex // 强类型，消除 interface{} 类型断言 panic
}

func NewAgent(_template *template.Template, tools ...ToolInterface) (a *Agent) {
	a = &Agent{
		Models:         []*Model{ModelDefault},
		toolsCallbacks: map[string]func(Param interface{}, CallMemory map[string]any) error{},
		PromptTemplate: _template,
	}
	// UseTools 原地修改，不再返回 Clone 副本
	a.UseTools(tools...)
	a.WithToolcallParser(nil)
	return a
}

func (a *Agent) WithToolCallMutexRun() *Agent {
	a.ToolCallRunningMutex = &sync.Mutex{}
	return a
}

func (a *Agent) WithToolCallsCheckedBeforeCalling(checkToolCallsBeforeCalling func(toolCalls []*FunctionCall) error) *Agent {
	a.CheckToolCallsBeforeCalling = checkToolCallsBeforeCalling
	return a
}

// UseTools 原地添加工具，与 WithModels / WithCallback 等 mutator 风格一致。
// 如需派生独立配置的 agent，请先 Clone() 再 UseTools()。
func (a *Agent) UseTools(tools ...ToolInterface) *Agent {
	for _, tool := range tools {
		a.Tools = append(a.Tools, *tool.OaiTool())
		a.toolsCallbacks[tool.Name()] = tool.HandleCallback
	}
	return a
}

type FieldReaderFunc func(content string) (field string)

// Clone 返回 Agent 的独立副本，深拷贝 Tools / toolsCallbacks / Models / parsers。
// 注意：Clone 不提供并发安全保证 — 如果多 goroutine 共享同一 agent，
// 应在调用方做同步。Clone 的用途是"派生一个独立配置的 agent 实例"。
func (a *Agent) Clone() *Agent {
	var b Agent = *a
	b.toolsCallbacks = make(map[string]func(interface{}, map[string]any) error, len(a.toolsCallbacks))
	for k, v := range a.toolsCallbacks {
		b.toolsCallbacks[k] = v
	}
	b.Tools = append([]openai.Tool{}, a.Tools...)
	b.functioncallParsers = append([]func(openai.ChatCompletionResponse) []*FunctionCall{}, a.functioncallParsers...)
	b.Models = append([]*Model{}, a.Models...)

	return &b
}
func (a *Agent) WithModels(Model ...*Model) *Agent {
	a.Models = Model
	return a
}

func (a *Agent) WithCallback(callback func(ctx context.Context, inputs string) error) *Agent {
	a.CallBack = callback
	return a
}

// Message 渲染 prompt。返回空串和 error 表示渲染失败。
func (a *Agent) Message(params map[string]any) (string, error) {
	var buf bytes.Buffer
	tmpl := a.PromptTemplate
	if t, ok := params[UseTemplate].(*template.Template); ok && t != nil {
		tmpl = t
	}
	if tmpl == nil {
		return "", fmt.Errorf("no prompt template available")
	}
	if err := tmpl.Execute(&buf, params); err != nil {
		return "", fmt.Errorf("render prompt: %w", err)
	}
	return buf.String(), nil
}

// Messege 兼容老调用方的拼写。Deprecated: use Message.
func (a *Agent) Messege(params map[string]any) string {
	s, _ := a.Message(params)
	return s
}

// Call generates goals based on the provided file contents.
// It renders the prompt, sends a request to the OpenAI model, and processes the response.
func (a *Agent) Call(memories ...map[string]any) (err error) {
	// 用本地副本承载本次调用的渲染上下文，避免污染调用方传入的 map，
	// 也避免与并发的其他 Call 共享写。
	params := make(map[string]any)
	var caller map[string]any
	if len(memories) > 0 && memories[0] != nil {
		caller = memories[0]
	}

	// 1) 先铺底 SharedMemory（全局/会话级状态）
	if shared, ok := caller[UseSharedMemory].(map[string]any); ok && shared != nil {
		for k, v := range shared {
			params[k] = v
		}
	}
	// 2) 再用本次调用的 caller 覆盖（本次入参优先级更高）
	for k, v := range caller {
		if k == UseSharedMemory {
			continue
		}
		params[k] = v
	}
	// 3) 注入受控字段
	params["ThisAgent"] = a

	if memDeCliboardKey, _ok := params[UseContentFromClipboardAsParam].(string); _ok && memDeCliboardKey != "" {
		textbytes := clipboard.Read(clipboard.FmtText)
		if len(textbytes) == 0 {
			fmt.Println("no data in clipboard")
			return nil
		}
		params[memDeCliboardKey] = string(textbytes)
	}

	messege, err := a.Message(params)
	if err != nil {
		return err
	}
	fmt.Printf("Requesting prompt: %v\n", messege)

	// LoadbalancedPick 只调用一次，确保 model 和 params[UseModel] 指向同一个实例
	model, ok := params[UseModel].(*Model)
	if !ok || model == nil {
		model = LoadbalancedPick(a.Models...)
		params[UseModel] = model
	}

	// Create the chat completion request with function calls enabled
	req := openai.ChatCompletionRequest{
		Model:       model.Name,
		Messages:    []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: messege}},
		TopP:        model.TopP,
		Temperature: model.Temperature,
	}
	if model.SystemMessage != "" {
		req.Messages = append([]openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleSystem, Content: model.SystemMessage}}, req.Messages...)
	}
	if model.Temperature > 0 {
		req.Temperature = model.Temperature
	}
	if model.TopP > 0 {
		req.TopP = model.TopP
	}
	if len(a.Tools) > 0 {
		if model.ToolInPrompt != nil {
			model.ToolInPrompt.WithToolcallSysMsg(a.Tools, &req)
		} else {
			req.Tools = a.Tools
		}
	}

	if copyPromptOnly, ok := params[UseCopyPromptOnly].(bool); ok && copyPromptOnly {
		msg := strings.Join(lo.Map(req.Messages, func(m openai.ChatCompletionMessage, _ int) string { return m.Content }), "\n")
		err := clipboard.Init()
		if err != nil {
			return fmt.Errorf("error initializing clipboard: %w", err)
		}
		var sb strings.Builder
		for _, r := range msg {
			if r != '\x00' {
				sb.WriteRune(r)
			}
		}
		fmt.Println("copy prompt to clipboard", msg)
		msg = sb.String()
		clipboard.Write(clipboard.FmtText, []byte(msg))
		return nil
	}
	timestart := time.Now()
	reqCtx := context.Background()
	if c, ok := params["Context"].(context.Context); ok && c != nil {
		reqCtx = c
	}
	// 兜底：如果调用方没有自带超时，给 30 分钟硬上限，避免请求永远悬挂
	if _, hasDeadline := reqCtx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		reqCtx, cancel = context.WithTimeout(reqCtx, 30*time.Minute)
		defer cancel()
	}

	// loading Message response
	var resp openai.ChatCompletionResponse
	msgClipboardUsed := false
	if MsgClipboard, _ok := params[UseContentFromClipboard].(bool); _ok && MsgClipboard {
		textbytes := clipboard.Read(clipboard.FmtText)
		if len(textbytes) == 0 {
			return fmt.Errorf("no data in clipboard")
		}
		msg := openai.ChatCompletionMessage{Role: "assistant", Content: string(textbytes)}
		resp = openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: msg}}}
		msgClipboardUsed = true
	} else if len(req.Messages) > 0 {
		resp, err = model.Client.CreateChatCompletion(reqCtx, req)
	} else {
		return fmt.Errorf("no messages in request")
	}

	if err != nil {
		fmt.Println("Error creating chat completion:", err)
		if len(req.Messages) > 0 {
			fmt.Println("req:", req.Messages[0].Content)
		}
		return err
	}
	if !msgClipboardUsed {
		model.ResponseTime(time.Since(timestart))
	}
	if model.PrintMsg {
		fmt.Println("resp:", resp)
	}

	if len(resp.Choices) == 0 {
		return fmt.Errorf("empty choices in response")
	}
	content := resp.Choices[0].Message.Content

	// saving to memory: 回写到调用方传入的 caller，使调用方能在 Call 返回后读到结果
	if msgToMemKey, _ok := params[UseContentToParam].(string); _ok && msgToMemKey != "" {
		params[msgToMemKey] = content
		if caller != nil {
			caller[msgToMemKey] = content
		}
	}

	if a.CallBack != nil {
		a.CallBack(reqCtx, content)
	}

	// Parse and handle function calls in the response
	// 使用 (name, argsHash) 联合键去重，避免同名不同参或异名同参的误判
	type toolCallKey struct {
		name string
		hash uint64
	}
	seen := map[toolCallKey]struct{}{}
	var nonRedundantToolCalls []*FunctionCall
	for _, parser := range a.functioncallParsers {
		for _, tc := range parser(resp) {
			hash, _ := GetCanonicalHash(tc.Arguments)
			key := toolCallKey{name: tc.Name, hash: hash}
			if _, dup := seen[key]; dup {
				continue
			}
			seen[key] = struct{}{}
			nonRedundantToolCalls = append(nonRedundantToolCalls, tc)
		}
	}
	if a.CheckToolCallsBeforeCalling != nil {
		if err := a.CheckToolCallsBeforeCalling(nonRedundantToolCalls); err != nil {
			return err
		}
	}

	// 工具回调：聚合错误而非静默吞没，使用 defer 释放锁防 panic 泄漏
	var toolErrs []error
	for _, toolcall := range nonRedundantToolCalls {
		_tool, ok := a.toolsCallbacks[toolcall.Name]
		if !ok {
			return fmt.Errorf("tool %q not found in FunctionMap", toolcall.Name)
		}
		callErr := func() error {
			if a.ToolCallRunningMutex != nil {
				a.ToolCallRunningMutex.Lock()
				defer a.ToolCallRunningMutex.Unlock()
			}
			return _tool(toolcall.Arguments, params)
		}()
		if callErr != nil {
			toolErrs = append(toolErrs, fmt.Errorf("tool %q execution failed: %w", toolcall.Name, callErr))
		}
	}

	if len(toolErrs) > 0 {
		return fmt.Errorf("one or more tool calls failed: %w", errors.Join(toolErrs...))
	}
	return nil
}
