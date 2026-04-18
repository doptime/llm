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

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/samber/lo"

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

// Agent is responsible for proposing goals using an OpenAI-compatible model,
// handling function calls, and managing callbacks.
type Agent struct {
	Models                      []*Model
	PromptTemplate              *template.Template
	Tools                       []openai.ChatCompletionToolUnionParam
	ToolInSystemPrompt          bool
	ToolInUserPrompt            bool
	toolsCallbacks              map[string]func(Param interface{}, CallMemory map[string]any) error
	functioncallParsers         []func(resp *openai.ChatCompletion) (toolCalls []*FunctionCall)
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
		a.Tools = append(a.Tools, tool.OaiTool())
		a.toolsCallbacks[tool.Name()] = tool.HandleCallback
	}
	return a
}

type FieldReaderFunc func(content string) (field string)

// Clone 返回 Agent 的独立副本，深拷贝 Tools / toolsCallbacks / Models / parsers。
func (a *Agent) Clone() *Agent {
	var b Agent = *a
	b.toolsCallbacks = make(map[string]func(interface{}, map[string]any) error, len(a.toolsCallbacks))
	for k, v := range a.toolsCallbacks {
		b.toolsCallbacks[k] = v
	}
	b.Tools = append([]openai.ChatCompletionToolUnionParam{}, a.Tools...)
	b.functioncallParsers = append([]func(*openai.ChatCompletion) []*FunctionCall{}, a.functioncallParsers...)
	b.Models = append([]*Model{}, a.Models...)
	return &b
}

func (a *Agent) UseModels(Model ...*Model) *Agent {
	a.Models = Model
	return a
}

func (a *Agent) UseModelsNamed(Models ...string) *Agent {
	for _, name := range Models {
		m, ok := ModelsMap[name]
		if !ok {
			panic(fmt.Sprintf("model %q not found in ModelsMap", name))
		}
		a.Models = append(a.Models, m)
	}
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

// Call generates output by sending the rendered prompt to the chosen model and
// dispatching any returned tool calls to their registered callbacks.
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

	// 构造请求体（新版 openai-go 使用 ChatCompletionNewParams）
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(messege),
	}
	if model.SystemMessage != "" {
		messages = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage(model.SystemMessage)}, messages...)
	}

	req := openai.ChatCompletionNewParams{
		Model:    model.Name,
		Messages: messages,
	}
	if model.Temperature > 0 {
		req.Temperature = openai.Float(model.Temperature)
	}
	if model.TopP > 0 {
		req.TopP = openai.Float(model.TopP)
	}
	if len(a.Tools) > 0 {
		if model.ToolInPrompt != nil {
			// 把工具签名塞进 prompt（同时清空 req.Tools）
			model.ToolInPrompt.WithToolcallSysMsg(a.Tools, &req)
		} else {
			req.Tools = a.Tools
		}
	}

	// extra_body：把每个 key 通过 option.WithJSONSet 注入到 HTTP body 顶层
	// 例如 ExtraBody = {"chat_template_kwargs": {"enable_thinking": false}}
	// 会在请求 JSON 里出现 "chat_template_kwargs": {"enable_thinking": false}
	var reqOpts []option.RequestOption
	for k, v := range model.ExtraBody {
		reqOpts = append(reqOpts, option.WithJSONSet(k, v))
	}

	if copyPromptOnly, ok := params[UseCopyPromptOnly].(bool); ok && copyPromptOnly {
		msg := strings.Join(lo.Map(req.Messages, func(m openai.ChatCompletionMessageParamUnion, _ int) string {
			return getMessageText(m)
		}), "\n")
		if err := clipboard.Init(); err != nil {
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
	var resp *openai.ChatCompletion
	msgClipboardUsed := false
	if MsgClipboard, _ok := params[UseContentFromClipboard].(bool); _ok && MsgClipboard {
		textbytes := clipboard.Read(clipboard.FmtText)
		if len(textbytes) == 0 {
			return fmt.Errorf("no data in clipboard")
		}
		// 伪造一个 ChatCompletion，便于走完后续解析路径。
		// 注意：在 openai-go 中 ChatCompletionMessage.Role 类型是 constant.Assistant，
		// 不是 string，零值即 "assistant"，无需也不能显式赋值字符串。
		resp = &openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: string(textbytes),
					},
				},
			},
		}
		msgClipboardUsed = true
	} else if len(req.Messages) > 0 {
		if model.Client == nil {
			return fmt.Errorf("model %q has no initialized client", model.Name)
		}
		resp, err = model.Client.Chat.Completions.New(reqCtx, req, reqOpts...)
	} else {
		return fmt.Errorf("no messages in request")
	}

	if err != nil {
		fmt.Println("Error creating chat completion:", err)
		if len(req.Messages) > 0 {
			fmt.Println("req:", getMessageText(req.Messages[0]))
		}
		return err
	}
	if !msgClipboardUsed {
		model.ResponseTime(time.Since(timestart))
	}
	if model.PrintMsg {
		fmt.Println("resp:", resp)
	}

	if resp == nil || len(resp.Choices) == 0 {
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
		// 与原版一致：CallBack 的错误不阻塞工具执行
		_ = a.CallBack(reqCtx, content)
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

// getMessageText 从 ChatCompletionMessageParamUnion 中提取一段可读文本，
// 仅用于日志/复制到剪贴板等非语义场景。多模态内容会被简单拼接。
func getMessageText(m openai.ChatCompletionMessageParamUnion) string {
	switch {
	case m.OfSystem != nil:
		if m.OfSystem.Content.OfString.Valid() {
			return m.OfSystem.Content.OfString.Value
		}
	case m.OfUser != nil:
		if m.OfUser.Content.OfString.Valid() {
			return m.OfUser.Content.OfString.Value
		}
	case m.OfAssistant != nil:
		if m.OfAssistant.Content.OfString.Valid() {
			return m.OfAssistant.Content.OfString.Value
		}
	case m.OfDeveloper != nil:
		if m.OfDeveloper.Content.OfString.Valid() {
			return m.OfDeveloper.Content.OfString.Value
		}
	case m.OfTool != nil:
		if m.OfTool.Content.OfString.Valid() {
			return m.OfTool.Content.OfString.Value
		}
	}
	return ""
}
