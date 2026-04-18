package llm

import (
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// Model represents an OpenAI-compatible model with its associated client and config.
//
// 注意：与旧版 sashabaranov/go-openai 不同，新版 openai-go 的 Client 不是指针类型，
// 而是值类型 openai.Client。我们这里持有一份指针副本，便于 nil 判断与零值处理。
type Model struct {
	Client          *openai.Client
	ApiKey          string // API key for authentication
	SystemMessage   string
	ExtraBody       map[string]any // 透传到底层 HTTP body 的额外字段（vLLM/Qwen 等需要）
	BaseURL         string         // Base URL for the OpenAI-compatible API
	Name            string
	TopP            float64
	TopK            float64 // 仅记录用，OpenAI 标准 API 不接受 top_k；如需透传请放入 ExtraBody
	Temperature     float64
	ToolInPrompt    *ToolInPrompt
	avgResponseTime time.Duration
	lastReceived    time.Time
	PrintMsg        bool
	requestPerMin   float64
	mutex           sync.RWMutex
}

func (model *Model) ResponseTime(duration ...time.Duration) time.Duration {
	if len(duration) == 0 {
		// 修复：无参读取必须加读锁，防止 data race
		model.mutex.RLock()
		defer model.mutex.RUnlock()
		return model.avgResponseTime
	}
	model.mutex.Lock()
	defer model.mutex.Unlock()

	// 使用 float64 进行平滑计算，消除中途类型转换导致的精度截断
	alpha := 0.1
	currentAvg := float64(model.avgResponseTime)
	newVal := float64(duration[0])
	model.avgResponseTime = time.Duration((newVal * alpha) + (currentAvg * (1.0 - alpha)))

	timeSinceLast := time.Since(model.lastReceived).Microseconds()
	if timeSinceLast <= 0 {
		timeSinceLast = 100 // 防止除零及负时间异常
	}
	model.requestPerMin += (60000000.0/float64(timeSinceLast+100) - model.requestPerMin) * 0.01
	model.lastReceived = time.Now()
	return model.avgResponseTime
}

// Stats 提供并发安全的批量读取，避免调用方分别读 avgResponseTime 和 requestPerMin 时的锁重入问题
func (model *Model) Stats() (avg time.Duration, rpm float64) {
	model.mutex.RLock()
	defer model.mutex.RUnlock()
	return model.avgResponseTime, model.requestPerMin
}

// NewModel initializes a new Model with the given baseURL, apiKey env var name, and modelName.
//
// apiKey 参数若是环境变量名（os.Getenv 命中），会被替换成对应的真实 key。
// 否则原样作为 API key 使用。这与旧实现保持一致。
func NewModel(baseURL, apiKey, modelName string) *Model {
	if envKey := os.Getenv(apiKey); envKey != "" {
		apiKey = envKey
	}

	httpClient := &http.Client{
		Timeout: 3600 * time.Second, // 整个请求的总超时时间，包括连接和接收响应
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   3600 * time.Second,
				KeepAlive: 3600 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   30 * time.Second,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          100,
			IdleConnTimeout:       3600 * time.Second,
			ExpectContinueTimeout: 3600 * time.Second,
			MaxIdleConnsPerHost:   100,
			DisableKeepAlives:     false,
		},
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(httpClient),
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}

	client := openai.NewClient(opts...)

	model := &Model{
		Client:          &client,
		Name:            modelName,
		ApiKey:          apiKey,
		BaseURL:         baseURL,
		avgResponseTime: 600 * time.Second,
		lastReceived:    time.Now(), // 修复：避免首次 time.Since(零值) 导致 requestPerMin 计算溢出
	}
	model.RegisterToMap()

	return model
}

func (m *Model) RegisterToMap() *Model {
	ModelsMap[m.Name] = m
	return m
}

func (m *Model) WithToolsInSystemPrompt() *Model {
	m.ToolInPrompt = &ToolInPrompt{InSystemPrompt: true}
	return m
}

func (m *Model) WithToolsInUserPrompt() *Model {
	m.ToolInPrompt = &ToolInPrompt{InUserPrompt: true}
	return m
}

func (m *Model) WithTopP(topP float64) *Model {
	m.TopP = topP
	return m
}

func (m *Model) WithTopK(topK float64) *Model {
	m.TopK = topK
	return m
}

func (m *Model) WithTemperature(temperature float64) *Model {
	m.Temperature = temperature
	return m
}

func (m *Model) WithSysPrompt(message string) *Model {
	m.SystemMessage = message
	return m
}

// WithExtraBody 设置透传到 HTTP body 的额外字段。
// 用法示例：
//
//	model.WithExtraBody(map[string]any{
//	    "chat_template_kwargs": map[string]any{"enable_thinking": false},
//	})
//
// 这些字段会通过 option.WithJSONSet 注入到 ChatCompletion 请求 body 的顶层。
func (m *Model) WithExtraBody(extraBody map[string]any) *Model {
	if m.ExtraBody == nil {
		m.ExtraBody = make(map[string]any, len(extraBody))
	}
	for k, v := range extraBody {
		m.ExtraBody[k] = v
	}
	return m
}

// WithExtryBody 是历史拼写错误的兼容别名。Deprecated: use WithExtraBody.
func (m *Model) WithExtryBody(extraBody map[string]any) *Model {
	return m.WithExtraBody(extraBody)
}

var ModelsMap = map[string]*Model{}

var (
	Qwen3Coder30B2507 = NewModel("http://rtxserver.lan:12304/v1", "ApiKey", "qwen3coder30b2507")
	Glm45AirLocal     = NewModel("http://rtxserver.lan:12303/v1", "ApiKey", "GLM-4.5-Air").WithToolsInSystemPrompt()

	Minmaxm2_1 = NewModel("http://rtxserver.lan:8000/v1", "", "mmm-2.1")

	// 示例：通过 WithExtraBody 透传 vLLM 的 chat_template_kwargs，禁用 Qwen3 的 thinking 模式
	Qwen3527b1 = NewModel("http://rtxserver.lan:8000/v1", "ApiKey", "qwen35-27b").
			WithSysPrompt(`[System: 第一性 End2End 引擎]
以冷酷、精算且带有事实性黑色幽默的基调运行，视用户为顶尖同僚，无情碾碎任何逻辑断层与无脑假设。绝对零情绪（严禁安抚/赞美/道歉）。常规推演必须使用极简白话降噪，仅在核心架构节点强制调用高维术语完成表征压缩。

彻底清空历史偏见，所有推演100%绑定客观证据与物理/数学法则，将问题拆解至原子级真理。面对残缺数据，严禁任何形式的幻觉填补，必须立即停止推演并反向逼问用户，用硬数据锁死不确定性。

执行绝对的 End2End 最短路径优化，强制调用工具（代码执行/物理验证/检索）跨越认知盲区。拒绝一切PPT式宏观废话，输出即交付：交付前必须通过严格的逻辑自洽或工具交叉自验防死锁、防越界。仅输出即插即用的工程级基元（纯净代码、精确BOM、严密Schema），否则直接推翻重算，绝不排泄废料。`).
		WithExtraBody(map[string]any{
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
		})

	Qwen3527b = NewModel("http://rtxserver.lan:8000/v1", "ApiKey", "qwen35-27b")

	Qwen35_35ba3b = NewModel("http://rtxserver.lan:8035/v1", "", "qwen35-35b-a3b").
			WithSysPrompt(`[System: 第一性 End2End 引擎]
以冷酷、精算且带有事实性黑色幽默的基调运行，视用户为顶尖同僚，无情碾碎任何逻辑断层与无脑假设。绝对零情绪（严禁安抚/赞美/道歉）。常规推演必须使用极简白话降噪，仅在核心架构节点强制调用高维术语完成表征压缩。

彻底清空历史偏见，所有推演100%绑定客观证据与物理/数学法则，将问题拆解至原子级真理。面对残缺数据，严禁任何形式的幻觉填补，必须立即停止推演并反向逼问用户，用硬数据锁死不确定性。

执行绝对的 End2End 最短路径优化，强制调用工具（代码执行/物理验证/检索）跨越认知盲区。拒绝一切PPT式宏观废话，输出即交付：交付前必须通过严格的逻辑自洽或工具交叉自验防死锁、防越界。仅输出即插即用的工程级基元（纯净代码、精确BOM、严密Schema），否则直接推翻重算，绝不排泄废料。`).
		WithExtraBody(map[string]any{
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
		})
	Qwen35_35ba3bNonthining = NewModel("http://rtxserver.lan:8035/v1", "", "qwen35-35b-a3b").WithExtraBody(map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}})

	// Qwen3Next80B 在 modelPick.go 的 EloModels 中被引用，原项目可能定义在外部文件，
	// 这里补一个占位定义以保证编译通过；实际 BaseURL/模型名按部署调整。
	Qwen3Next80B = NewModel("http://rtxserver.lan:8000/v1", "ApiKey", "qwen3-next-80b")

	ModelDefault = Qwen3527b
)
