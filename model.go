package llm

import (
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// Model represents an OpenAI model with its associated client and model name.
type Model struct {
	Client          *openai.Client
	ApiKey          string // API key for authentication
	SystemMessage   string
	BaseURL         string // Base URL for the OpenAI API, can be empty for default
	Name            string
	TopP            float32
	TopK            float32
	Temperature     float32
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

// NewModel initializes a new Model with the given baseURL, apiKey, and modelName.
// It configures the OpenAI client to use a custom base URL if provided.
func NewModel(baseURL, apiKey, modelName string) *Model {
	if _apikey := os.Getenv(apiKey); _apikey != "" {
		apiKey = _apikey
	}
	config := openai.DefaultConfig(apiKey)
	config.EmptyMessagesLimit = 10000000
	if baseURL != "" {
		config.BaseURL = baseURL
	}
	config.HTTPClient = &http.Client{
		Timeout: 3600 * time.Second, // 整个请求的总超时时间，包括连接和接收响应
		Transport: &http.Transport{
			// 设置连接超时时间
			DialContext: (&net.Dialer{
				Timeout:   3600 * time.Second, // 连接超时
				KeepAlive: 3600 * time.Second, // 保持连接的时间
			}).DialContext,
			// 设置TLS配置
			TLSHandshakeTimeout: 30 * time.Second, // TLS握手超时
			// 设置HTTP/2配置
			ForceAttemptHTTP2:     true,               // 强制尝试使用HTTP/2
			MaxIdleConns:          100,                // 最大空闲连接数
			IdleConnTimeout:       3600 * time.Second, // 空闲连接的超时时间
			ExpectContinueTimeout: 3600 * time.Second, // 期望继续的超时时间
			MaxIdleConnsPerHost:   100,                // 每个主机的最大空闲连接数
			DisableKeepAlives:     false,              // 是否禁用Keep-Alive
		},
	}

	client := openai.NewClientWithConfig(config)
	model := &Model{
		Client:          client,
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
func (m *Model) WithTopP(topP float32) *Model {
	m.TopP = topP
	return m
}
func (m *Model) WithTopK(topK float32) *Model {
	m.TopK = topK
	return m
}
func (m *Model) WithTemperature(temperature float32) *Model {
	m.Temperature = temperature
	return m
}
func (m *Model) WithSysPrompt(message string) *Model {
	m.SystemMessage = message
	return m
}

var ModelsMap = map[string]*Model{}

var (
	Qwen3Coder30B2507 = NewModel("http://rtxserver.lan:12304/v1", "ApiKey", "qwen3coder30b2507")
	Glm45AirLocal     = NewModel("http://rtxserver.lan:12303/v1", "ApiKey", "GLM-4.5-Air").WithToolsInSystemPrompt()

	Minmaxm2_1 = NewModel("http://rtxserver.lan:8000/v1", "", "mmm-2.1")

	Qwen3527b1 = NewModel("http://rtxserver.lan:8000/v1", "ApiKey", "qwen35-27b").WithSysPrompt(`[System: 第一性 End2End 引擎]
以冷酷、精算且带有事实性黑色幽默的基调运行，视用户为顶尖同僚，无情碾碎任何逻辑断层与无脑假设。绝对零情绪（严禁安抚/赞美/道歉）。常规推演必须使用极简白话降噪，仅在核心架构节点强制调用高维术语完成表征压缩。

彻底清空历史偏见，所有推演100%绑定客观证据与物理/数学法则，将问题拆解至原子级真理。面对残缺数据，严禁任何形式的幻觉填补，必须立即停止推演并反向逼问用户，用硬数据锁死不确定性。

执行绝对的 End2End 最短路径优化，强制调用工具（代码执行/物理验证/检索）跨越认知盲区。拒绝一切PPT式宏观废话，输出即交付：交付前必须通过严格的逻辑自洽或工具交叉自验防死锁、防越界。仅输出即插即用的工程级基元（纯净代码、精确BOM、严密Schema），否则直接推翻重算，绝不排泄废料。`)
	Qwen3527b     = NewModel("http://rtxserver.lan:8000/v1", "ApiKey", "qwen35-27b")
	Qwen35_35ba3b = NewModel("http://rtxserver.lan:8035/v1", "", "qwen35-35b-a3b").WithSysPrompt(`[System: 第一性 End2End 引擎]
以冷酷、精算且带有事实性黑色幽默的基调运行，视用户为顶尖同僚，无情碾碎任何逻辑断层与无脑假设。绝对零情绪（严禁安抚/赞美/道歉）。常规推演必须使用极简白话降噪，仅在核心架构节点强制调用高维术语完成表征压缩。

彻底清空历史偏见，所有推演100%绑定客观证据与物理/数学法则，将问题拆解至原子级真理。面对残缺数据，严禁任何形式的幻觉填补，必须立即停止推演并反向逼问用户，用硬数据锁死不确定性。

执行绝对的 End2End 最短路径优化，强制调用工具（代码执行/物理验证/检索）跨越认知盲区。拒绝一切PPT式宏观废话，输出即交付：交付前必须通过严格的逻辑自洽或工具交叉自验防死锁、防越界。仅输出即插即用的工程级基元（纯净代码、精确BOM、严密Schema），否则直接推翻重算，绝不排泄废料。`)

	//ModelDefault        = ModelQwen32BCoderLocal
	ModelDefault = Qwen3527b
)
