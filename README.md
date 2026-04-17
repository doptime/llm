# LLM Agent Framework (Golang)

这是一个轻量级的 Golang LLM 交互框架，旨在通过 **模板驱动 (Template-driven)** 和 **工具化 (Tool-calling)** 实现结构化的 Agent 逻辑。它非常适合用于构建类似 `AlphaEvolve` 这种需要高度精准、闭环反馈的代码演进系统。

## 🚀 快速开始

### 1. 安装依赖

在你的 Go 项目中引入核心包：

```bash
go get github.com/doptime/llm
```

### 2. 核心示例 (`demo.go`)

以下是一个模拟 `AlphaEvolve` 中 **Critic (裁决大脑)** 的逻辑实现：

```go
package main

import (
	"fmt"
	"log"
	"strings"
	"text/template"

	"github.com/doptime/llm"
)

// VerdictPayload 定义了 LLM 返回的结构化裁决结果
type VerdictPayload struct {
	CognitiveScore  float64  `json:"cognitive_score" jsonschema:"description=逻辑与代码健壮性评分"`
	PhysicsScore    float64  `json:"physics_score" jsonschema:"description=物理模拟或性能表现评分"`
	FocusDimension  string   `json:"focus_dimension" jsonschema:"enum=Cognitive,enum=Physics,description=当前最需要突破的维度"`
	ActionableTodos []string `json:"actionable_todos" jsonschema:"description=具体的代码修改待办事项"`
}

func main() {
	// 1. 声明接收变量
	var verdict VerdictPayload

	// 2. 注册 LLM Tool：将函数调用与结构体解构绑定
	evalTool := llm.NewTool(
		"SubmitVerdict",
		"提交代码质量裁决与进化方案",
		func(p *VerdictPayload) { verdict = *p },
	)

	// 3. 定义 Prompt 模板
	tpl := template.Must(template.New("critic").Parse(`
你现在是系统的 Critic (大脑)。你的任务是评估代码并规划进化方向。

【目标代码】:
{{.CodeBlock}}

【执行日志】:
{{.ExecLogs}}

⚠️ 严格调用 SubmitVerdict 输出 JSON。禁止输出任何解释性文本。
`))

	// 4. 初始化 Agent
	// 注意：此处使用了 github.com/doptime/llm 包
	judgeAgent := llm.NewAgent(tpl).
		UseTools(evalTool).
		UseModels(llm.ModelDefault)

	// 5. 执行调用
	fmt.Println("🛰️  正在解析代码并生成进化策略...")
	err := judgeAgent.Call(map[string]any{
		"CodeBlock": `function update() { ball.x += speed; // 缺少重力逻辑 }`,
		"ExecLogs":  `[Warning] Ball floating without gravity component.`,
	})

	if err != nil {
		log.Fatalf("❌ 调用失败: %v", err)
	}

	// 6. 结果展示
	fmt.Printf("\n🏆 进化方向分析完成!\n")
	fmt.Println(strings.Repeat("-", 30))
	fmt.Printf("突破维度: %s\n", verdict.FocusDimension)
	fmt.Printf("综合评分: Cognitive(%.1f) / Physics(%.1f)\n", verdict.CognitiveScore, verdict.PhysicsScore)
	fmt.Println("后续待办:")
	for _, todo := range verdict.ActionableTodos {
		fmt.Printf(" - %s\n", todo)
	}
}
```

## 🛠️ 核心特性

* **强类型输出**：利用 `jsonschema` 标签强制 LLM 遵循预定义的 JSON 结构。
* **工具绑定**：通过 `llm.NewTool` 简单地将 LLM 输出映射到本地 Go 结构体。
* **模板驱动**：使用标准的 `text/template` 语法，方便动态注入复杂的上下文（如代码区块、AST 状态）。
* **环境隔离**：支持通过 `WithModels` 快速切换底层推理引擎（如 Qwen 系列、GPT 等）。

---