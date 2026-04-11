# 改进变更日志

## 已接受的改进（按优先级排序）

### P0 — 必须修复的 bug

#### 1. `Create` 丢弃 `UseTools` 返回值 ⭐⭐⭐⭐⭐
**文件**: `agent.go`
**问题**: `UseTools` 返回 Clone 副本，但 `Create` 丢弃了返回值，导致创建的 Agent 完全没有 tools。这是当前最严重的隐藏 bug。
**修复**: 将 `UseTools` 改回原地修改（mutator 风格），与 `WithModels`、`WithCallback` 等保持一致。同时回撤了"使用 Clone 确保不污染原 Agent 的 map"这个改动——真正需要 Clone 的场景应由调用方显式 `a.Clone().UseTools(...)`。

#### 2. `WithToolcallParser` 自定义 parser 完全失效 ⭐⭐⭐⭐⭐
**文件**: `toolcall.go`
**问题**: 传入自定义 parser 时不 append，参数白传；`parse = ToolcallParserDefault` 赋值后从未使用。
**修复**: 统一逻辑——nil 时用默认，非 nil 时 append 自定义 parser。

#### 3. `parseOneToolcall` 类型断言 panic ⭐⭐⭐⭐⭐
**文件**: `toolcall.go`
**问题**: `tool.Arguments.(map[string]any)` 裸断言，`json.Unmarshal` 到 `interface{}` 可能得到 string/float64/[]any；空串时 `toolcallString[:len-1]` 也会 panic。
**修复**: 用 `tryUnmarshal` 辅助函数 + 安全类型检查 `args, ok := tool.Arguments.(map[string]any)`，消除所有 panic 路径。

#### 4. 工具调用去重逻辑完全失效 ⭐⭐⭐⭐
**文件**: `agent.go`
**问题**: (a) 漏了 `ToolCallHash[hash] = true`，去重从不生效；(b) 仅用 Arguments hash 做 key，不同工具同参数会被误吞。
**修复**: 改用 `(name, hash)` 联合键 + `seen` map 正确记录已见条目。

#### 5. `params` map 并发污染 + SharedMemory 合并方向反了 ⭐⭐⭐⭐⭐
**文件**: `agent.go`
**问题**: (a) 直接使用调用方的 map 引用，并发 Call 会互相污染；(b) SharedMemory 覆盖 params，优先级反了（本次入参应优先）；(c) `params["Params"] = params` 自引用循环，JSON 序列化会 stack overflow。
**修复**: 创建本地 `params` 副本；先铺 SharedMemory 再覆盖 caller 入参；移除自引用。回写结果时同时写 `params` 和 `caller`。

### P1 — 生产稳定性改进

#### 6. `ToolCallRunningMutext` 强类型化 ⭐⭐
**文件**: `agent.go`
**问题**: `interface{}` 类型存储 `*sync.Mutex`，运行时断言可能 panic；拼写错误 `Mutext`。
**修复**: 改为 `ToolCallRunningMutex *sync.Mutex`，消除类型断言 panic。方法名改为 `WithToolCallMutexRun`。

#### 7. `Messege` 拼写 + 吞错误 ⭐⭐
**文件**: `agent.go`
**问题**: 方法名拼写错误；第一个分支模板执行的 error 被丢弃。
**修复**: 新增 `Message(params) (string, error)` 正确签名；保留 `Messege` 作为 deprecated 兼容 alias。`Call` 内改用 `Message` 并检查 error。

#### 8. `ModelList.SelectCursor` data race ⭐⭐⭐⭐
**文件**: `modelPick.go`
**问题**: `SelectOne` 和 `SequentialPick` 直接读写 `SelectCursor` 无锁，高并发必现 race。
**修复**: 在 `ModelList` 中新增 `mutex sync.Mutex`，`SequentialPick` 和 `SelectOne` 的 roundrobin 分支加锁。

#### 9. `SelectOne` 除零 panic 防护 ⭐⭐⭐
**文件**: `modelPick.go`
**问题**: `avgTime.Microseconds()` 为 0 时 `1/float64(0)` 产生 Inf，但整数除法路径可能 panic。
**修复**: `avgUs <= 0` 时钳制为 1。

#### 10. `PrintAverageResponseTime` 全局锁竞争 ⭐⭐⭐
**文件**: `modelPick.go`
**问题**: 每次 `SelectOne` 都要获取全局 mutex 检查时间，高吞吐时成为瓶颈。
**修复**: 改用 `atomic.CompareAndSwapInt64`，无锁化。

#### 11. `ToolcallParserDefault` 字符串替换链低效 ⭐⭐⭐
**文件**: `toolcall.go`
**问题**: 10+ 次 `strings.ReplaceAll` 每次分配新字符串，GC 压力随 QPS 线性上升。
**修复**: 使用包级 `strings.NewReplacer` 预编译，单遍替换。保留了原有的解析流程和兜底逻辑，只优化了字符串处理效率。

#### 12. 工具回调错误被静默吞噬 ⭐⭐⭐⭐
**文件**: `agent.go`
**问题**: 工具调用失败仅 `fmt.Printf`，不返回错误，调用方无法感知。
**修复**: 用 `errors.Join` 聚合所有工具错误，Call 返回聚合错误。

#### 13. `HandleCallback` 字段回写用 Go 名而非 json tag ⭐⭐⭐
**文件**: `tool.go`
**问题**: 用 `t.Field(i).Name`（Go 字段名）写回 memory，但模板和参数系统用 json tag 名，导致不一致。
**修复**: 改用 `getFieldName(tt.Field(i))` 统一使用 json tag。

#### 14. `SequentialPick` 索引计算修正 ⭐
**文件**: `modelPick.go`
**问题**: `slices.Index` 返回 -1 时 `-1 + len(models)` 掩盖了"找不到"的错误。
**修复**: 显式检查 `idx >= 0` 再赋值。

#### 15. `SelectOne` roundrobin 简化 ⭐⭐
**文件**: `modelPick.go`
**问题**: 10% 偏向最快模型的逻辑语义不清，与 roundrobin 命名矛盾。
**修复**: roundrobin 改为严格轮询。加权随机已有 "random" 分支。

#### 16. `Call` 兜底 context 超时 ⭐⭐⭐
**文件**: `agent.go`
**问题**: 调用方忘传 context 时请求永不超时。
**修复**: 检测无 deadline 时自动加 30 分钟硬上限。

---

## 未接受的改进及理由

| 建议 | 理由 |
|---|---|
| **移除剪贴板功能** | 这是作者本地工作流的核心组件（配合 DeepSeek/Gemini 中转使用），移除会破坏其使用场景。保留原样。 |
| **`Call` 签名改为 `Call(ctx context.Context, ...)`** | Breaking API change，影响所有调用方。已通过 params["Context"] + 兜底超时缓解。 |
| **HTTP 客户端超时从 3600s 改小** | 本地 vLLM 长 reasoning 确实可能跑很久。`ResponseHeaderTimeout` 的建议有道理，但改动风险高，暂不动。 |
| **`omitempty` 不再作为 optional 信号** | 既有 tools 可能依赖此行为，是 breaking change。defer。 |
| **完整正则重写 `ToolcallParserDefault`** | 风险太高，`strings.NewReplacer` 已获得 80% 收益。保留原解析流程的兜底能力。 |
| **剪贴板超时包装** | 低优先级，仅影响无头服务器场景，而框架明确面向本地桌面使用。 |
| **`avgResponseTime` 初值 600s→5s** | clamp 已经处理，实际影响仅冷启动 1-2 个请求。 |
| **RPM 公式重构** | 当前公式可用，重构收益低。 |
| **Module 移到独立文件** | 纯 cosmetic，不影响功能。 |
| **GLM 模型字符串匹配改配置** | 好想法但低优先级，当前仅一个模型需要特殊模板。 |
| **`NewModel` 文档注释** | 合理但纯文档工作，本次聚焦代码改进。 |

---

## 未修改的文件

- **model.go** — 无需改动，已有的修复（Stats()、lastReceived 初值等）均正确。
- **toolcall-in-msg.go** — 无需改动，text/template 修复和 SetEscapeHTML(false) 均正确。
- **utils.go** — 无需改动。
