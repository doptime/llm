package llm

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mroth/weightedrand"
	"github.com/spf13/afero"
)

// 模块结构体（简化版）
type Module struct {
	Name     string // 模块名称
	BranchId string // 模块Id

	EloScore  int64   // 模块评分
	Milestone float64 // 1: file/code constructed, 2:file/code tested, 3:hardware constructed, 4:hardware tested, 5:Income generated

	ProblemToSolve []string // 模块所属问题域
	DesignIdeas    []string
	OuterModuleIds []string
	InnerModuleIds []string
}

func (m *Module) SourceCodes() (fileList []string) {
	fs := afero.NewOsFs()
	files, _ := afero.ReadDir(fs, "./"+m.BranchId)
	for _, file := range files {
		content, _ := afero.ReadFile(fs, "./"+m.BranchId+"/"+file.Name())
		fileList = append(fileList, "file-name:\n"+file.Name()+"\ncontent:\n"+string(content))
	}
	return fileList
}

func (m Module) Id() string {
	return m.BranchId
}

func (m Module) Rating(delta int) int {
	return int(m.EloScore) + delta
}

func LoadbalancedPick(models ...*Model) *Model {
	if len(models) == 0 {
		return nil
	}
	if len(models) == 1 {
		return models[0]
	}
	Choices := make([]weightedrand.Choice, 0, len(models))
	for _, model := range models {
		avg, rpm := model.Stats() // 使用并发安全的 Stats() 方法
		// 钳制 avg 在 [0.1s, 60s]，防止冷启动 600s 把模型饿死，也防止 0s 算到无穷
		s := avg.Seconds()
		if s < 0.1 {
			s = 0.1
		}
		if s > 60 {
			s = 60
		}
		w := uint(50000 / (s + math.Sqrt(rpm+1)))
		if w == 0 {
			w = 1
		}
		Choices = append(Choices, weightedrand.Choice{Item: model, Weight: w})
	}
	ModelPicker, err := weightedrand.NewChooser(Choices...)
	if err != nil {
		return models[0]
	}
	return ModelPicker.Pick().(*Model)
}

type ModelList struct {
	Name         string
	SelectCursor int
	Models       []*Model
	mutex        sync.Mutex // 保护 SelectCursor 的并发访问
}

var EloModels = ModelList{
	Name: "EloModels",
	Models: []*Model{
		Qwen3Next80B,
		Qwen36_35ba3b,
	},
}

func NewModelList(name string, models ...*Model) *ModelList {
	return &ModelList{
		Name:   name,
		Models: models,
	}
}

func (list *ModelList) SequentialPick(firstToStart ...*Model) (ret *Model) {
	if len(list.Models) == 0 {
		panic("no models defined for list")
	}
	list.mutex.Lock()
	defer list.mutex.Unlock()
	if list.SelectCursor == 0 && len(firstToStart) > 0 {
		idx := slices.Index(list.Models, firstToStart[0])
		if idx >= 0 {
			list.SelectCursor = idx
		}
	}
	ret = list.Models[list.SelectCursor%len(list.Models)]
	list.SelectCursor++
	return ret
}

var lastPrintUnix int64 // 使用 atomic 替代 mutex，避免热路径上的锁竞争

// PrintAverageResponseTime 使用 atomic CAS 替代全局 mutex，不阻塞模型选择热路径
func PrintAverageResponseTime() {
	now := time.Now().Unix()
	last := atomic.LoadInt64(&lastPrintUnix)

	if now-last < 10 {
		return
	}

	// CAS 确保只有一个 goroutine 执行打印
	if atomic.CompareAndSwapInt64(&lastPrintUnix, last, now) {
		for _, model := range EloModels.Models {
			avg, rpm := model.Stats()
			fmt.Printf("Model %s: avg=%v rpm=%.2f\n", model.Name, avg, rpm)
		}
	}
}

func (list *ModelList) SelectOne(policy string) *Model {
	if len(list.Models) == 0 {
		return nil
	}
	PrintAverageResponseTime()
	// Calculate weights for each model
	weights := make([]float64, len(list.Models))
	var sum float64
	fastestIndex := 0
	fastestResponseTime := int64(99999999999)
	for i, model := range list.Models {
		avgTime, _ := model.Stats()
		avgUs := avgTime.Microseconds()
		if avgUs <= 0 {
			avgUs = 1 // 防护：避免除零 panic
		}
		if avgUs < fastestResponseTime {
			fastestResponseTime = avgUs
			fastestIndex = i
		}
		weights[i] = math.Sqrt(1.0 / float64(avgUs))
		sum += weights[i]
	}

	switch policy {
	case "random":
		randNum := rand.Float64()
		var cumulativeWeight float64

		for i, weight := range weights {
			cumulativeWeight += (weight / sum)
			if randNum < cumulativeWeight {
				return list.Models[i]
			}
		}
		fmt.Println("No model selected! use last model")
		return list.Models[len(list.Models)-1]

	case "roundrobin":
		list.mutex.Lock()
		selectIndex := list.SelectCursor % len(list.Models)
		list.SelectCursor++
		list.mutex.Unlock()

		// 严格轮询；加权随机由 "random" 分支承载
		_ = fastestIndex
		return list.Models[selectIndex]
	}
	return list.Models[0]
}
