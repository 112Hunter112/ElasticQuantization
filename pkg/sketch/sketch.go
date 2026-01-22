package sketch

import (
	"sync"
)

type SketchConfig struct {
	HLLPrecision  int
	CMSketchWidth int
	CMSketchDepth int
}

type SketchAggregator struct {
	hll      *HyperLogLogPP
	cmSketch *CountMinSketch
	mu       sync.RWMutex
	config   SketchConfig
}

func NewSketchAggregator(config SketchConfig) *SketchAggregator {
	return &SketchAggregator{
		hll:      NewHyperLogLogPP(config.HLLPrecision),
		cmSketch: NewCountMinSketch(config.CMSketchWidth, config.CMSketchDepth),
		config:   config,
	}
}

func (sa *SketchAggregator) AddToHLL(table, id string) {
	sa.mu.Lock()
	defer sa.mu.Unlock()
	sa.hll.Add(table + ":" + id)
}

func (sa *SketchAggregator) AddToCMSketch(key string, count int64) {
	sa.mu.Lock()
	defer sa.mu.Unlock()
	sa.cmSketch.Add(key, count)
}

func (sa *SketchAggregator) GetUniqueCount() uint64 {
	sa.mu.RLock()
	defer sa.mu.RUnlock()
	return sa.hll.Count()
}

func (sa *SketchAggregator) GetFrequency(key string) int64 {
	sa.mu.RLock()
	defer sa.mu.RUnlock()
	return sa.cmSketch.Query(key)
}

func (sa *SketchAggregator) GetStats() SketchStats {
	sa.mu.RLock()
	defer sa.mu.RUnlock()

	return SketchStats{
		UniqueCount: sa.hll.Count(),
		TotalEvents: sa.cmSketch.TotalCount(),
	}
}

type SketchStats struct {
	UniqueCount uint64
	TotalEvents int64
}
