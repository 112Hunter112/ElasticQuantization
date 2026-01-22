package sketch

import (
	"hash/fnv"
	"math"
)

type CountMinSketch struct {
	width      int
	depth      int
	counters   [][]int64
	totalCount int64
}

func NewCountMinSketch(width, depth int) *CountMinSketch {
	counters := make([][]int64, depth)
	for i := range counters {
		counters[i] = make([]int64, width)
	}

	return &CountMinSketch{
		width:    width,
		depth:    depth,
		counters: counters,
	}
}

func (cms *CountMinSketch) Add(key string, count int64) {
	for i := 0; i < cms.depth; i++ {
		hash := cms.hash(key, i)
		idx := hash % uint64(cms.width)
		cms.counters[i][idx] += count
	}
	cms.totalCount += count
}

func (cms *CountMinSketch) Query(key string) int64 {
	min := int64(math.MaxInt64)

	for i := 0; i < cms.depth; i++ {
		hash := cms.hash(key, i)
		idx := hash % uint64(cms.width)
		if cms.counters[i][idx] < min {
			min = cms.counters[i][idx]
		}
	}

	return min
}

func (cms *CountMinSketch) TotalCount() int64 {
	return cms.totalCount
}

func (cms *CountMinSketch) hash(key string, seed int) uint64 {
	h := fnv.New64a()
	h.Write([]byte(key))
	h.Write([]byte{byte(seed)})
	return h.Sum64()
}
