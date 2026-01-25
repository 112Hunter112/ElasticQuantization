package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/yourusername/consistency-auditor/pkg/quantizer"
	"github.com/yourusername/consistency-auditor/pkg/sketch"
)

const (
	numItems     = 100_000
	vectorDims   = 128
	reducedDims  = 32
	numVectors   = 1000
	topK         = 10
)

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

func getMemUsage() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc
}

func TestMemoryFootprint(t *testing.T) {
	// Garbage collect to start clean
	runtime.GC()
	startMem := getMemUsage()

	// 1. Standard Map
	stdMap := make(map[string]int)
	for i := 0; i < numItems; i++ {
		key := fmt.Sprintf("key-%d", i)
		stdMap[key] = i
	}
	
	runtime.GC() // GC to clean up temporary strings if possible, but keep map
	mapMem := getMemUsage() - startMem

	// Clean up map
	stdMap = nil
	runtime.GC()
	startMemSketch := getMemUsage()

	// 2. CountMinSketch
	// Width 2000, Depth 5 is reasonable for heavy hitters approximation
	cms := sketch.NewCountMinSketch(2000, 5)
	for i := 0; i < numItems; i++ {
		key := fmt.Sprintf("key-%d", i)
		cms.Add(key, int64(i))
	}

	runtime.GC()
	cmsMem := getMemUsage() - startMemSketch

	fmt.Printf("\n=== Memory Footprint Benchmark (N=%d) ===\n", numItems)
	fmt.Printf("Standard Map[string]int: %d MB\n", bToMb(mapMem))
	fmt.Printf("CountMinSketch:          %d MB\n", bToMb(cmsMem))

	if mapMem > 0 {
		var savings float64
		if mapMem > cmsMem {
			savings = float64(mapMem-cmsMem) / float64(mapMem) * 100
		} else {
			savings = 0 // No savings or measurement noise
		}
		fmt.Printf("Savings:                 %.2f%%\n", savings)
	} else {
		fmt.Println("Savings:                 N/A (Map memory too low)")
	}
}

func TestRecall(t *testing.T) {
	// Generate random vectors
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float64, vectorDims)
		for j := 0; j < vectorDims; j++ {
			vectors[i][j] = rand.Float64()
		}
	}

	// Train PCA Quantizer
	pq := quantizer.NewPCAQuantizer(vectorDims, reducedDims, 8)
	if err := pq.Fit(vectors); err != nil {
		t.Fatalf("Failed to fit quantizer: %v", err)
	}

	// Generate Query Vector
	query := make([]float64, vectorDims)
	for j := 0; j < vectorDims; j++ {
		query[j] = rand.Float64()
	}

	// 1. Exact Search (Brute Force on Raw Vectors)
	exactIndices := bruteForceSearch(query, vectors, topK)

	// 2. Quantized Search
	// Reconstruct all vectors
	reconstructed := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		qBytes, _ := pq.Quantize(vectors[i])
		rec, _ := pq.Dequantize(qBytes)
		reconstructed[i] = rec
	}
	
	quantizedIndices := bruteForceSearch(query, reconstructed, topK)

	// Calculate Recall
	recall := calculateRecall(exactIndices, quantizedIndices)

	fmt.Printf("\n=== Search Accuracy (Recall@%d) ===\n", topK)
	fmt.Printf("Dimensions: %d -> %d\n", vectorDims, reducedDims)
	fmt.Printf("Recall:     %.2f%%\n", recall*100)
}

func bruteForceSearch(query []float64, dataset [][]float64, k int) []int {
	type result struct {
		index int
		dist  float64
	}
	results := make([]result, len(dataset))

	for i, vec := range dataset {
		results[i] = result{index: i, dist: euclideanDistance(query, vec)}
	}

	// Simple sort
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].dist > results[j].dist {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	indices := make([]int, k)
	for i := 0; i < k && i < len(results); i++ {
		indices[i] = results[i].index
	}
	return indices
}

func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func calculateRecall(groundTruth, predicted []int) float64 {
	match := 0
	gtMap := make(map[int]bool)
	for _, idx := range groundTruth {
		gtMap[idx] = true
	}

	for _, idx := range predicted {
		if gtMap[idx] {
			match++
		}
	}
	return float64(match) / float64(len(groundTruth))
}
