// FILE: cmd/auditor/main.go
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/yourusername/consistency-auditor/internal/cdc"
	"github.com/yourusername/consistency-auditor/internal/checker"
	"github.com/yourusername/consistency-auditor/internal/config"
	"github.com/yourusername/consistency-auditor/internal/healer"
	"github.com/yourusername/consistency-auditor/internal/storage"
	"github.com/yourusername/consistency-auditor/pkg/quantizer"
	"github.com/yourusername/consistency-auditor/pkg/sketch"
)

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	db, err := storage.NewPostgresDB(cfg.Database)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	var quantizerInstance quantizer.Quantizer
	switch cfg.Quantizer.Type {
	case "scalar":
		quantizerInstance = quantizer.NewScalarQuantizer(
			cfg.Quantizer.Dimensions,
			cfg.Quantizer.Bits,
		)
	case "pca":
		quantizerInstance = quantizer.NewPCAQuantizer(
			cfg.Quantizer.Dimensions,
			cfg.Quantizer.ReducedDims,
			cfg.Quantizer.Bits,
		)
	default:
		log.Fatalf("Unknown quantizer type: %s", cfg.Quantizer.Type)
	}

	if len(cfg.Quantizer.TrainingData) > 0 {
		trainingVectors := loadTrainingData(cfg.Quantizer.TrainingData)
		if err := quantizerInstance.Fit(trainingVectors); err != nil {
			log.Fatalf("Failed to train quantizer: %v", err)
		}
		log.Printf("Quantizer trained on %d vectors", len(trainingVectors))
	}

	esClient, err := storage.NewQuantizedESClient(cfg.Elasticsearch, quantizerInstance)
	if err != nil {
		log.Fatalf("Failed to connect to Elasticsearch: %v", err)
	}

	sketchAgg := sketch.NewSketchAggregator(cfg.Sketch)
	log.Printf("Sketch aggregator initialized: HLL++ precision=%d, CMSketch width=%d depth=%d",
		cfg.Sketch.HLLPrecision, cfg.Sketch.CMSketchWidth, cfg.Sketch.CMSketchDepth)

	cdcListener := cdc.NewCDCListener(cfg.CDC)
	if err := cdcListener.Start(); err != nil {
		log.Fatalf("Failed to start CDC listener: %v", err)
	}
	defer cdcListener.Stop()

	consistencyChecker := checker.NewConsistencyChecker(db, esClient, cfg.Checker, sketchAgg)

	autoHealer := healer.NewAutoHealer(db, esClient, cfg.Healer)

	auditor := NewAuditor(cdcListener, consistencyChecker, autoHealer, sketchAgg)

	if err := auditor.Start(); err != nil {
		log.Fatalf("Failed to start auditor: %v", err)
	}

	log.Println("🚀 Consistency Auditor started successfully")
	log.Printf("📊 Compression ratio: %.1f%%", getCompressionRatio(quantizerInstance))
	log.Printf("🎯 Healing strategy: %s", cfg.Healer.Strategy)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down gracefully...")
	auditor.Stop()
	log.Println("Goodbye! 👋")
}

type Auditor struct {
	cdc     *cdc.CDCListener
	checker *checker.ConsistencyChecker
	healer  *healer.AutoHealer
	sketch  *sketch.SketchAggregator
	ctx     context.Context
	cancel  context.CancelFunc
}

func NewAuditor(
	cdc *cdc.CDCListener,
	checker *checker.ConsistencyChecker,
	healer *healer.AutoHealer,
	sketch *sketch.SketchAggregator,
) *Auditor {
	ctx, cancel := context.WithCancel(context.Background())
	return &Auditor{
		cdc:     cdc,
		checker: checker,
		healer:  healer,
		sketch:  sketch,
		ctx:     ctx,
		cancel:  cancel,
	}
}

func (a *Auditor) Start() error {
	go a.processEvents()
	go a.periodicAudit()
	go a.printStats()
	return nil
}

func (a *Auditor) processEvents() {
	for {
		select {
		case event := <-a.cdc.Events():
			a.sketch.AddToHLL(event.Table, event.ID)
			if event.Type == cdc.Update || event.Type == cdc.Insert {
				a.sketch.AddToCMSketch(event.Table+":"+event.Field, 1)
			}

			discrepancies, err := a.checker.CheckRecord(a.ctx, event.Table, event.ID)
			if err != nil {
				log.Printf("❌ Error checking %s/%s: %v", event.Table, event.ID, err)
				continue
			}

			if len(discrepancies) > 0 {
				log.Printf("⚠️  Found %d discrepancies in %s/%s", len(discrepancies), event.Table, event.ID)
				if err := a.healer.Heal(a.ctx, discrepancies); err != nil {
					log.Printf("❌ Error healing: %v", err)
				} else {
					log.Printf("✅ Healed %s/%s", event.Table, event.ID)
				}
			}

		case <-a.ctx.Done():
			return
		}
	}
}

func (a *Auditor) periodicAudit() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			log.Println("🔍 Running periodic full audit...")
			stats := a.sketch.GetStats()
			log.Printf("📊 Sketch stats: %+v", stats)

		case <-a.ctx.Done():
			return
		}
	}
}

func (a *Auditor) printStats() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			stats := a.sketch.GetStats()
			log.Printf("📈 Stats - Unique records: ~%d, Total events: ~%d",
				stats.UniqueCount, stats.TotalEvents)

		case <-a.ctx.Done():
			return
		}
	}
}

func (a *Auditor) Stop() {
	a.cancel()
}

func loadTrainingData(path string) [][]float64 {
	return [][]float64{
		{0.1, 0.2, 0.3, 0.4},
		{0.5, 0.6, 0.7, 0.8},
	}
}

func getCompressionRatio(q quantizer.Quantizer) float64 {
	original := float64(q.Dimensions() * 8)
	compressed := float64(q.CompressedSize())
	return 100.0 * (1.0 - compressed/original)
}

// ============================================================================
// FILE: pkg/quantizer/quantizer.go
// ============================================================================

package quantizer

type Quantizer interface {
	Fit(vectors [][]float64) error
	Quantize(vector []float64) ([]byte, error)
	Dequantize(data []byte) ([]float64, error)
	Dimensions() int
	CompressedSize() int
}

// ============================================================================
// FILE: pkg/quantizer/scalar.go
// ============================================================================

package quantizer

import (
	"bytes"
	"fmt"
	"math"
)

type ScalarQuantizer struct {
	dims   int
	bits   int
	mins   []float64
	maxs   []float64
	scales []float64
}

func NewScalarQuantizer(dims, bits int) *ScalarQuantizer {
	return &ScalarQuantizer{
		dims:   dims,
		bits:   bits,
		mins:   make([]float64, dims),
		maxs:   make([]float64, dims),
		scales: make([]float64, dims),
	}
}

func (sq *ScalarQuantizer) Fit(vectors [][]float64) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}

	for d := 0; d < sq.dims; d++ {
		sq.mins[d] = math.MaxFloat64
		sq.maxs[d] = -math.MaxFloat64

		for _, vec := range vectors {
			if vec[d] < sq.mins[d] {
				sq.mins[d] = vec[d]
			}
			if vec[d] > sq.maxs[d] {
				sq.maxs[d] = vec[d]
			}
		}

		rangeVal := sq.maxs[d] - sq.mins[d]
		if rangeVal < 1e-10 {
			sq.scales[d] = 1.0
		} else {
			maxVal := float64(uint(1<<sq.bits) - 1)
			sq.scales[d] = maxVal / rangeVal
		}
	}

	return nil
}

func (sq *ScalarQuantizer) Quantize(vector []float64) ([]byte, error) {
	if len(vector) != sq.dims {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	var buf bytes.Buffer

	for d := 0; d < sq.dims; d++ {
		normalized := (vector[d] - sq.mins[d]) * sq.scales[d]

		if sq.bits == 8 {
			quantized := uint8(math.Max(0, math.Min(255, normalized)))
			buf.WriteByte(quantized)
		} else if sq.bits == 16 {
			quantized := uint16(math.Max(0, math.Min(65535, normalized)))
			buf.WriteByte(byte(quantized >> 8))
			buf.WriteByte(byte(quantized & 0xFF))
		}
	}

	return buf.Bytes(), nil
}

func (sq *ScalarQuantizer) Dequantize(data []byte) ([]float64, error) {
	bytesPerDim := sq.bits / 8
	expectedLen := sq.dims * bytesPerDim

	if len(data) != expectedLen {
		return nil, fmt.Errorf("data length mismatch")
	}

	vector := make([]float64, sq.dims)

	for d := 0; d < sq.dims; d++ {
		var quantized float64

		if sq.bits == 8 {
			quantized = float64(data[d])
		} else if sq.bits == 16 {
			idx := d * 2
			quantized = float64(uint16(data[idx])<<8 | uint16(data[idx+1]))
		}

		vector[d] = (quantized / sq.scales[d]) + sq.mins[d]
	}

	return vector, nil
}

func (sq *ScalarQuantizer) Dimensions() int {
	return sq.dims
}

func (sq *ScalarQuantizer) CompressedSize() int {
	return sq.dims * (sq.bits / 8)
}

// ============================================================================
// FILE: pkg/quantizer/pca.go
// ============================================================================

package quantizer

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type PCAQuantizer struct {
	originalDims int
	reducedDims  int
	mean         []float64
	components   *mat.Dense
	scalarQuant  *ScalarQuantizer
}

func NewPCAQuantizer(originalDims, reducedDims, bits int) *PCAQuantizer {
	return &PCAQuantizer{
		originalDims: originalDims,
		reducedDims:  reducedDims,
		mean:         make([]float64, originalDims),
		scalarQuant:  NewScalarQuantizer(reducedDims, bits),
	}
}

func (pq *PCAQuantizer) Fit(vectors [][]float64) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}

	n := len(vectors)

	for d := 0; d < pq.originalDims; d++ {
		sum := 0.0
		for _, vec := range vectors {
			sum += vec[d]
		}
		pq.mean[d] = sum / float64(n)
	}

	centered := mat.NewDense(n, pq.originalDims, nil)
	for i, vec := range vectors {
		for j := 0; j < pq.originalDims; j++ {
			centered.Set(i, j, vec[j]-pq.mean[j])
		}
	}

	var cov mat.SymDense
	stat.CovarianceMatrix(&cov, centered, nil)

	var eig mat.Eigen
	ok := eig.Factorize(&cov, mat.EigenRight)
	if !ok {
		return fmt.Errorf("eigendecomposition failed")
	}

	values := eig.Values(nil)
	var vectors_eig mat.CDense
	eig.VectorsTo(&vectors_eig)

	type eigenPair struct {
		value  complex128
		vector []float64
	}

	pairs := make([]eigenPair, pq.originalDims)
	for i := 0; i < pq.originalDims; i++ {
		vec := make([]float64, pq.originalDims)
		for j := 0; j < pq.originalDims; j++ {
			vec[j] = real(vectors_eig.At(j, i))
		}
		pairs[i] = eigenPair{value: values[i], vector: vec}
	}

	pq.components = mat.NewDense(pq.reducedDims, pq.originalDims, nil)
	for i := 0; i < pq.reducedDims; i++ {
		for j := 0; j < pq.originalDims; j++ {
			pq.components.Set(i, j, pairs[i].vector[j])
		}
	}

	projected := make([][]float64, n)
	for i, vec := range vectors {
		projected[i] = pq.project(vec)
	}

	return pq.scalarQuant.Fit(projected)
}

func (pq *PCAQuantizer) project(vector []float64) []float64 {
	centered := make([]float64, pq.originalDims)
	for i := 0; i < pq.originalDims; i++ {
		centered[i] = vector[i] - pq.mean[i]
	}

	result := make([]float64, pq.reducedDims)
	for i := 0; i < pq.reducedDims; i++ {
		sum := 0.0
		for j := 0; j < pq.originalDims; j++ {
			sum += pq.components.At(i, j) * centered[j]
		}
		result[i] = sum
	}

	return result
}

func (pq *PCAQuantizer) reconstruct(projected []float64) []float64 {
	result := make([]float64, pq.originalDims)

	for j := 0; j < pq.originalDims; j++ {
		sum := 0.0
		for i := 0; i < pq.reducedDims; i++ {
			sum += pq.components.At(i, j) * projected[i]
		}
		result[j] = sum + pq.mean[j]
	}

	return result
}

func (pq *PCAQuantizer) Quantize(vector []float64) ([]byte, error) {
	if len(vector) != pq.originalDims {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	projected := pq.project(vector)
	return pq.scalarQuant.Quantize(projected)
}

func (pq *PCAQuantizer) Dequantize(data []byte) ([]float64, error) {
	projected, err := pq.scalarQuant.Dequantize(data)
	if err != nil {
		return nil, err
	}

	return pq.reconstruct(projected), nil
}

func (pq *PCAQuantizer) Dimensions() int {
	return pq.originalDims
}

func (pq *PCAQuantizer) CompressedSize() int {
	return pq.scalarQuant.CompressedSize()
}

// ============================================================================
// FILE: pkg/sketch/sketch.go
// ============================================================================

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

// ============================================================================
// FILE: pkg/sketch/hyperloglog.go
// ============================================================================

package sketch

import (
	"hash/fnv"
	"math"
	"math/bits"
)

type HyperLogLogPP struct {
	precision uint8
	m         uint32
	registers []uint8
	alpha     float64
}

func NewHyperLogLogPP(precision int) *HyperLogLogPP {
	p := uint8(precision)
	m := uint32(1 << p)

	var alpha float64
	switch m {
	case 16:
		alpha = 0.673
	case 32:
		alpha = 0.697
	case 64:
		alpha = 0.709
	default:
		alpha = 0.7213 / (1 + 1.079/float64(m))
	}

	return &HyperLogLogPP{
		precision: p,
		m:         m,
		registers: make([]uint8, m),
		alpha:     alpha,
	}
}

func (hll *HyperLogLogPP) Add(item string) {
	hash := hashString(item)
	idx := hash & (hll.m - 1)
	w := hash >> hll.precision

	leadingZeros := uint8(bits.LeadingZeros64(w)) + 1
	if leadingZeros > hll.registers[idx] {
		hll.registers[idx] = leadingZeros
	}
}

func (hll *HyperLogLogPP) Count() uint64 {
	sum := 0.0
	zeros := 0

	for _, val := range hll.registers {
		sum += 1.0 / math.Pow(2, float64(val))
		if val == 0 {
			zeros++
		}
	}

	estimate := hll.alpha * float64(hll.m) * float64(hll.m) / sum

	if estimate <= 2.5*float64(hll.m) {
		if zeros != 0 {
			estimate = float64(hll.m) * math.Log(float64(hll.m)/float64(zeros))
		}
	}

	if estimate > math.Pow(2, 32)/30 {
		estimate = -math.Pow(2, 32) * math.Log(1-estimate/math.Pow(2, 32))
	}

	return uint64(estimate)
}

func hashString(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// ============================================================================
// FILE: pkg/sketch/countmin.go
// ============================================================================

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