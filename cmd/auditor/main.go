package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/yourusername/consistency-auditor/internal/cdc"
	"github.com/yourusername/consistency-auditor/internal/checker"
	"github.com/yourusername/consistency-auditor/internal/config"
	"github.com/yourusername/consistency-auditor/internal/healer"
	"github.com/yourusername/consistency-auditor/internal/metrics"
	"github.com/yourusername/consistency-auditor/internal/ml"
	"github.com/yourusername/consistency-auditor/internal/storage"
	"github.com/yourusername/consistency-auditor/pkg/quantizer"
	"github.com/yourusername/consistency-auditor/pkg/sketch"
)

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	metricsAddr := flag.String("metrics-addr", ":9090", "address to listen on for metrics")
	flag.Parse()

	// Start metrics server
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		log.Printf("Metrics server listening on %s", *metricsAddr)
		if err := http.ListenAndServe(*metricsAddr, nil); err != nil {
			log.Fatalf("Metrics server failed: %v", err)
		}
	}()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	mlURL := fmt.Sprintf("http://%s:%d", cfg.NeuralODE.Host, cfg.NeuralODE.Port)
	mlClient := ml.NewClient(mlURL)
	log.Printf("Connected to Neural ODE Service at %s", mlURL)
	log.Println("Neural ODE Client initialized")

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

	// Train the quantizer
	// CRITICAL FIX: Always train, even if config is empty, to prevent nil pointer crashes
	var trainingVectors [][]float64
	if len(cfg.Quantizer.TrainingData) > 0 {
		trainingVectors = loadTrainingData(cfg.Quantizer.TrainingData)
	} else {
		log.Println("⚠️ No training data provided in config. Generating synthetic data for initial PCA...")
		// Generate enough random vectors to ensure PCA doesn't fail mathematically
		// Use Dimensions + 10 to ensure we have more samples than dimensions for stability
		trainingVectors = generateSyntheticData(cfg.Quantizer.Dimensions, cfg.Quantizer.Dimensions+10)
	}

	if err := quantizerInstance.Fit(trainingVectors); err != nil {
		log.Fatalf("Failed to train quantizer: %v", err)
	}
	log.Printf("Quantizer trained on %d vectors", len(trainingVectors))

	esClient, err := storage.NewQuantizedESClient(cfg.Elasticsearch, cfg.Guardrail, quantizerInstance)
	if err != nil {
		log.Fatalf("Failed to connect to Elasticsearch: %v", err)
	}

	// Create a sketch.SketchConfig using values from the global config
	sketchAgg := sketch.NewSketchAggregator(sketch.SketchConfig{
		HLLPrecision:  cfg.Sketch.HLLPrecision,
		CMSketchWidth: cfg.Sketch.CMSketchWidth,
		CMSketchDepth: cfg.Sketch.CMSketchDepth,
	})

	cdcListener := cdc.NewCDCListener(cfg.CDC, cfg.Database)
	if err := cdcListener.Start(); err != nil {
		log.Fatalf("Failed to start CDC listener: %v", err)
	}
	defer cdcListener.Stop()

	consistencyChecker := checker.NewConsistencyChecker(db, esClient, cfg.Checker, sketchAgg, mlClient)

	autoHealer := healer.NewAutoHealer(db, esClient, cfg.Healer)

	auditor := NewAuditor(cdcListener, consistencyChecker, autoHealer, sketchAgg)

	if err := auditor.Start(); err != nil {
		log.Fatalf("Failed to start auditor: %v", err)
	}

	log.Println("Consistency Auditor started successfully")
	ratio := getCompressionRatio(quantizerInstance)
	log.Printf("Compression ratio: %.1f%%", ratio)
	metrics.VectorCompressionRatio.Set(ratio)
	log.Printf("Healing strategy: %s", cfg.Healer.Strategy)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down gracefully...")
	auditor.Stop()
	log.Println("Goodbye")
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
				log.Printf("Error checking %s/%s: %v", event.Table, event.ID, err)
				continue
			}

			if len(discrepancies) > 0 {
				metrics.DiscrepanciesTotal.WithLabelValues(event.Table).Add(float64(len(discrepancies)))
				log.Printf("Found %d discrepancies in %s/%s", len(discrepancies), event.Table, event.ID)
				if err := a.healer.Heal(a.ctx, discrepancies); err != nil {
					metrics.HealingOperationsTotal.WithLabelValues(event.Table, "failure").Inc()
					log.Printf("Error healing: %v", err)
				} else {
					metrics.HealingOperationsTotal.WithLabelValues(event.Table, "success").Inc()
					log.Printf("Healed %s/%s", event.Table, event.ID)
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
			log.Println("Running periodic full audit...")
			stats := a.sketch.GetStats()
			log.Printf("Sketch stats: %+v", stats)

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
			log.Printf("Stats - Unique records: ~%d, Total events: ~%d",
				stats.UniqueCount, stats.TotalEvents)

		case <-a.ctx.Done():
			return
		}
	}
}

func (a *Auditor) Stop() {
	a.cancel()
}

// loadTrainingData loads vectors from a CSV file.
// If the file is missing, it generates synthetic data to prevent crashes.
func loadTrainingData(path string) [][]float64 {
	f, err := os.Open(path)
	if err != nil {
		log.Printf("Training data not found at %s. Generating synthetic 768-dim data for PCA...", path)
		return generateSyntheticData(768, 100) // 100 samples of 768 dimensions
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		log.Printf("Error reading CSV: %v. Using synthetic data.", err)
		return generateSyntheticData(768, 100)
	}

	var data [][]float64
	for _, row := range records {
		var vec []float64
		for _, val := range row {
			if v, err := strconv.ParseFloat(val, 64); err == nil {
				vec = append(vec, v)
			}
		}
		// Ensure we only keep valid vectors
		if len(vec) > 0 {
			data = append(data, vec)
		}
	}

	log.Printf("Loaded %d training vectors from %s", len(data), path)
	return data
}

// Helper to prevent crashes if no data exists
func generateSyntheticData(dim, samples int) [][]float64 {
	data := make([][]float64, samples)
	for i := 0; i < samples; i++ {
		vec := make([]float64, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float64()
		}
		data[i] = vec
	}
	return data
}

func getCompressionRatio(q quantizer.Quantizer) float64 {
	original := float64(q.Dimensions() * 8)
	compressed := float64(q.CompressedSize())
	return 100.0 * (1.0 - compressed/original)
}
