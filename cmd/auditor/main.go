package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/yourusername/consistency-auditor/internal/cdc"
	"github.com/yourusername/consistency-auditor/internal/checker"
	"github.com/yourusername/consistency-auditor/internal/config"
	"github.com/yourusername/consistency-auditor/internal/healer"
	"github.com/yourusername/consistency-auditor/internal/metrics"
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

	esClient, err := storage.NewQuantizedESClient(cfg.Elasticsearch, cfg.Guardrail, quantizerInstance)
	if err != nil {
		log.Fatalf("Failed to connect to Elasticsearch: %v", err)
	}

	sketchAgg := sketch.NewSketchAggregator(cfg.Sketch)
	log.Printf("Sketch aggregator initialized: HLL++ precision=%d, CMSketch width=%d depth=%d",
		cfg.Sketch.HLLPrecision, cfg.Sketch.CMSketchWidth, cfg.Sketch.CMSketchDepth)

	cdcListener := cdc.NewCDCListener(cfg.CDC, cfg.Database)
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
