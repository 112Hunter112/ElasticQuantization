package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	DiscrepanciesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "discrepancies_detected_total",
		Help: "The total number of data discrepancies detected between DB and ES",
	}, []string{"table"})

	HealingOperationsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "healing_operations_total",
		Help: "The total number of healing operations performed",
	}, []string{"table", "status"}) // status: success, failure

	VectorCompressionRatio = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "vector_compression_ratio_average",
		Help: "The current average vector compression ratio (percentage)",
	})
)
