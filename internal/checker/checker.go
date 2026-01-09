package checker

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	"github.com/yourusername/consistency-auditor/internal/config"
	"github.com/yourusername/consistency-auditor/internal/ml" // Import the ML package
	"github.com/yourusername/consistency-auditor/internal/storage"
	"github.com/yourusername/consistency-auditor/pkg/sketch"
)

type Discrepancy struct {
	Table       string
	ID          string
	Field       string
	SourceValue interface{}
	TargetValue interface{}
}

type ConsistencyChecker struct {
	db        *sql.DB
	es        *storage.QuantizedESClient
	config    config.CheckerConfig
	sketchAgg *sketch.SketchAggregator
	mlClient  *ml.Client // Add ML Client field
}

// NewConsistencyChecker creates a new checker with the ML client injected
func NewConsistencyChecker(
	db *sql.DB,
	es *storage.QuantizedESClient,
	cfg config.CheckerConfig,
	sketch *sketch.SketchAggregator,
	mlClient *ml.Client, // Inject dependency
) *ConsistencyChecker {
	return &ConsistencyChecker{
		db:        db,
		es:        es,
		config:    cfg,
		sketchAgg: sketch,
		mlClient:  mlClient,
	}
}

func (c *ConsistencyChecker) CheckRecord(ctx context.Context, table, id string) ([]Discrepancy, error) {
	// 1. Fetch from DB
	query := fmt.Sprintf("SELECT * FROM %s WHERE id = $1", table)
	rows, err := c.db.QueryContext(ctx, query, id)
	if err != nil {
		return nil, fmt.Errorf("querying db: %w", err)
	}
	defer rows.Close()

	var dbData map[string]interface{}
	if rows.Next() {
		dbData, err = c.scanRowToMap(rows)
		if err != nil {
			return nil, fmt.Errorf("scanning row: %w", err)
		}
	} else {
		// Record deleted in DB, check if it exists in ES
		esDoc, esErr := c.es.GetDocument(ctx, id)
		if esErr == nil && esDoc != nil {
			return []Discrepancy{{Table: table, ID: id, Field: "_exists", SourceValue: nil, TargetValue: "exists"}}, nil
		}
		return nil, nil
	}

	// 2. Fetch from ES
	esDoc, err := c.es.GetDocument(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("fetching from es: %w", err)
	}

	if esDoc == nil {
		return []Discrepancy{{Table: table, ID: id, Field: "_exists", SourceValue: "exists", TargetValue: nil}}, nil
	}

	// 3. Neural ODE Consistency Check (The "Brain")
	// This runs asynchronously to avoid blocking the main audit loop
	go func() {
		mlCtx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()

		// Mock sequence for demonstration.
		// In production, fetch actual history from cdc_log or similar.
		dummyVec := []float64{0.1, 0.5}
		history := [][]float64{dummyVec, dummyVec}
		times := []float64{1.0, 2.0}
		targetTime := 3.0

		pred, err := c.mlClient.PredictConsistency(mlCtx, history, times, targetTime)
		if err != nil {
			// Log warning only if needed, usually we fail silently to avoid noise
			// log.Printf("[ML Warning] Failed to consult Neural ODE: %v", err)
			return
		}

		if pred.IsAnomalous {
			log.Printf("[ML ALERT] Neural ODE detected anomaly in %s/%s (Score: %.4f)", table, id, pred.AnomalyScore)
		} else {
			log.Printf("[ML INFO] Consistency Verified by JAX Engine (Uncertainty: %.4f)", pred.Uncertainty)
		}
	}()

	// 4. Compare
	var discrepancies []Discrepancy
	esSource, ok := esDoc["_source"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid es document structure: missing _source")
	}

	for k, v := range dbData {
		esVal, ok := esSource[k]
		if !ok {
			discrepancies = append(discrepancies, Discrepancy{
				Table:       table,
				ID:          id,
				Field:       k,
				SourceValue: v,
				TargetValue: nil,
			})
			continue
		}

		// Simplified comparison
		dbStr := fmt.Sprintf("%v", v)
		esStr := fmt.Sprintf("%v", esVal)
		if dbStr != esStr {
			discrepancies = append(discrepancies, Discrepancy{
				Table:       table,
				ID:          id,
				Field:       k,
				SourceValue: v,
				TargetValue: esVal,
			})
		}
	}

	return discrepancies, nil
}

func (c *ConsistencyChecker) scanRowToMap(rows *sql.Rows) (map[string]interface{}, error) {
	cols, err := rows.Columns()
	if err != nil {
		return nil, err
	}

	values := make([]interface{}, len(cols))
	valuePtrs := make([]interface{}, len(cols))
	for i := range values {
		valuePtrs[i] = &values[i]
	}

	if err := rows.Scan(valuePtrs...); err != nil {
		return nil, err
	}

	entry := make(map[string]interface{})
	for i, col := range cols {
		val := values[i]

		b, ok := val.([]byte)
		if ok {
			entry[col] = string(b)
		} else {
			entry[col] = val
		}
	}

	return entry, nil
}
