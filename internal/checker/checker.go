package checker

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/yourusername/consistency-auditor/internal/config"
	"github.com/yourusername/consistency-auditor/internal/ml"
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
	mlClient  *ml.Client
}

func NewConsistencyChecker(
	db *sql.DB,
	es *storage.QuantizedESClient,
	cfg config.CheckerConfig,
	sketch *sketch.SketchAggregator,
	mlClient *ml.Client,
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
	if err := validateTableName(table); err != nil {
		return nil, fmt.Errorf("invalid table name: %w", err)
	}

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
		// Record deleted in DB check
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
	// Runs asynchronously with REAL data from cdc_log
	go func() {
		// Create a detached context so the DB query doesn't die if the parent request finishes fast
		mlCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		// A. Extract current vector from the live DB row
		currentVec, err := c.extractVector(dbData)
		if err != nil {
			// If no vector column or parse error, skip ML check silently
			return
		}

		// B. Fetch History from CDC Log
		historyVectors, historyTimes, err := c.fetchHistory(mlCtx, table, id, currentVec)
		if err != nil {
			log.Printf("[ML Warning] Failed to fetch history: %v", err)
			return
		}

		// If insufficient history, skip check
		if len(historyVectors) < 2 {
			return
		}

		// C. Predict
		// Target time is the next logical step; here we assume unit steps for simplicity
		targetTime := historyTimes[len(historyTimes)-1] + 1.0

		pred, err := c.mlClient.PredictConsistency(mlCtx, historyVectors, historyTimes, targetTime)
		if err != nil {
			return
		}

		if pred.IsAnomalous {
			log.Printf("[ML ALERT] Neural ODE detected anomaly in %s/%s (Score: %.4f)", table, id, pred.AnomalyScore)
		} else {
			log.Printf("[ML INFO] Consistency Verified (Uncertainty: %.4f)", pred.Uncertainty)
		}
	}()

	// 4. Compare (Standard Logic)
	var discrepancies []Discrepancy
	esSource, ok := esDoc["_source"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid es document structure")
	}

	for k, v := range dbData {
		// Skip comparing the raw vector string directly as formats might differ (string vs list)
		if k == "embedding" {
			continue
		}

		esVal, ok := esSource[k]
		if !ok {
			discrepancies = append(discrepancies, Discrepancy{Table: table, ID: id, Field: k, SourceValue: v, TargetValue: nil})
			continue
		}

		dbStr := fmt.Sprintf("%v", v)
		esStr := fmt.Sprintf("%v", esVal)
		if dbStr != esStr {
			discrepancies = append(discrepancies, Discrepancy{Table: table, ID: id, Field: k, SourceValue: v, TargetValue: esVal})
		}
	}

	return discrepancies, nil
}

// fetchHistory retrieves the last N versions of the vector from cdc_log
func (c *ConsistencyChecker) fetchHistory(ctx context.Context, table, id string, currentVec []float64) ([][]float64, []float64, error) {
	// Fetch last 5 versions
	query := `
		SELECT row_data->>'embedding', EXTRACT(EPOCH FROM changed_at) 
		FROM cdc_log 
		WHERE table_name = $1 AND record_id = $2 
		AND row_data->>'embedding' IS NOT NULL
		ORDER BY changed_at DESC 
		LIMIT 5`

	rows, err := c.db.QueryContext(ctx, query, table, id)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()

	var vectors [][]float64
	var timestamps []float64

	// Add current state first (will reverse later)
	vectors = append(vectors, currentVec)
	timestamps = append(timestamps, float64(time.Now().Unix()))

	for rows.Next() {
		var vecStr string
		var ts float64
		if err := rows.Scan(&vecStr, &ts); err != nil {
			continue
		}

		vec, err := c.parseVectorString(vecStr)
		if err == nil {
			vectors = append(vectors, vec)
			timestamps = append(timestamps, ts)
		}
	}

	// Reverse to get chronological order [Oldest -> Newest]
	// (Production implementation would use a proper reverse loop)
	for i, j := 0, len(vectors)-1; i < j; i, j = i+1, j-1 {
		vectors[i], vectors[j] = vectors[j], vectors[i]
		timestamps[i], timestamps[j] = timestamps[j], timestamps[i]
	}

	// Normalize timestamps to start at 0.0 for numerical stability in Neural ODE
	if len(timestamps) > 0 {
		start := timestamps[0]
		for i := range timestamps {
			timestamps[i] = timestamps[i] - start
		}
	}

	return vectors, timestamps, nil
}

// extractVector gets the vector slice from the generic DB map
func (c *ConsistencyChecker) extractVector(data map[string]interface{}) ([]float64, error) {
	val, ok := data["embedding"]
	if !ok || val == nil {
		return nil, fmt.Errorf("no embedding field")
	}

	// Postgres returns vectors as string "[1,2,3]"
	strVal, ok := val.(string)
	if !ok {
		// Might be []byte depending on driver config
		if bVal, ok := val.([]byte); ok {
			strVal = string(bVal)
		} else {
			return nil, fmt.Errorf("unknown type for embedding")
		}
	}

	return c.parseVectorString(strVal)
}

// parseVectorString converts "[0.1,0.2,...]" to []float64
func (c *ConsistencyChecker) parseVectorString(s string) ([]float64, error) {
	s = strings.Trim(s, "[]")
	if s == "" {
		return nil, fmt.Errorf("empty vector")
	}
	parts := strings.Split(s, ",")
	vec := make([]float64, len(parts))
	for i, part := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, err
		}
		vec[i] = f
	}
	return vec, nil
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
		if b, ok := val.([]byte); ok {
			entry[col] = string(b)
		} else {
			entry[col] = val
		}
	}
	return entry, nil
}

func validateTableName(table string) error {
	if table == "" {
		return fmt.Errorf("table name is empty")
	}
	// Allow only alphanumeric characters, underscores, and dots (for schema.table)
	for _, r := range table {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '.') {
			return fmt.Errorf("invalid character in table name: %c", r)
		}
	}
	return nil
}
