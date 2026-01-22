package checker

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/yourusername/consistency-auditor/internal/config"
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
}

func NewConsistencyChecker(
	db *sql.DB,
	es *storage.QuantizedESClient,
	cfg config.CheckerConfig,
	sketch *sketch.SketchAggregator,
) *ConsistencyChecker {
	return &ConsistencyChecker{
		db:        db,
		es:        es,
		config:    cfg,
		sketchAgg: sketch,
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

	// 3. Compare
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

		// Simplified comparison - for production, need more robust type-aware comparison
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
