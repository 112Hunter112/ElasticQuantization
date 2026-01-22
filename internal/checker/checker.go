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
	// This is a simplified query. In production, this needs to dynamically build queries based on table schema.
	var dbValue string
	query := fmt.Sprintf("SELECT content FROM %s WHERE id = $1", table)
	err := c.db.QueryRowContext(ctx, query, id).Scan(&dbValue)
	if err != nil {
		if err == sql.ErrNoRows {
			// Record deleted in DB, check if it exists in ES
			esDoc, esErr := c.es.GetDocument(ctx, id)
			if esErr == nil && esDoc != nil {
				return []Discrepancy{{Table: table, ID: id, Field: "_exists", SourceValue: nil, TargetValue: "exists"}}, nil
			}
			return nil, nil
		}
		return nil, fmt.Errorf("fetching from db: %w", err)
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
	// Simplified comparison logic
	var discrepancies []Discrepancy
	if sourceVal, ok := esDoc["_source"].(map[string]interface{}); ok {
		if esContent, ok := sourceVal["content"].(string); ok {
			if dbValue != esContent {
				discrepancies = append(discrepancies, Discrepancy{
					Table:       table,
					ID:          id,
					Field:       "content",
					SourceValue: dbValue,
					TargetValue: esContent,
				})
			}
		}
	}

	return discrepancies, nil
}
