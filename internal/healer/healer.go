package healer

import (
	"context"
	"database/sql"
	"fmt"
	"log"

	"github.com/yourusername/consistency-auditor/internal/checker"
	"github.com/yourusername/consistency-auditor/internal/config"
	"github.com/yourusername/consistency-auditor/internal/storage"
)

type AutoHealer struct {
	db     *sql.DB
	es     *storage.QuantizedESClient
	config config.HealerConfig
}

func NewAutoHealer(
	db *sql.DB,
	es *storage.QuantizedESClient,
	cfg config.HealerConfig,
) *AutoHealer {
	return &AutoHealer{
		db:     db,
		es:     es,
		config: cfg,
	}
}

func (h *AutoHealer) Heal(ctx context.Context, discrepancies []checker.Discrepancy) error {
	if h.config.Strategy == "alert_only" {
		log.Printf("Alert: %d discrepancies found (Healer strategy set to alert_only)", len(discrepancies))
		return nil
	}

	for _, d := range discrepancies {
		if err := h.healSingle(ctx, d); err != nil {
			return fmt.Errorf("healing discrepancy for %s/%s: %w", d.Table, d.ID, err)
		}
	}
	return nil
}

func (h *AutoHealer) healSingle(ctx context.Context, d checker.Discrepancy) error {
	log.Printf("Healing %s/%s field %s", d.Table, d.ID, d.Field)

	if err := validateTableName(d.Table); err != nil {
		return fmt.Errorf("invalid table name: %w", err)
	}

	// Fetch fresh data from Source of Truth (DB)
	query := fmt.Sprintf("SELECT * FROM %s WHERE id = $1", d.Table)
	rows, err := h.db.QueryContext(ctx, query, d.ID)
	if err != nil {
		return fmt.Errorf("querying db for healing: %w", err)
	}
	defer rows.Close()

	var dbData map[string]interface{}
	if rows.Next() {
		dbData, err = h.scanRowToMap(rows)
		if err != nil {
			return fmt.Errorf("scanning row for healing: %w", err)
		}
	} else {
		// Record deleted in DB, delete from ES
		log.Printf("Record %s/%s deleted in DB, deleting from ES", d.Table, d.ID)
		if err := h.es.DeleteDocument(ctx, d.ID); err != nil {
			return fmt.Errorf("deleting document from ES: %w", err)
		}
		return nil
	}

	// Upsert to ES
	// Note: basic upsert using the data found in DB.
	// We might need to handle specific vector fields if they exist in the map
	return h.es.IndexDocument(ctx, d.ID, dbData)
}

func (h *AutoHealer) scanRowToMap(rows *sql.Rows) (map[string]interface{}, error) {
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

func validateTableName(table string) error {
	if table == "" {
		return fmt.Errorf("table name is empty")
	}
	for _, r := range table {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '.') {
			return fmt.Errorf("invalid character in table name: %c", r)
		}
	}
	return nil
}
