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

	// Fetch fresh data from Source of Truth (DB)
	var content string
	var vector []byte // Placeholder for vector data handling
	
	query := fmt.Sprintf("SELECT content FROM %s WHERE id = $1", d.Table)
	err := h.db.QueryRowContext(ctx, query, d.ID).Scan(&content)
	if err != nil {
		if err == sql.ErrNoRows {
			// Delete from ES if it doesn't exist in DB
			// implementation left as exercise
			return nil
		}
		return err
	}

	// Upsert to ES
	doc := map[string]interface{}{
		"content": content,
		// "vector": ... need logic to fetch and deserialize vector
	}
	
	// Use _ if vector is unused for now
	_ = vector

	return h.es.IndexDocument(ctx, d.ID, doc)
}
