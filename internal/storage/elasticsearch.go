package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv" // <--- Added
	"strings" // <--- Added

	"github.com/yourusername/consistency-auditor/internal/config"
	"github.com/yourusername/consistency-auditor/internal/guardrail"
	"github.com/yourusername/consistency-auditor/pkg/quantizer"
)

type QuantizedESClient struct {
	client    *http.Client
	config    config.ElasticsearchConfig
	quantizer quantizer.Quantizer
	guardrail *guardrail.MappingGuardrail
}

func NewQuantizedESClient(cfg config.ElasticsearchConfig, guardrailCfg config.GuardrailConfig, q quantizer.Quantizer) (*QuantizedESClient, error) {
	return &QuantizedESClient{
		client:    &http.Client{},
		config:    cfg,
		quantizer: q,
		guardrail: guardrail.NewMappingGuardrail(guardrailCfg),
	}, nil
}

// IndexDocument indexes a document with quantized vector data.
func (c *QuantizedESClient) IndexDocument(ctx context.Context, id string, data map[string]interface{}) error {
	// Validate against guardrails
	if err := c.guardrail.Validate(data); err != nil {
		return fmt.Errorf("guardrail validation failed: %w", err)
	}

	// Check if vector exists (handle both []float64 and string formats)
	var vec []float64

	// Case 1: Already a slice of floats (Native Driver)
	if v, ok := data["embedding"].([]float64); ok {
		vec = v
	} else if vStr, ok := data["embedding"].(string); ok {
		// Case 2: String format "[0.1, 0.2]" - Parse it manually
		vStr = strings.Trim(vStr, "[]")
		parts := strings.Split(vStr, ",")
		vec = make([]float64, 0, len(parts))
		for _, p := range parts {
			if f, err := strconv.ParseFloat(strings.TrimSpace(p), 64); err == nil {
				vec = append(vec, f)
			}
		}
	}

	// Perform Quantization if we successfully extracted a vector
	if len(vec) > 0 {
		quantized, err := c.quantizer.Quantize(vec)
		if err != nil {
			return fmt.Errorf("quantizing vector: %w", err)
		}
		data["vector_quantized"] = quantized
		delete(data, "embedding") // Remove original vector to save space
	}

	body, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("marshaling document: %w", err)
	}

	url := fmt.Sprintf("%s/%s/_doc/%s", c.config.Addresses[0], c.config.IndexName, id)
	req, err := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("elasticsearch error: %s", resp.Status)
	}

	return nil
}

func (c *QuantizedESClient) GetDocument(ctx context.Context, id string) (map[string]interface{}, error) {
	url := fmt.Sprintf("%s/%s/_doc/%s", c.config.Addresses[0], c.config.IndexName, id)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return nil, nil // Not found
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

// DeleteDocument removes a document from the index.
func (c *QuantizedESClient) DeleteDocument(ctx context.Context, id string) error {
	url := fmt.Sprintf("%s/%s/_doc/%s", c.config.Addresses[0], c.config.IndexName, id)
	req, err := http.NewRequestWithContext(ctx, "DELETE", url, nil)
	if err != nil {
		return fmt.Errorf("creating delete request: %w", err)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("executing delete request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return nil // Already deleted
	}

	if resp.StatusCode >= 400 {
		return fmt.Errorf("elasticsearch delete error: %s", resp.Status)
	}

	return nil
}
