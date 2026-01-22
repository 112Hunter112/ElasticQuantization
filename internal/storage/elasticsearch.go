package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/yourusername/consistency-auditor/internal/config"
	"github.com/yourusername/consistency-auditor/pkg/quantizer"
)

type QuantizedESClient struct {
	client    *http.Client
	config    config.ElasticsearchConfig
	quantizer quantizer.Quantizer
}

func NewQuantizedESClient(cfg config.ElasticsearchConfig, q quantizer.Quantizer) (*QuantizedESClient, error) {
	return &QuantizedESClient{
		client:    &http.Client{},
		config:    cfg,
		quantizer: q,
	}, nil
}

// IndexDocument indexes a document with quantized vector data.
func (c *QuantizedESClient) IndexDocument(ctx context.Context, id string, data map[string]interface{}) error {
	// Check if vector exists and quantize it
	if vec, ok := data["vector"].([]float64); ok {
		quantized, err := c.quantizer.Quantize(vec)
		if err != nil {
			return fmt.Errorf("quantizing vector: %w", err)
		}
		data["vector_quantized"] = quantized
		delete(data, "vector") // Remove original vector to save space
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
