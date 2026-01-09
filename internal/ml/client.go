package ml

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// PredictionResponse matches the JSON returned by your JAX service
type PredictionResponse struct {
	PredictedVector []float64 `json:"predicted_vector"`
	Uncertainty     float64   `json:"uncertainty"`
	AnomalyScore    float64   `json:"anomaly_score"`
	IsAnomalous     bool      `json:"is_anomalous"`
	Backend         string    `json:"backend"`
}

type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a connection to the Neural ODE service
func NewClient(url string) *Client {
	return &Client{
		baseURL: url,
		httpClient: &http.Client{
			Timeout: 500 * time.Millisecond, // Fail fast to keep the auditor speedy
		},
	}
}

// PredictConsistency asks the brain if the data looks valid
func (c *Client) PredictConsistency(ctx context.Context, vectors [][]float64, times []float64, targetTime float64) (*PredictionResponse, error) {
	reqBody := map[string]interface{}{
		"vectors":     vectors,
		"timestamps":  times,
		"target_time": targetTime,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/predict", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ml service unreachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ml service error: status %d", resp.StatusCode)
	}

	var result PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}
