package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config represents the global configuration for the consistency auditor.
type Config struct {
	Database      DatabaseConfig      `yaml:"database"`
	Elasticsearch ElasticsearchConfig `yaml:"elasticsearch"`
	NeuralODE     NeuralODEConfig     `yaml:"neural_ode"` // Added this field
	Quantizer     QuantizerConfig     `yaml:"quantizer"`
	Sketch        SketchConfig        `yaml:"sketch"`
	CDC           CDCConfig           `yaml:"cdc"`
	Checker       CheckerConfig       `yaml:"checker"`
	Healer        HealerConfig        `yaml:"healer"`
	Guardrail     GuardrailConfig     `yaml:"guardrail"`
}

type DatabaseConfig struct {
	Host     string `yaml:"host"`
	Port     int    `yaml:"port"`
	User     string `yaml:"user"`
	Password string `yaml:"password"`
	Database string `yaml:"database"`
	SSLMode  string `yaml:"ssl_mode"`
}

type ElasticsearchConfig struct {
	Addresses []string `yaml:"addresses"`
	Username  string   `yaml:"username"`
	Password  string   `yaml:"password"`
	IndexName string   `yaml:"index_name"`
}

// NeuralODEConfig holds connection details for the ML service
type NeuralODEConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

type QuantizerConfig struct {
	Type         string `yaml:"type"`
	Dimensions   int    `yaml:"dimensions"`
	ReducedDims  int    `yaml:"reduced_dims"`
	Bits         int    `yaml:"bits"`
	TrainingData string `yaml:"training_data"`
}

type SketchConfig struct {
	HLLPrecision  int `yaml:"hll_precision"`
	CMSketchWidth int `yaml:"cm_sketch_width"`
	CMSketchDepth int `yaml:"cm_sketch_depth"`
}

type CDCConfig struct {
	SlotName     string `yaml:"slot_name"`
	Publication  string `yaml:"publication"`
	BatchSize    int    `yaml:"batch_size"`
	PollInterval string `yaml:"poll_interval"`
}

type CheckerConfig struct {
	MaxRetries int `yaml:"max_retries"`
}

type HealerConfig struct {
	Strategy   string `yaml:"strategy"`
	MaxRetries int    `yaml:"max_retries"`
}

type GuardrailConfig struct {
	MaxFieldCount int `yaml:"max_field_count"`
	MaxDepth      int `yaml:"max_depth"`
}

// Load reads the configuration from the specified file path.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing config file: %w", err)
	}

	return &cfg, nil
}
