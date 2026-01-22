package guardrail

import (
	"fmt"

	"github.com/yourusername/consistency-auditor/internal/config"
)

type MappingGuardrail struct {
	config config.GuardrailConfig
}

func NewMappingGuardrail(cfg config.GuardrailConfig) *MappingGuardrail {
	return &MappingGuardrail{
		config: cfg,
	}
}

// Validate checks if the document complies with the mapping guardrails.
func (g *MappingGuardrail) Validate(doc map[string]interface{}) error {
	// If limits are 0, we assume disabled checks
	if g.config.MaxDepth == 0 && g.config.MaxFieldCount == 0 {
		return nil
	}

	fieldCount, err := g.checkRecursive(doc, 1)
	if err != nil {
		return err
	}

	if g.config.MaxFieldCount > 0 && fieldCount > g.config.MaxFieldCount {
		return fmt.Errorf("document exceeds maximum field count: %d > %d", fieldCount, g.config.MaxFieldCount)
	}

	return nil
}

func (g *MappingGuardrail) checkRecursive(data interface{}, currentDepth int) (int, error) {
	if g.config.MaxDepth > 0 && currentDepth > g.config.MaxDepth {
		return 0, fmt.Errorf("document exceeds maximum nesting depth: %d", g.config.MaxDepth)
	}

	m, ok := data.(map[string]interface{})
	if !ok {
		// Not a map, so it's a leaf node in terms of structure traversal (or a list, which we might want to traverse if it contains maps)
		// For simplicity, let's check lists too if they contain objects
		if list, ok := data.([]interface{}); ok {
			totalFields := 0
			for _, item := range list {
				c, err := g.checkRecursive(item, currentDepth) // List doesn't increase depth for the list itself, but items are at same level? Or depth+1?
				// Usually JSON arrays don't increase mapping depth unless they are objects.
				// Let's treat array items as being at the same depth as the array field itself for simplicity,
				// or valid JSON structure says array is a value.
				// If we have "a": [{"b": 1}], "a.b" is the field.
				if err != nil {
					return 0, err
				}
				totalFields += c
			}
			return totalFields, nil
		}
		return 0, nil // Scalar value, counts as 0 extra fields (the key was counted by the parent)
	}

	count := 0
	for _, v := range m {
		count++ // Count this key
		subCount, err := g.checkRecursive(v, currentDepth+1)
		if err != nil {
			return 0, err
		}
		count += subCount
	}

	return count, nil
}
