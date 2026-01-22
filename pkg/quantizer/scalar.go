package quantizer

import (
	"bytes"
	"fmt"
	"math"
)

type ScalarQuantizer struct {
	dims   int
	bits   int
	mins   []float64
	maxs   []float64
	scales []float64
}

func NewScalarQuantizer(dims, bits int) *ScalarQuantizer {
	return &ScalarQuantizer{
		dims:   dims,
		bits:   bits,
		mins:   make([]float64, dims),
		maxs:   make([]float64, dims),
		scales: make([]float64, dims),
	}
}

func (sq *ScalarQuantizer) Fit(vectors [][]float64) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}

	for d := 0; d < sq.dims; d++ {
		sq.mins[d] = math.MaxFloat64
		sq.maxs[d] = -math.MaxFloat64

		for _, vec := range vectors {
			if vec[d] < sq.mins[d] {
				sq.mins[d] = vec[d]
			}
			if vec[d] > sq.maxs[d] {
				sq.maxs[d] = vec[d]
			}
		}

		rangeVal := sq.maxs[d] - sq.mins[d]
		if rangeVal < 1e-10 {
			sq.scales[d] = 1.0
		} else {
			maxVal := float64(uint(1<<sq.bits) - 1)
			sq.scales[d] = maxVal / rangeVal
		}
	}

	return nil
}

func (sq *ScalarQuantizer) Quantize(vector []float64) ([]byte, error) {
	if len(vector) != sq.dims {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	var buf bytes.Buffer

	for d := 0; d < sq.dims; d++ {
		normalized := (vector[d] - sq.mins[d]) * sq.scales[d]

		if sq.bits == 8 {
			quantized := uint8(math.Max(0, math.Min(255, normalized)))
			buf.WriteByte(quantized)
		} else if sq.bits == 16 {
			quantized := uint16(math.Max(0, math.Min(65535, normalized)))
			buf.WriteByte(byte(quantized >> 8))
			buf.WriteByte(byte(quantized & 0xFF))
		}
	}

	return buf.Bytes(), nil
}

func (sq *ScalarQuantizer) Dequantize(data []byte) ([]float64, error) {
	bytesPerDim := sq.bits / 8
	expectedLen := sq.dims * bytesPerDim

	if len(data) != expectedLen {
		return nil, fmt.Errorf("data length mismatch")
	}

	vector := make([]float64, sq.dims)

	for d := 0; d < sq.dims; d++ {
		var quantized float64

		if sq.bits == 8 {
			quantized = float64(data[d])
		} else if sq.bits == 16 {
			idx := d * 2
			quantized = float64(uint16(data[idx])<<8 | uint16(data[idx+1]))
		}

		vector[d] = (quantized / sq.scales[d]) + sq.mins[d]
	}

	return vector, nil
}

func (sq *ScalarQuantizer) Dimensions() int {
	return sq.dims
}

func (sq *ScalarQuantizer) CompressedSize() int {
	return sq.dims * (sq.bits / 8)
}
