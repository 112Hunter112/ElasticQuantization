package quantizer

type Quantizer interface {
	Fit(vectors [][]float64) error
	Quantize(vector []float64) ([]byte, error)
	Dequantize(data []byte) ([]float64, error)
	Dimensions() int
	CompressedSize() int
}
