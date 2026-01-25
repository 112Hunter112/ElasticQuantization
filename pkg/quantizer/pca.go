package quantizer

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type PCAQuantizer struct {
	originalDims int
	reducedDims  int
	mean         []float64
	components   *mat.Dense
	scalarQuant  *ScalarQuantizer
}

func NewPCAQuantizer(originalDims, reducedDims, bits int) *PCAQuantizer {
	return &PCAQuantizer{
		originalDims: originalDims,
		reducedDims:  reducedDims,
		mean:         make([]float64, originalDims),
		scalarQuant:  NewScalarQuantizer(reducedDims, bits),
	}
}

func (pq *PCAQuantizer) Fit(vectors [][]float64) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}

	n := len(vectors)

	for d := 0; d < pq.originalDims; d++ {
		sum := 0.0
		for _, vec := range vectors {
			sum += vec[d]
		}
		pq.mean[d] = sum / float64(n)
	}

	centered := mat.NewDense(n, pq.originalDims, nil)
	for i, vec := range vectors {
		for j := 0; j < pq.originalDims; j++ {
			centered.Set(i, j, vec[j]-pq.mean[j])
		}
	}

	var cov mat.SymDense
	stat.CovarianceMatrix(&cov, centered, nil)

	var eig mat.Eigen
	ok := eig.Factorize(&cov, mat.EigenRight)
	if !ok {
		return fmt.Errorf("eigendecomposition failed")
	}

	values := eig.Values(nil)
	var vectors_eig mat.CDense
	eig.VectorsTo(&vectors_eig)

	type eigenPair struct {
		value  float64
		vector []float64
	}

	pairs := make([]eigenPair, pq.originalDims)
	for i := 0; i < pq.originalDims; i++ {
		vec := make([]float64, pq.originalDims)
		for j := 0; j < pq.originalDims; j++ {
			vec[j] = real(vectors_eig.At(j, i))
		}
		pairs[i] = eigenPair{value: real(values[i]), vector: vec}
	}

	// Sort by eigenvalue descending
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})

	pq.components = mat.NewDense(pq.reducedDims, pq.originalDims, nil)
	for i := 0; i < pq.reducedDims; i++ {
		for j := 0; j < pq.originalDims; j++ {
			pq.components.Set(i, j, pairs[i].vector[j])
		}
	}

	projected := make([][]float64, n)
	for i, vec := range vectors {
		projected[i] = pq.project(vec)
	}

	return pq.scalarQuant.Fit(projected)
}

func (pq *PCAQuantizer) project(vector []float64) []float64 {
	centered := make([]float64, pq.originalDims)
	for i := 0; i < pq.originalDims; i++ {
		centered[i] = vector[i] - pq.mean[i]
	}

	result := make([]float64, pq.reducedDims)
	for i := 0; i < pq.reducedDims; i++ {
		sum := 0.0
		for j := 0; j < pq.originalDims; j++ {
			sum += pq.components.At(i, j) * centered[j]
		}
		result[i] = sum
	}

	return result
}

func (pq *PCAQuantizer) reconstruct(projected []float64) []float64 {
	result := make([]float64, pq.originalDims)

	for j := 0; j < pq.originalDims; j++ {
		sum := 0.0
		for i := 0; i < pq.reducedDims; i++ {
			sum += pq.components.At(i, j) * projected[i]
		}
		result[j] = sum + pq.mean[j]
	}

	return result
}

func (pq *PCAQuantizer) Quantize(vector []float64) ([]byte, error) {
	if len(vector) != pq.originalDims {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	projected := pq.project(vector)
	return pq.scalarQuant.Quantize(projected)
}

func (pq *PCAQuantizer) Dequantize(data []byte) ([]float64, error) {
	projected, err := pq.scalarQuant.Dequantize(data)
	if err != nil {
		return nil, err
	}

	return pq.reconstruct(projected), nil
}

func (pq *PCAQuantizer) Dimensions() int {
	return pq.originalDims
}

func (pq *PCAQuantizer) CompressedSize() int {
	return pq.scalarQuant.CompressedSize()
}
