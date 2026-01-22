package sketch

import (
	"hash/fnv"
	"math"
	"math/bits"
)

type HyperLogLogPP struct {
	precision uint8
	m         uint32
	registers []uint8
	alpha     float64
}

func NewHyperLogLogPP(precision int) *HyperLogLogPP {
	p := uint8(precision)
	m := uint32(1 << p)

	var alpha float64
	switch m {
	case 16:
		alpha = 0.673
	case 32:
		alpha = 0.697
	case 64:
		alpha = 0.709
	default:
		alpha = 0.7213 / (1 + 1.079/float64(m))
	}

	return &HyperLogLogPP{
		precision: p,
		m:         m,
		registers: make([]uint8, m),
		alpha:     alpha,
	}
}

func (hll *HyperLogLogPP) Add(item string) {
	hash := hashString(item)
	idx := hash & (hll.m - 1)
	w := hash >> hll.precision

	leadingZeros := uint8(bits.LeadingZeros64(w)) + 1
	if leadingZeros > hll.registers[idx] {
		hll.registers[idx] = leadingZeros
	}
}

func (hll *HyperLogLogPP) Count() uint64 {
	sum := 0.0
	zeros := 0

	for _, val := range hll.registers {
		sum += 1.0 / math.Pow(2, float64(val))
		if val == 0 {
			zeros++
		}
	}

	estimate := hll.alpha * float64(hll.m) * float64(hll.m) / sum

	if estimate <= 2.5*float64(hll.m) {
		if zeros != 0 {
			estimate = float64(hll.m) * math.Log(float64(hll.m)/float64(zeros))
		}
	}

	if estimate > math.Pow(2, 32)/30 {
		estimate = -math.Pow(2, 32) * math.Log(1-estimate/math.Pow(2, 32))
	}

	return uint64(estimate)
}

func hashString(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}
