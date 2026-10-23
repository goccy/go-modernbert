package modernbert

import (
	"math"
	"math/rand"
	"testing"
)

func TestF32TilesMatchScalar(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewSource(2))
	fill := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	for _, k := range []int{4, 64, 128} {
		aStride, bStride := k+8, k+4
		a, b := fill(4*aStride), fill(4*bStride)
		var got [16]float32
		dotF32Tile4x4(&a[0], &b[0], k, aStride, bStride, &got[0])
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				var want float64
				for c := 0; c < k; c++ {
					want += float64(a[i*aStride+c]) * float64(b[j*bStride+c])
				}
				if math.Abs(float64(got[i*4+j])-want) > 1e-3*(1+math.Abs(want)) {
					t.Fatalf("dot k=%d out[%d][%d] = %v, want %v", k, i, j, got[i*4+j], want)
				}
			}
		}
	}
	for _, n := range []int{1, 7, 64, 513} {
		pStride, vStride, outStride := n+3, 16+9, 16+5
		p, v := fill(4*pStride), fill(n*vStride)
		out := make([]float32, 4*outStride)
		axpyTile4x16(&p[0], pStride, &v[0], vStride, n, &out[0], outStride)
		for i := 0; i < 4; i++ {
			for d := 0; d < 16; d++ {
				var want float64
				for j := 0; j < n; j++ {
					want += float64(p[i*pStride+j]) * float64(v[j*vStride+d])
				}
				if math.Abs(float64(out[i*outStride+d])-want) > 1e-3*(1+math.Abs(want)) {
					t.Fatalf("axpy n=%d out[%d][%d] = %v, want %v", n, i, d, out[i*outStride+d], want)
				}
			}
		}
	}
}
