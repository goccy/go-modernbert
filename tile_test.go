package modernbert

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestTileGemm runs the tile GEMM through the architecture's pack and
// quantize functions against a float32 reference, for the GEMM shape
// (weights packed once) and the attention shapes (blocks spaced out by a
// larger stride, unsigned probabilities).
func TestTileGemm(t *testing.T) {
	kt := testKernels(t).tile
	if kt == nil {
		t.Skip("no int8 tile")
	}
	t.Logf("cols=%d kt.actOffset=%d actLevels=%v kt.probLevels=%v", kt.cols, kt.actOffset, kt.actLevels, kt.probLevels)
	for _, tc := range []struct{ k, ncb int }{{8, 1}, {64, 3}, {512, 5}, {2048, 2}} {
		t.Run(fmt.Sprintf("k=%d/ncb=%d", tc.k, tc.ncb), func(t *testing.T) {
			rng := rand.New(rand.NewSource(1))
			rows, cols := tileRows, kt.cols*tc.ncb
			x := make([]float32, rows*tc.k)
			w := make([]float32, cols*tc.k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			// Weights: quantized per row like a Linear, packed once.
			l := Linear{Out: cols, In: tc.k, W: w}
			quantizeLinear(&l)
			packed := make([]byte, cols*tc.k)
			corr := make([]int32, cols)
			kt.packB(l.W8, cols, tc.k, kt.actOffset, packed, corr)
			a := make([]byte, rows*tc.k)
			xs := make([]float32, rows)
			kt.quantA(&x[0], tc.k, tc.k, &a[0], &xs[0])
			stride := cols + 3
			out := make([]float32, rows*stride)
			kt.gemm(&a[0], &packed[0], tc.k, tc.ncb, kt.cols*tc.k, &xs[0], &l.Scale[0], &corr[0], &out[0], stride)
			assertClose(t, "gemm", x, w, rows, cols, tc.k, out, stride, 3e-2)

			// Attention shapes: B from quantizeTileB with blocks spaced out,
			// A as probabilities in [0, 1].
			gap := 2 * kt.cols * tc.k
			spaced := make([]byte, tc.ncb*gap)
			corr2 := make([]int32, cols)
			ws2 := make([]float32, cols)
			tmp := make([]int8, cols*tc.k)
			for cb := 0; cb < tc.ncb; cb++ {
				kt.quantB(w[cb*kt.cols*tc.k:], kt.cols, tc.k, tc.k, 0, kt.valueLevels, spaced[cb*gap:], corr2[cb*kt.cols:], ws2[cb*kt.cols:], tmp)
			}
			p := make([]float32, rows*tc.k)
			for i := range p {
				p[i] = rng.Float32()
			}
			probs := make([]byte, rows*tc.k)
			kt.quantProbs(&p[0], tc.k, tc.k, &probs[0])
			ps := make([]float32, rows)
			for r := range ps {
				ps[r] = 1 / kt.probLevels
			}
			kt.gemmProbs(&probs[0], &spaced[0], tc.k, tc.ncb, gap, &ps[0], &ws2[0], &corr2[0], &out[0], stride)
			assertClose(t, "probs", p, w, rows, cols, tc.k, out, stride, 3e-2)
		})
	}
}

// assertClose checks out against x·wᵀ in float32, relative to the largest
// magnitude of the reference.
func assertClose(t *testing.T, name string, x, w []float32, rows, cols, k int, out []float32, stride int, tol float64) {
	t.Helper()
	var maxRef, maxDiff float64
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			var s float64
			for i := 0; i < k; i++ {
				s += float64(x[r*k+i]) * float64(w[c*k+i])
			}
			maxRef = math.Max(maxRef, math.Abs(s))
			maxDiff = math.Max(maxDiff, math.Abs(s-float64(out[r*stride+c])))
		}
	}
	t.Logf("%s: max|diff| %.3e of max|ref| %.3e", name, maxDiff, maxRef)
	if maxDiff > tol*maxRef {
		t.Fatalf("%s diverges: %.3e > %v × %.3e", name, maxDiff, tol, maxRef)
	}
}
