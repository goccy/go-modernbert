package modernbert

import (
	"math/rand"
	"testing"
)

// TestGemmInt8MatchesScalar checks the tiled int8 GEMM (the CPU's tile
// kernel for the bulk, scalar Go on the edges) against the scalar dot for
// shapes with and without tile remainders, including values at the int8
// extremes.
func TestGemmInt8MatchesScalar(t *testing.T) {
	kernel := testKernels(t).dot
	t.Logf("tile kernel: %dx%d kAlign %d unsignedX %v", kernel.rows, kernel.cols, kernel.kAlign, kernel.unsignedX)
	rng := rand.New(rand.NewSource(1))
	for _, shape := range []struct{ T, N, K int }{{4, 4, 16}, {8, 8, 512}, {5, 7, 32}, {1, 3, 2048}, {9, 13, 48}, {3, 4, 20}, {4, 4, 64}, {6, 9, 128}, {2, 4, 32}, {8, 12, 192}} {
		x := make([]int8, shape.T*shape.K)
		w := make([]int8, shape.N*shape.K)
		for i := range x {
			x[i] = int8(rng.Intn(255) - 127)
		}
		for i := range w {
			w[i] = int8(rng.Intn(255) - 127)
		}
		// Extremes in the first rows: every product at ±127² must survive.
		for c := 0; c < shape.K; c++ {
			x[c] = -127
			w[c] = 127
			if c%2 == 1 {
				x[c] = 127
			}
		}
		ones := func(n int) []float32 {
			s := make([]float32, n)
			for i := range s {
				s[i] = 1
			}
			return s
		}
		got := make([]float32, shape.T*shape.N)
		ws := &workspace{xu: make([]int8, len(x)), pool: newPool(3)}
		t.Cleanup(ws.pool.close)
		gemmInt8(kernel, x, shape.T, shape.K, w, rowSums(w, shape.N, shape.K), shape.N, ws, ones(shape.T), ones(shape.N), got)
		for ti := 0; ti < shape.T; ti++ {
			for n := 0; n < shape.N; n++ {
				want := float32(dotInt8(x[ti*shape.K:(ti+1)*shape.K], w[n*shape.K:(n+1)*shape.K]))
				if got[ti*shape.N+n] != want {
					t.Fatalf("shape %+v: out[%d][%d] = %v, want %v", shape, ti, n, got[ti*shape.N+n], want)
				}
			}
		}
	}
}

func BenchmarkGemmInt8(b *testing.B) {
	const T, N, K = 64, 512, 512
	x := make([]int8, T*K)
	w := make([]int8, N*K)
	for i := range x {
		x[i] = int8(i % 7)
	}
	for i := range w {
		w[i] = int8(i % 5)
	}
	rs := rowSums(w, N, K)
	xs := make([]float32, T)
	wscale := make([]float32, N)
	out := make([]float32, T*N)
	ws := &workspace{xu: make([]int8, len(x)), pool: newPool(3)}
	b.Cleanup(ws.pool.close)
	b.ResetTimer()
	kn := testKernels(b).dot
	for i := 0; i < b.N; i++ {
		gemmInt8(kn, x, T, K, w, rs, N, ws, xs, wscale, out)
	}
	b.ReportMetric(float64(T*N*K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
}

func rowSums(w []int8, n, k int) []int32 {
	out := make([]int32, n)
	for r := 0; r < n; r++ {
		for _, v := range w[r*k : (r+1)*k] {
			out[r] += int32(v)
		}
	}
	return out
}
