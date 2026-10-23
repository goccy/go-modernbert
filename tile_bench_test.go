package modernbert

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkTileGemm times the architecture's tile GEMM on the shapes of
// the 130m model's linear layers (512 rows) to read its MAC rate.
func BenchmarkTileGemm(b *testing.B) {
	kt := testKernels(b).tile
	if kt == nil {
		b.Skip("no int8 tile")
	}
	for _, shape := range []struct{ K, N int }{{512, 512}, {512, 4096}, {2048, 512}} {
		K, N := shape.K, shape.N
		const T = 512
		rng := rand.New(rand.NewSource(1))
		w := make([]int8, N*K)
		for i := range w {
			w[i] = int8(rng.Intn(255) - 127)
		}
		packed := make([]byte, N*K)
		corr := make([]int32, N)
		kt.packB(w, N, K, kt.actOffset, packed, corr)
		a := make([]byte, T*K)
		xs := make([]float32, T)
		ws := make([]float32, N)
		out := make([]float32, T*N)
		b.Run(fmt.Sprintf("K=%d/N=%d", K, N), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for g := 0; g < T/tileRows; g++ {
					kt.gemm(&a[g*tileRows*K], &packed[0], K, N/kt.cols, kt.cols*K, &xs[g*tileRows], &ws[0], &corr[0], &out[g*tileRows*N], N)
				}
			}
			b.ReportMetric(float64(T)*float64(K)*float64(N)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
		})
	}
}
