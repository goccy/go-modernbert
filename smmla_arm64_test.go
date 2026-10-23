//go:build arm64

package modernbert

import (
	"golang.org/x/sys/cpu"

	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func TestGemmSmmla8(t *testing.T) {
	if !cpu.ARM64.HasI8MM {
		t.Skip("no i8mm")
	}
	for _, tc := range []struct{ k, ncb int }{{8, 1}, {64, 1}, {512, 3}, {2048, 5}} {
		t.Run(fmt.Sprintf("k=%d/ncb=%d", tc.k, tc.ncb), func(t *testing.T) {
			rng := rand.New(rand.NewSource(1))
			rows, cols := 8, 8*tc.ncb
			x := make([]int8, rows*tc.k)
			w := make([]int8, cols*tc.k)
			for i := range x {
				x[i] = int8(rng.Intn(255) - 127)
			}
			for i := range w {
				w[i] = int8(rng.Intn(255) - 127)
			}
			xs := make([]float32, rows)
			ws := make([]float32, cols)
			for i := range xs {
				xs[i] = rng.Float32() + 0.5
			}
			for i := range ws {
				ws[i] = rng.Float32() + 0.5
			}
			a := make([]byte, len(x))
			b := make([]byte, len(w))
			packInt8Blocks8(x, rows, tc.k, a)
			packInt8Blocks8(w, cols, tc.k, b)
			stride := cols + 3
			out := make([]float32, rows*stride)
			gemmSmmla8((*int8)(unsafe.Pointer(&a[0])), (*int8)(unsafe.Pointer(&b[0])), tc.k, tc.ncb, 8*tc.k, &xs[0], &ws[0], &out[0], stride)
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					var s int64
					for i := 0; i < tc.k; i++ {
						s += int64(x[r*tc.k+i]) * int64(w[c*tc.k+i])
					}
					want := float32(s) * xs[r] * ws[c]
					if got := out[r*stride+c]; math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-3 {
						t.Fatalf("out[%d][%d] = %v, want %v", r, c, got, want)
					}
				}
			}
			// Unsigned activations, and weight blocks spaced out by a
			// larger stride than the packed block size.
			ua := make([]byte, len(x))
			for i := range x {
				ua[i] = byte(x[i]) ^ 0x80
			}
			packInt8Blocks8(unsafe.Slice((*int8)(unsafe.Pointer(&ua[0])), len(ua)), rows, tc.k, a)
			gap := 8 * tc.k * 2
			bw := make([]byte, tc.ncb*gap)
			for cb := 0; cb < tc.ncb; cb++ {
				copy(bw[cb*gap:], b[cb*8*tc.k:(cb+1)*8*tc.k])
			}
			gemmUsmmla8(&a[0], (*int8)(unsafe.Pointer(&bw[0])), tc.k, tc.ncb, gap, &xs[0], &ws[0], &out[0], stride)
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					var s int64
					for i := 0; i < tc.k; i++ {
						s += int64(ua[r*tc.k+i]) * int64(w[c*tc.k+i])
					}
					want := float32(s) * xs[r] * ws[c]
					if got := out[r*stride+c]; math.Abs(float64(got-want)) > 1e-3*math.Abs(float64(want))+1e-3 {
						t.Fatalf("unsigned out[%d][%d] = %v, want %v", r, c, got, want)
					}
				}
			}
		})
	}
}

// BenchmarkGemmSmmla8 times the tile on model shapes (512 rows) to read
// its MAC rate: weights resident in L1 (N=64) and streamed (N=4096).
func BenchmarkGemmSmmla8(b *testing.B) {
	if !cpu.ARM64.HasI8MM {
		b.Skip("no i8mm")
	}
	for _, shape := range []struct{ K, N, aOff int }{{512, 64, 0}, {512, 64, 64}, {512, 512, 0}, {512, 512, 64}, {512, 4096, 0}, {512, 4096, 64}, {2048, 512, 0}, {2048, 512, 64}} {
		K, N := shape.K, shape.N
		const T = 512
		a := make([]int8, T*K+shape.aOff)[shape.aOff:]
		w := make([]int8, N*K)
		xs := make([]float32, T)
		ws := make([]float32, N)
		out := make([]float32, T*N)
		b.Run(fmt.Sprintf("K=%d/N=%d/aOff=%d", K, N, shape.aOff), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for g := 0; g < T/8; g++ {
					gemmSmmla8(&a[g*8*K], &w[0], K, N/8, 8*K, &xs[g*8], &ws[0], &out[g*8*N], N)
				}
			}
			b.ReportMetric(float64(T)*float64(K)*float64(N)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
		})
	}
}
