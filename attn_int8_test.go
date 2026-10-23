package modernbert

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestAttentionInt8 compares the int8 attention with the float32 one on
// random projections, global and sliding-window, with a sequence that is
// not a multiple of the block.
func TestAttentionInt8(t *testing.T) {
	if !hasInt8Tiles {
		t.Skip("no int8 tile kernels")
	}
	m := &Model{Config: Config{NEmbd: 128, NHead: 2, NFF: 256, NLayer: 3, RopeBase: 10000, RopeBaseSW: 10000, Window: 128, Pattern: 3}, threads: 3}
	m.int8Attention = true
	for _, T := range []int{203, 64} {
		for il := 0; il < 2; il++ {
			t.Run(fmt.Sprintf("T=%d/layer=%d", T, il), func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(T + il)))
				ws := m.newWorkspace(T)
				defer ws.pool.close()
				D := m.NEmbd
				for _, buf := range [][]float32{ws.q, ws.k, ws.v} {
					for i := range buf[:T*D] {
						buf[i] = float32(rng.NormFloat64())
					}
				}
				want := make([]float32, T*D)
				m.attention(ws.q, ws.k, ws.v, T, il, want, ws)
				got := make([]float32, T*D)
				m.attentionInt8(ws.q, ws.k, ws.v, T, il, got, ws)
				var maxRef, maxDiff float64
				for i := range want {
					maxRef = math.Max(maxRef, math.Abs(float64(want[i])))
					maxDiff = math.Max(maxDiff, math.Abs(float64(want[i]-got[i])))
				}
				t.Logf("max|diff| %.3e of max|ref| %.3e", maxDiff, maxRef)
				// Random projections give near-uniform attention, the
				// coarsest case for 8-bit weights.
				if maxDiff > 3e-2*maxRef {
					t.Fatalf("int8 attention diverges: %.3e > 3e-2 × %.3e", maxDiff, maxRef)
				}
			})
		}
	}
}
