package modernbert

import (
	"math"
	"math/rand"
	"testing"
)

// TestQuantizeTileA compares the architecture's activation quantizer
// with the scalar one: scales exact, bytes within one unit (the vector
// kernels round ties differently).
func TestQuantizeTileA(t *testing.T) {
	if !hasInt8Tiles {
		t.Skip("no int8 tile")
	}
	const k, stride = 64, 80
	rng := rand.New(rand.NewSource(3))
	x := make([]float32, 8*stride)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 10
	}
	clear(x[3*stride : 3*stride+k]) // an all-zero row
	x[5*stride] = 0.5
	got := make([]byte, 8*k)
	gotS := make([]float32, 8)
	quantizeTileA(&x[0], k, stride, &got[0], &gotS[0])
	for r := 0; r < 8; r++ {
		row := x[r*stride : r*stride+k]
		var amax float32
		for _, v := range row {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		wantS := amax / actLevels
		if math.Abs(float64(gotS[r]-wantS)) > 1e-6*float64(wantS) {
			t.Fatalf("scale[%d] = %v, want %v", r, gotS[r], wantS)
		}
		for c, v := range row {
			want := 0
			if wantS > 0 {
				want = int(math.Round(float64(v / wantS)))
			}
			g := int(got[tileAIndex(r, c, k)]) - actOffset
			if actOffset == 0 {
				g = int(int8(got[tileAIndex(r, c, k)]))
			}
			if d := g - want; d < -1 || d > 1 {
				t.Fatalf("row %d col %d = %d, want %d", r, c, g, want)
			}
		}
	}
}

// tileAIndex is where row r, value c of an eight-row block of k lands in
// the architecture's activation layout.
func tileAIndex(r, c, k int) int {
	if actOffset == 0 { // arm64: eight-row SMMLA blocks
		return c/8*64 + r*8 + c%8
	}
	return r*k + c
}

func TestMaxF32Fast(t *testing.T) {
	if maxF32Fast == nil {
		t.Skip("no vector max")
	}
	for _, n := range []int{maxF32Align, 4 * maxF32Align, 9 * maxF32Align} {
		x := make([]float32, n)
		for i := range x {
			x[i] = -float32(i) - 1
		}
		x[n-2] = 7
		if got := maxF32Fast(&x[0], n); got != 7 {
			t.Fatalf("n=%d: max = %v, want 7", n, got)
		}
	}
}

func TestLayerNormRowsFast(t *testing.T) {
	if layerNormRowsFast == nil {
		t.Skip("no vector LayerNorm")
	}
	const T, D = 5, 64
	rng := rand.New(rand.NewSource(4))
	x := make([]float32, T*D)
	w := make([]float32, D)
	for i := range x {
		x[i] = float32(rng.NormFloat64())*3 + 1
	}
	for i := range w {
		w[i] = float32(rng.Float64()) + 0.5
	}
	want := append([]float32(nil), x...)
	layerNormRows(want, T, D, w, 1e-5)
	layerNormRowsFast(&x[0], T, D, &w[0], 1e-5)
	for i := range want {
		if math.Abs(float64(x[i]-want[i])) > 1e-5 {
			t.Fatalf("x[%d] = %v, want %v", i, x[i], want[i])
		}
	}
}
