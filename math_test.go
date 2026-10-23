package modernbert

import (
	"math"
	"math/rand"
	"testing"
)

func TestSoftmaxExpFast(t *testing.T) {
	if softmaxExpFast == nil {
		t.Skip("no vector softmax")
	}
	rng := rand.New(rand.NewSource(7))
	n := 6 * softmaxExpAlign
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 20
	}
	x[1], x[2] = -200, 30 // clamped below; near the maximum
	maxS := float32(30)
	want := make([]float32, n)
	var wantSum float64
	for i, v := range x {
		want[i] = float32(math.Exp(float64(v - maxS)))
		wantSum += float64(want[i])
	}
	got := append([]float32(nil), x...)
	sum := softmaxExpFast(&got[0], n, maxS)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 2e-6*float64(want[i])+1e-30 {
			t.Fatalf("exp[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	if math.Abs(float64(sum)-wantSum) > 2e-6*wantSum {
		t.Fatalf("sum = %v, want %v", sum, wantSum)
	}
}

func TestGegluErfFast(t *testing.T) {
	if gegluErfFast == nil {
		t.Skip("no vector GeGLU")
	}
	rng := rand.New(rand.NewSource(8))
	n := 5 * gegluErfAlign
	x := make([]float32, n)
	g := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * 4
		g[i] = float32(rng.NormFloat64())
	}
	x[0], x[1], x[2] = 0, -12, 12
	got := make([]float32, n)
	gegluErfFast(&got[0], &x[0], &g[0], n)
	for i := range got {
		xf := float64(x[i])
		want := 0.5 * xf * (1 + math.Erf(xf/math.Sqrt2)) * float64(g[i])
		if math.Abs(float64(got[i])-want) > 1e-6*math.Abs(want)+2e-6 {
			t.Fatalf("a[%d] = %v, want %v (x=%v g=%v)", i, got[i], want, x[i], g[i])
		}
	}
}
