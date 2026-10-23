package modernbert

import (
	"math"
	"testing"
)

func TestExpfErff(t *testing.T) {
	t.Parallel()
	for x := float32(-20); x <= 20; x += 0.013 {
		if got, want := float64(expf(x)), math.Exp(float64(x)); math.Abs(got-want) > 1e-6*want {
			t.Fatalf("expf(%v) = %v, want %v", x, got, want)
		}
		if got, want := float64(erff(x)), math.Erf(float64(x)); math.Abs(got-want) > 1e-6 {
			t.Fatalf("erff(%v) = %v, want %v", x, got, want)
		}
	}
	if expf(-100) != 0 || expf(100) != math.MaxFloat32 {
		t.Fatalf("expf clamps: %v %v", expf(-100), expf(100))
	}
}
