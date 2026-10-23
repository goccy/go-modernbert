package modernbert

import "math"

// Float32 transcendental approximations for the hot loops. The relative
// error of expf is ~2e-7 (a degree-7 polynomial after range reduction to
// [-ln2/2, ln2/2]); erff follows Abramowitz & Stegun 7.1.26, absolute error
// below 1.5e-7, so GELU's y = x·Φ(x) is exact to float32 for practical
// purposes.

const (
	log2e32 = 1.4426950408889634
	ln2Hi32 = 0.693145751953125
	ln2Lo32 = 1.428606765330187e-06
)

// expf returns e^x for float32 x; inputs below -87 flush to 0, above 88
// saturate.
func expf(x float32) float32 {
	if x < -87 {
		return 0
	}
	if x > 88 {
		return math.MaxFloat32
	}
	n := float32(math.Floor(float64(x*log2e32 + 0.5)))
	r := x - n*ln2Hi32 - n*ln2Lo32
	// e^r on [-0.35, 0.35]: Taylor to degree 7 (the degree-8 remainder is
	// below 6e-9 of the value).
	p := 1 + r*(1+r*(0.5+r*(1.0/6+r*(1.0/24+r*(1.0/120+r*(1.0/720+r*(1.0/5040)))))))
	return math.Float32frombits(math.Float32bits(p) + uint32(int32(n))<<23)
}

// erff approximates erf(x) (A&S 7.1.26).
func erff(x float32) float32 {
	sign := float32(1)
	if x < 0 {
		sign, x = -1, -x
	}
	const p = 0.3275911
	const a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
	t := 1 / (1 + p*x)
	y := 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*expf(-x*x)
	return sign * y
}

// geluErf is the exact-form GELU, x·½(1+erf(x/√2)), with the float32
// approximations above.
func geluErf(x float32) float32 {
	return 0.5 * x * (1 + erff(x*(1/math.Sqrt2)))
}
