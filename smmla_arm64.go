//go:build arm64

package modernbert

// gemmSmmla8 is the 8×8 SMMLA tile kernel (smmla_arm64.s): one block of
// eight packed activation rows against ncb blocks of eight packed weight
// rows bStride bytes apart, k a positive multiple of 8, scaled per row
// (xs) and per column (ws) into float32 out with outStride floats per row.
//
//go:noescape
func gemmSmmla8(a, b *int8, k, ncb, bStride int, xs, ws *float32, out *float32, outStride int)

// gemmUsmmla8 is gemmSmmla8 with unsigned activation bytes (USMMLA).
//
//go:noescape
func gemmUsmmla8(a *uint8, b *int8, k, ncb, bStride int, xs, ws *float32, out *float32, outStride int)
