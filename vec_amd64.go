//go:build amd64

package modernbert

// The AVX2 row kernels (vec_amd64.s); selected by hasInt8Tiles in
// tile_amd64.go, since both tiles need AVX2.

//go:noescape
func quantizeRowsU8AVX2(x *float32, k, stride int, out *byte, scales *float32, levels float32, offset int32)

//go:noescape
func quantizeProbsU8AVX2(e *float32, k, stride int, out *byte, levels float32)

//go:noescape
func maxF32AVX2(x *float32, n int) float32

//go:noescape
func layerNormRowsAVX2(x *float32, T, D int, w *float32, eps float32)
