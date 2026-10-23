//go:build arm64

package modernbert

// dotF32Tile4x4 (attn_arm64.s): out[i*4+j] is the float32 dot product of a
// row i and b row j over k values (k a positive multiple of 4); strides in
// elements.
//
//go:noescape
func dotF32Tile4x4(a, b *float32, k, aStride, bStride int, out *float32)

// axpyTile4x16 (attn_arm64.s): out[i*outStride+d] = Σ_{j<n}
// p[i*pStride+j] * v[j*vStride+d] for i < 4, d < 16; n ≥ 1.
//
//go:noescape
func axpyTile4x16(p *float32, pStride int, v *float32, vStride int, n int, out *float32, outStride int)

const hasF32Tiles = true
