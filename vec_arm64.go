//go:build arm64

package modernbert

// hasVecNeon reports whether the NEON row kernels (vec_arm64.s) run here;
// they need only the base SIMD.
const hasVecNeon = true

// quantizePack8 quantizes eight rows of k float32 (k a multiple of 8,
// rows stride floats apart) to ±levels with one scale per row, written
// straight into the eight-row block layout of packInt8Blocks8.
//
//go:noescape
func quantizePack8(x *float32, k, stride int, out *int8, scales *float32, levels float32)

// quantizeProbs8 writes eight rows of k values in [0, 1] (rows stride
// floats apart) as round(255·e) unsigned bytes in the eight-row block
// layout.
//
//go:noescape
func quantizeProbs8(e *float32, k, stride int, out *uint8)

// maxF32x4 is the maximum of n float32, n a positive multiple of 4.
//
//go:noescape
func maxF32x4(x *float32, n int) float32

// layerNormRowsNeon is layerNormRows for D a multiple of 8.
//
//go:noescape
func layerNormRowsNeon(x *float32, T, D int, w *float32, eps float32)

func init() {
	layerNormRowsFast = layerNormRowsNeon
	maxF32Fast, maxF32Align = maxF32x4, 4
}
