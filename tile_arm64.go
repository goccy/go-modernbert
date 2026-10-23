//go:build arm64

package modernbert

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

// arm64: the 8×8 SMMLA tile (smmla_arm64.s). Both operands are packed in
// blocks of eight rows laid out as 2×8 matrix-multiply operands; the
// activations are signed, so there is no offset to correct, and every
// operand uses the full 8 bits.

const (
	tileCols            = 8
	tileAlign           = 8
	actOffset           = 0
	actLevels   float32 = 127
	valueLevels float32 = 127
	probLevels  float32 = 255
)

var hasInt8Tiles = cpu.ARM64.HasI8MM && hasVecNeon

// packTileB packs rows (a multiple of 8) of k int8 (k a multiple of 8)
// into eight-row blocks; corr (one per row) is zero.
func packTileB(q []int8, rows, k, offset int, out []byte, corr []int32) {
	packInt8Blocks8(q, rows, k, out)
	clear(corr[:rows])
}

// quantizeTileA quantizes eight rows of k float32 (rows stride floats
// apart) into one packed block with one scale per row.
func quantizeTileA(x *float32, k, stride int, out *byte, scales *float32) {
	quantizePack8(x, k, stride, (*int8)(unsafe.Pointer(out)), scales, actLevels)
}

// quantizeTileB quantizes rows (a multiple of 8) of k float32 (rows stride
// floats apart) to ±levels into packed blocks with one scale per row; corr
// is zero. tmp is unused.
func quantizeTileB(x []float32, rows, k, stride, offset int, levels float32, out []byte, corr []int32, scales []float32, tmp []int8) {
	for g := 0; g < rows/tileRows; g++ {
		quantizePack8(&x[g*tileRows*stride], k, stride, (*int8)(unsafe.Pointer(&out[g*tileRows*k])), &scales[g*tileRows], levels)
	}
	clear(corr[:rows])
}

// quantizeTileProbs writes eight rows of k values in [0, 1] as
// round(255·e) bytes in the packed block layout.
func quantizeTileProbs(e *float32, k, stride int, out *byte) {
	quantizeProbs8(e, k, stride, out)
}

func tileGemm(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int) {
	gemmSmmla8((*int8)(unsafe.Pointer(a)), (*int8)(unsafe.Pointer(b)), k, ncb, bStride, xs, ws, out, outStride)
}

func tileGemmProbs(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int) {
	gemmUsmmla8(a, (*int8)(unsafe.Pointer(b)), k, ncb, bStride, xs, ws, out, outStride)
}
