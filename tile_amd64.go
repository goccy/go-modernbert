//go:build amd64

package modernbert

import (
	"math"
	"os"
	"unsafe"

	"golang.org/x/sys/cpu"
)

// amd64: the activations stay row-major as unsigned bytes (x + actOffset)
// and are broadcast four at a time; the weights are packed per block of
// tileCols rows as [k/4 steps][tileCols rows][4 values], one vector per
// step. VPDPBUSD (AVX-512 VNNI) takes full 8-bit operands. The AVX2 pair
// VPMADDUBSW/VPMADDWD sums two u8·s8 products into int16, so there one
// side of each product gives up a bit: the activations are 7-bit (±63,
// offset 64) against 8-bit weights and keys, and the values are 7-bit
// (±63) against 8-bit softmax weights.

var (
	tileCols     int
	tileAlign    int
	actOffset    int
	actLevels    float32
	valueLevels  float32
	probLevels   float32
	hasInt8Tiles bool
	tileGemmFn   func(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int)
)

// gemmTileAVX2 (tile_amd64.s): eight rows against ncb blocks of eight
// weight rows, k a positive multiple of 4.
//
//go:noescape
func gemmTileAVX2(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int)

// gemmTileVNNI (tile_amd64.s): eight rows against ncb blocks of sixteen
// weight rows, k a positive multiple of 4.
//
//go:noescape
func gemmTileVNNI(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int)

// init picks the tile the CPU runs: VNNI, else AVX2, else none.
// MODERNBERT_KERNEL (vnni / avx2 / none) overrides the choice, for
// comparing the kernels on one machine or running one the CPU does not
// advertise.
func init() {
	force := os.Getenv("MODERNBERT_KERNEL")
	vnni := cpu.X86.HasAVX512F && cpu.X86.HasAVX512VNNI && cpu.X86.HasAVX512BW
	switch {
	case force == "none":
	case vnni && force == "" || force == "vnni":
		tileCols, tileAlign, actOffset, actLevels, valueLevels, probLevels = 16, 16, 128, 127, 127, 255
		tileGemmFn, hasInt8Tiles = gemmTileVNNI, true
	case cpu.X86.HasAVX2 && force == "" || force == "avx2":
		tileCols, tileAlign, actOffset, actLevels, valueLevels, probLevels = 8, 8, 64, 63, 63, 255
		tileGemmFn, hasInt8Tiles = gemmTileAVX2, true
	}
	if hasInt8Tiles {
		layerNormRowsFast = layerNormRowsAVX2
		maxF32Fast, maxF32Align = maxF32AVX2, 8
	}
}

// packTileB packs rows (a multiple of tileCols) of k int8 (k a multiple
// of 4) into blocks of tileCols rows, [k/4][tileCols][4], and sets corr
// (one per row) to offset·Σ row.
func packTileB(q []int8, rows, k, offset int, out []byte, corr []int32) {
	for cb := 0; cb < rows/tileCols; cb++ {
		blk := out[cb*tileCols*k : (cb+1)*tileCols*k]
		for c := 0; c < tileCols; c++ {
			row := q[(cb*tileCols+c)*k : (cb*tileCols+c+1)*k]
			var s int32
			for i, v := range row {
				blk[i/4*4*tileCols+c*4+i%4] = byte(v)
				s += int32(v)
			}
			corr[cb*tileCols+c] = int32(offset) * s
		}
	}
}

// quantizeTileA quantizes eight rows of k float32 (rows stride floats
// apart) to ±actLevels with one scale per row, stored row-major as
// unsigned bytes plus actOffset.
func quantizeTileA(x *float32, k, stride int, out *byte, scales *float32) {
	if hasInt8Tiles && k%8 == 0 {
		quantizeRowsU8AVX2(x, k, stride, out, scales, actLevels, int32(actOffset))
		return
	}
	xs := unsafe.Slice(x, (tileRows-1)*stride+k)
	o := unsafe.Slice(out, tileRows*k)
	sc := unsafe.Slice(scales, tileRows)
	for r := 0; r < tileRows; r++ {
		row := xs[r*stride : r*stride+k]
		var amax float32
		for _, v := range row {
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}
		s := amax / actLevels
		inv := float32(0)
		if s > 0 {
			inv = 1 / s
		}
		sc[r] = s
		q := o[r*k : (r+1)*k]
		for i, v := range row {
			q[i] = byte(int(roundHalfAway(v*inv)) + actOffset)
		}
	}
}

// quantizeTileB quantizes rows (a multiple of tileCols) of k float32 (rows
// stride floats apart) to ±levels with one scale per row into tmp (at
// least rows*k) and packs them with packTileB.
func quantizeTileB(x []float32, rows, k, stride, offset int, levels float32, out []byte, corr []int32, scales []float32, tmp []int8) {
	q := tmp[:rows*k]
	for r := 0; r < rows; r++ {
		quantizeRowsInt8(x[r*stride:r*stride+k], 1, k, q[r*k:(r+1)*k], scales[r:r+1], levels)
	}
	packTileB(q, rows, k, offset, out, corr)
}

// quantizeTileProbs writes eight rows of k values in [0, 1] (rows stride
// floats apart) as round(probLevels·e) bytes, row-major.
func quantizeTileProbs(e *float32, k, stride int, out *byte) {
	if hasInt8Tiles && k%8 == 0 {
		quantizeProbsU8AVX2(e, k, stride, out, probLevels)
		return
	}
	es := unsafe.Slice(e, (tileRows-1)*stride+k)
	o := unsafe.Slice(out, tileRows*k)
	for r := 0; r < tileRows; r++ {
		row := es[r*stride : r*stride+k]
		q := o[r*k : (r+1)*k]
		for i, v := range row {
			q[i] = byte(roundHalfAway(v * probLevels))
		}
	}
}

func roundHalfAway(f float32) float32 {
	return float32(math.Round(float64(f)))
}

func tileGemm(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int) {
	tileGemmFn(a, b, k, ncb, bStride, xs, ws, corr, out, outStride)
}

func tileGemmProbs(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int) {
	tileGemmFn(a, b, k, ncb, bStride, xs, ws, corr, out, outStride)
}
