//go:build amd64

package modernbert

import (
	"fmt"

	"golang.org/x/sys/cpu"
)

// amd64: the activations stay row-major as unsigned bytes (x + actOffset)
// and are broadcast four at a time; the weights are packed per block of
// cols rows as [k/4 steps][cols rows][4 values], one vector per step.
// VPDPBUSD (AVX-512 VNNI) takes full 8-bit operands. The AVX2 pair
// VPMADDUBSW/VPMADDWD sums two u8·s8 products into int16, so there one
// side of each product gives up a bit: the activations are 7-bit (±63,
// offset 64) against 8-bit weights and keys, and the values are 7-bit
// (±63) against 8-bit softmax weights. The row kernels need AVX2.

// selectKernels returns the kernels for k on this CPU. A kernel named
// explicitly is used whether or not the CPU advertises it.
func selectKernels(k Kernel) (*kernels, error) {
	switch k {
	case KernelAuto:
		switch {
		case cpu.X86.HasAVX512F && cpu.X86.HasAVX512VNNI && cpu.X86.HasAVX512BW:
			k = KernelVNNI
		case cpu.X86.HasAVX2:
			k = KernelAVX2
		default:
			k = KernelNone
		}
	case KernelAVX2, KernelVNNI, KernelNone:
	default:
		return nil, fmt.Errorf("modernbert: kernel %s is not available on amd64", k)
	}
	ks := &kernels{kernel: k, dot: dotTile(k)}
	if k == KernelNone {
		return ks, nil
	}
	ks.vecLayerNorm = layerNormRowsAVX2
	ks.vecMax, ks.maxAlign = maxF32AVX2, 8
	ks.vecSoftmaxExp, ks.softmaxAlign = softmaxExpAVX2, 8
	ks.vecGeglu, ks.gegluAlign = gegluErfAVX2, 8
	t := &tileSet{packB: nil}
	if k == KernelVNNI {
		*t = tileSet{cols: 16, align: 16, actOffset: 128, actLevels: 127, valueLevels: 127, probLevels: 255, gemm: gemmTileVNNI}
	} else {
		*t = tileSet{cols: 8, align: 8, actOffset: 64, actLevels: 63, valueLevels: 63, probLevels: 255, gemm: gemmTileAVX2}
	}
	t.gemmProbs = t.gemm
	t.packB = func(q []int8, rows, k, offset int, out []byte, corr []int32) {
		packRows4(q, rows, k, offset, t.cols, out, corr)
	}
	t.quantA = func(x *float32, k, stride int, out *byte, scales *float32) {
		quantizeRowsU8AVX2(x, k, stride, out, scales, t.actLevels, int32(t.actOffset))
	}
	t.quantB = func(x []float32, rows, k, stride, offset int, levels float32, out []byte, corr []int32, scales []float32, tmp []int8) {
		q := tmp[:rows*k]
		for r := 0; r < rows; r++ {
			quantizeRowsInt8(x[r*stride:r*stride+k], 1, k, q[r*k:(r+1)*k], scales[r:r+1], levels)
		}
		packRows4(q, rows, k, offset, t.cols, out, corr)
	}
	t.quantProbs = func(e *float32, k, stride int, out *byte) { quantizeProbsU8AVX2(e, k, stride, out, t.probLevels) }
	ks.tile = t
	return ks, nil
}

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

// packRows4 packs rows (a multiple of cols) of k int8 (k a multiple of 4)
// into blocks of cols rows, [k/4][cols][4], and sets corr (one per row)
// to offset·Σ row.
func packRows4(q []int8, rows, k, offset, cols int, out []byte, corr []int32) {
	for cb := 0; cb < rows/cols; cb++ {
		blk := out[cb*cols*k : (cb+1)*cols*k]
		for c := 0; c < cols; c++ {
			row := q[(cb*cols+c)*k : (cb*cols+c+1)*k]
			var s int32
			for i, v := range row {
				blk[i/4*4*cols+c*4+i%4] = byte(v)
				s += int32(v)
			}
			corr[cb*cols+c] = int32(offset) * s
		}
	}
}
