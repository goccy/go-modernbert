//go:build arm64

package modernbert

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/cpu"
)

// arm64: the 8×8 SMMLA tile (smmla_arm64.s). Both operands are packed in
// blocks of eight rows laid out as 2×8 matrix-multiply operands; the
// activations are signed, so there is no offset to correct, and every
// operand uses the full 8 bits. The row kernels need only NEON.

// selectKernels returns the kernels for k on this CPU.
func selectKernels(k Kernel) (*kernels, error) {
	switch k {
	case KernelAuto:
		k = KernelNone
		if cpu.ARM64.HasI8MM {
			k = KernelSMMLA
		}
	case KernelSMMLA, KernelNone:
	default:
		return nil, fmt.Errorf("modernbert: kernel %s is not available on arm64", k)
	}
	ks := &kernels{kernel: k, dot: dotTile(k)}
	if k == KernelNone {
		return ks, nil
	}
	ks.vecLayerNorm = layerNormRowsNeon
	ks.vecMax, ks.maxAlign = maxF32x4, 4
	ks.vecSoftmaxExp, ks.softmaxAlign = softmaxExpNeon, 4
	ks.vecGeglu, ks.gegluAlign = gegluErfNeon, 4
	ks.tile = &tileSet{
		cols: 8, align: 8,
		actOffset: 0, actLevels: 127, valueLevels: 127, probLevels: 255,
		gemm:       smmlaGemm,
		gemmProbs:  smmlaGemmProbs,
		packB:      smmlaPackB,
		quantA:     smmlaQuantA,
		quantB:     smmlaQuantB,
		quantProbs: quantizeProbs8,
	}
	return ks, nil
}

// packInt8Blocks8 packs rows (a multiple of 8) of k int8 values (k a
// multiple of 8) into blocks of eight rows: for each block and each step
// of eight along k, 64 bytes holding row pairs (0,1), (2,3), (4,5), (6,7),
// eight consecutive values of each row (the SMMLA operand layout).
func packInt8Blocks8(q []int8, rows, k int, out []byte) {
	steps := k / 8
	for g := 0; g < rows/tileRows; g++ {
		blk := out[g*tileRows*k : (g+1)*tileRows*k]
		for r := 0; r < tileRows; r++ {
			row := q[(g*tileRows+r)*k : (g*tileRows+r+1)*k]
			rb := unsafe.Slice((*byte)(unsafe.Pointer(&row[0])), k)
			for s := 0; s < steps; s++ {
				copy(blk[s*64+r*8:s*64+r*8+8], rb[8*s:8*s+8])
			}
		}
	}
}

func smmlaPackB(q []int8, rows, k, offset int, out []byte, corr []int32) {
	packInt8Blocks8(q, rows, k, out)
	clear(corr[:rows])
}

func smmlaQuantA(x *float32, k, stride int, out *byte, scales *float32) {
	quantizePack8(x, k, stride, (*int8)(unsafe.Pointer(out)), scales, 127)
}

func smmlaQuantB(x []float32, rows, k, stride, offset int, levels float32, out []byte, corr []int32, scales []float32, tmp []int8) {
	for g := 0; g < rows/tileRows; g++ {
		quantizePack8(&x[g*tileRows*stride], k, stride, (*int8)(unsafe.Pointer(&out[g*tileRows*k])), &scales[g*tileRows], levels)
	}
	clear(corr[:rows])
}

func smmlaGemm(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int) {
	gemmSmmla8((*int8)(unsafe.Pointer(a)), (*int8)(unsafe.Pointer(b)), k, ncb, bStride, xs, ws, out, outStride)
}

func smmlaGemmProbs(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int) {
	gemmUsmmla8(a, (*int8)(unsafe.Pointer(b)), k, ncb, bStride, xs, ws, out, outStride)
}
