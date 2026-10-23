//go:build amd64

package modernbert

import (
	"os"

	"golang.org/x/sys/cpu"
)

// dot2x4Int8AVX2 (kernel_amd64.s): out[i*4+j] is the int32 dot product of
// x row i (i < 2) and w row j (j < 4) over k values, k a positive multiple
// of 32.
//
//go:noescape
func dot2x4Int8AVX2(x, w *int8, k, xStride, wStride int, out *int32)

// dot4x4U8S8VNNI (kernel_amd64.s): out[i*4+j] is the int32 dot product of
// the unsigned x row i and the signed w row j (i, j < 4) over k values, k a
// positive multiple of 64.
//
//go:noescape
func dot4x4U8S8VNNI(xu *uint8, w *int8, k, xStride, wStride int, out *int32)

var (
	hasAVX2 = cpu.X86.HasAVX2
	hasVNNI = cpu.X86.HasAVX512F && cpu.X86.HasAVX512VNNI && cpu.X86.HasAVX512BW
)

// tileKernel describes the int8 tile the CPU runs: VNNI (4×4, unsigned
// activations) when available, else AVX2 (2×4), else none. MODERNBERT_KERNEL
// (vnni / avx2 / none) overrides the choice, for comparing the kernels on
// one machine or running one the CPU does not advertise.
func tileKernel() tile {
	force := os.Getenv("MODERNBERT_KERNEL")
	switch {
	case force == "none":
		return tile{}
	case hasVNNI && force == "" || force == "vnni":
		return tile{rows: 4, cols: 4, kAlign: 64, unsignedX: true, run: func(x, w *int8, k, xs, ws int, out *int32) {
			dot4x4U8S8VNNI((*uint8)(unsafePointer(x)), w, k, xs, ws, out)
		}}
	case hasAVX2 && force == "" || force == "avx2":
		return tile{rows: 2, cols: 4, kAlign: 32, run: dot2x4Int8AVX2}
	}
	return tile{}
}
