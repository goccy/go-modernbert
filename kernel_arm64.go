//go:build arm64

package modernbert

import "golang.org/x/sys/cpu"

// dot4x4Int8 is the SDOT tile kernel (kernel_arm64.s): out[i*4+j] is the
// int32 dot product of x row i and w row j over k values, k a positive
// multiple of 16.
//
//go:noescape
func dot4x4Int8(x, w *int8, k, xStride, wStride int, out *int32)

// dotTile is the dot-product tile of kernel k: SDOT 4×4 when the dotprod
// feature is present (and k is not KernelNone), else none.
func dotTile(k Kernel) tile {
	if k != KernelNone && cpu.ARM64.HasASIMDDP {
		return tile{rows: 4, cols: 4, kAlign: 16, run: dot4x4Int8}
	}
	return tile{}
}
