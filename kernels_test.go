package modernbert

import (
	"os"
	"testing"
)

// testKernel is the kernel the tests and benchmarks run on: KernelAuto,
// or the one MODERNBERT_KERNEL names (none / smmla / avx2 / vnni), for
// comparing kernels on one machine or running one the CPU does not
// advertise (AVX2 under Rosetta, say).
func testKernel(tb testing.TB) Kernel {
	switch s := os.Getenv("MODERNBERT_KERNEL"); s {
	case "", "auto":
		return KernelAuto
	case "none":
		return KernelNone
	case "smmla":
		return KernelSMMLA
	case "avx2":
		return KernelAVX2
	case "vnni":
		return KernelVNNI
	default:
		tb.Fatalf("MODERNBERT_KERNEL=%q: unknown kernel", s)
		return KernelAuto
	}
}

// testKernels is the kernel set of testKernel, skipping the test where
// the architecture has none.
func testKernels(tb testing.TB) *kernels {
	ks, err := selectKernels(testKernel(tb))
	if err != nil {
		tb.Skip(err)
	}
	return ks
}
