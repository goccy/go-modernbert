//go:build amd64

package modernbert

import (
	"os"

	"golang.org/x/sys/cpu"
)

// AVX2 row kernels for the softmax exponential and GeGLU (math_amd64.s).

//go:noescape
func softmaxExpAVX2(x *float32, n int, max float32) float32

//go:noescape
func gegluErfAVX2(a, x, gate *float32, n int)

func init() {
	force := os.Getenv("MODERNBERT_KERNEL")
	if force == "none" || !(cpu.X86.HasAVX2 || force == "avx2" || force == "vnni") {
		return
	}
	softmaxExpFast, softmaxExpAlign = softmaxExpAVX2, 8
	gegluErfFast, gegluErfAlign = gegluErfAVX2, 8
}
