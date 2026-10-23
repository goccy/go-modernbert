//go:build amd64

package modernbert

// AVX2 row kernels for the softmax exponential and GeGLU (math_amd64.s).

//go:noescape
func softmaxExpAVX2(x *float32, n int, max float32) float32

//go:noescape
func gegluErfAVX2(a, x, gate *float32, n int)
