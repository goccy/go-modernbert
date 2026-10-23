//go:build arm64

package modernbert

// NEON row kernels for the softmax exponential and GeGLU (math_arm64.s).

//go:noescape
func softmaxExpNeon(x *float32, n int, max float32) float32

//go:noescape
func gegluErfNeon(a, x, gate *float32, n int)

func init() {
	softmaxExpFast, softmaxExpAlign = softmaxExpNeon, 4
	gegluErfFast, gegluErfAlign = gegluErfNeon, 4
}
