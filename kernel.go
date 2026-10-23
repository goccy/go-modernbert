package modernbert

import "unsafe"

// tile is an architecture's int8 dot-product tile kernel: run fills
// out[i*cols+j] with the dot product of x row i and w row j over k values
// (k a multiple of kAlign). With unsignedX the kernel multiplies unsigned
// activation bytes (x + 128) by signed weights, and the caller corrects
// each product by 128·Σw of the weight row. A zero tile (rows == 0) means
// the scalar fallback.
type tile struct {
	rows, cols int
	kAlign     int
	unsignedX  bool
	run        func(x, w *int8, k, xStride, wStride int, out *int32)
}

// kernel is the tile this process runs, chosen once from the CPU features.
var kernel = tileKernel()

func unsafePointer(p *int8) unsafe.Pointer { return unsafe.Pointer(p) }
