package modernbert

import (
	"fmt"
	"math"
)

// Kernel selects the SIMD kernels a Model runs on (Options.Kernel).
type Kernel int

const (
	// KernelAuto picks the best kernels the CPU advertises.
	KernelAuto Kernel = iota
	// KernelNone runs everything in Go (the scalar paths).
	KernelNone
	// KernelSMMLA is the arm64 int8 tile on the i8mm extension with the
	// NEON row kernels.
	KernelSMMLA
	// KernelAVX2 is the amd64 int8 tile on AVX2 with the AVX2 row
	// kernels.
	KernelAVX2
	// KernelVNNI is the amd64 int8 tile on AVX-512 VNNI with the AVX2 row
	// kernels.
	KernelVNNI
)

func (k Kernel) String() string {
	switch k {
	case KernelAuto:
		return "auto"
	case KernelNone:
		return "none"
	case KernelSMMLA:
		return "smmla"
	case KernelAVX2:
		return "avx2"
	case KernelVNNI:
		return "vnni"
	}
	return fmt.Sprintf("Kernel(%d)", int(k))
}

// kernels is the set of SIMD kernels a Model runs on: the int8 tile (nil
// where there is none) and the row kernels (nil where there are none; the
// Go versions take over).
type kernels struct {
	kernel Kernel
	tile   *tileSet
	// dot is the dot-product tile of the linear layers a tile cannot run
	// (kernel.go); a zero tile means scalar.
	dot tile
	// layerNorm is layerNormRows for D a multiple of 8.
	vecLayerNorm func(x *float32, T, D int, w *float32, eps float32)
	// maxF32 is the maximum of n float32, n a positive multiple of
	// maxAlign.
	vecMax   func(x *float32, n int) float32
	maxAlign int
	// softmaxExp replaces x[:n] by exp(x - max) and returns the sum, n a
	// positive multiple of softmaxAlign.
	vecSoftmaxExp func(x *float32, n int, max float32) float32
	softmaxAlign  int
	// gegluErf sets a[:n] = gelu_erf(x) * gate, n a positive multiple of
	// gegluAlign.
	vecGeglu   func(a, x, gate *float32, n int)
	gegluAlign int
}

// The int8 tile: the linear layers and attention run on one GEMM shape,
// eight activation rows (A, int8 or unsigned with an offset) against
// blocks of cols weight rows (B, int8, packed per architecture), with
// int32 accumulation over the whole K and one scale per row and per
// column applied at the end:
//
//	out[r][c] = xs[r] * ws[c] * (Σ_k A[r][k]·B[c][k] - corr[c])
//
// corr[c] = actOffset·Σ_k B[c][k] cancels the offset an architecture adds
// to its activations so that they are unsigned bytes; it is zero where
// the activations are signed.
type tileSet struct {
	cols  int // weight rows per block
	align int // the row count sequences are padded to
	// Activations are quantized to ±actLevels and stored plus actOffset;
	// attention values to ±valueLevels; softmax weights to [0, probLevels].
	actOffset              int
	actLevels, valueLevels float32
	probLevels             float32
	// gemm multiplies one block of eight activation rows (a) by ncb blocks
	// of weight rows (b, bStride bytes apart), k a positive multiple of 8,
	// into out (outStride floats per row). gemmProbs is the same for
	// unsigned softmax weights.
	gemm      func(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int)
	gemmProbs func(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int)
	// packB packs rows (a multiple of cols) of k int8 into blocks and sets
	// corr to offset·Σ row.
	packB func(q []int8, rows, k, offset int, out []byte, corr []int32)
	// quantA quantizes eight rows of k float32 (rows stride floats apart)
	// into one activation block with one scale per row.
	quantA func(x *float32, k, stride int, out *byte, scales *float32)
	// quantB quantizes rows (a multiple of cols) of k float32 to ±levels
	// into packed blocks with one scale per row and corr; tmp holds at
	// least rows*k int8 where the architecture needs it.
	quantB func(x []float32, rows, k, stride, offset int, levels float32, out []byte, corr []int32, scales []float32, tmp []int8)
	// quantProbs writes eight rows of k values in [0, 1] as
	// round(probLevels·e) bytes in the activation block layout.
	quantProbs func(e *float32, k, stride int, out *byte)
}

// tileRows is the activation rows one tile covers.
const tileRows = 8

func roundUp(n, m int) int { return (n + m - 1) / m * m }

// softmaxExp replaces x by exp(x - max x) and returns the sum.
func (k *kernels) softmaxExp(x []float32) float32 {
	maxS := float32(math.Inf(-1))
	n := 0
	if k.vecMax != nil {
		n = len(x) &^ (k.maxAlign - 1)
		if n > 0 {
			maxS = k.vecMax(&x[0], n)
		}
	}
	for _, v := range x[n:] {
		if v > maxS {
			maxS = v
		}
	}
	return k.softmaxExpRow(x, maxS)
}

// softmaxExpRow replaces x by exp(x - maxS) and returns the sum, the
// vector kernel on the aligned part and scalar on the tail.
func (k *kernels) softmaxExpRow(x []float32, maxS float32) float32 {
	var sum float32
	n := 0
	if k.vecSoftmaxExp != nil {
		n = len(x) &^ (k.softmaxAlign - 1)
		if n > 0 {
			sum = k.vecSoftmaxExp(&x[0], n, maxS)
		}
	}
	for i := n; i < len(x); i++ {
		e := expf(x[i] - maxS)
		x[i] = e
		sum += e
	}
	return sum
}

// layerNormRows normalizes T rows of D in place with the vector kernel
// where it applies.
func (k *kernels) layerNormRows(x []float32, T, D int, w []float32, eps float32) {
	if k.vecLayerNorm != nil && D%8 == 0 && T > 0 {
		k.vecLayerNorm(&x[0], T, D, &w[0], eps)
		return
	}
	layerNormRows(x, T, D, w, eps)
}

// geglu sets a = gelu_erf(x) * gate over n values.
func (k *kernels) geglu(a, x, gate []float32, n int) {
	i := 0
	if k.vecGeglu != nil {
		i = n &^ (k.gegluAlign - 1)
		if i > 0 {
			k.vecGeglu(&a[0], &x[0], &gate[0], i)
		}
	}
	for ; i < n; i++ {
		a[i] = geluErf(x[i]) * gate[i]
	}
}
