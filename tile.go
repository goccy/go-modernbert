package modernbert

// The int8 tile: the linear layers and attention run on one GEMM shape,
// eight activation rows (A, int8 or unsigned with an offset) against
// blocks of tileCols weight rows (B, int8, packed per architecture), with
// int32 accumulation over the whole K and one scale per row and per
// column applied at the end:
//
//	out[r][c] = xs[r] * ws[c] * (Σ_k A[r][k]·B[c][k] - corr[c])
//
// corr[c] = actOffset·Σ_k B[c][k] cancels the offset an architecture adds
// to its activations so that they are unsigned bytes; it is zero where
// the activations are signed.
//
// Per architecture (tile_*.go): tileCols, tileAlign (the row count the
// sequence is padded to), actOffset and actLevels (activations are
// quantized to ±actLevels before the offset), probLevels (softmax weights
// are quantized to [0, probLevels]), hasInt8Tiles, and the functions
// packTileB, quantizeTileA, quantizeTileB, quantizeTileProbs, tileGemm and
// tileGemmProbs.

// tileRows is the activation rows one tile covers.
const tileRows = 8

// Vector row kernels an architecture may provide (nil otherwise):
// layerNormRowsFast is layerNormRows for D a multiple of 8, maxF32Fast the
// maximum of n float32 with n a positive multiple of maxF32Align.
var (
	layerNormRowsFast func(x *float32, T, D int, w *float32, eps float32)
	maxF32Fast        func(x *float32, n int) float32
	maxF32Align       int
	// softmaxExpFast replaces x[:n] by exp(x - max) and returns the sum,
	// n a positive multiple of softmaxExpAlign.
	softmaxExpFast  func(x *float32, n int, max float32) float32
	softmaxExpAlign int
	// gegluErfFast sets a[:n] = gelu_erf(x) * g, n a positive multiple of
	// gegluErfAlign.
	gegluErfFast  func(a, x, g *float32, n int)
	gegluErfAlign int
)

func roundUp(n, m int) int { return (n + m - 1) / m * m }
