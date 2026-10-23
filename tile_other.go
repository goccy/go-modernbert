//go:build !arm64 && !amd64

package modernbert

const (
	tileCols             = 8
	tileAlign            = 8
	actOffset            = 0
	actLevels    float32 = 127
	valueLevels  float32 = 127
	probLevels   float32 = 255
	hasInt8Tiles         = false
)

func packTileB(q []int8, rows, k, offset int, out []byte, corr []int32) {
	panic("modernbert: no int8 tile on this architecture")
}

func quantizeTileA(x *float32, k, stride int, out *byte, scales *float32) {
	panic("modernbert: no int8 tile on this architecture")
}

func quantizeTileB(x []float32, rows, k, stride, offset int, levels float32, out []byte, corr []int32, scales []float32, tmp []int8) {
	panic("modernbert: no int8 tile on this architecture")
}

func quantizeTileProbs(e *float32, k, stride int, out *byte) {
	panic("modernbert: no int8 tile on this architecture")
}

func tileGemm(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int) {
	panic("modernbert: no int8 tile on this architecture")
}

func tileGemmProbs(a, b *byte, k, ncb, bStride int, xs, ws *float32, corr *int32, out *float32, outStride int) {
	panic("modernbert: no int8 tile on this architecture")
}
