package modernbert

// Int8 tile path of the linear layers: per-row int8 weights packed per
// architecture (Linear.Packed, with Linear.Corr), per-token activations
// quantized into blocks of eight rows, and the tile GEMM (tile.go).

const int8BlockRows = tileRows

// matmulTile is x (T rows of l.In, float32) times l on the int8 tile
// path: activations quantized per row into ws.vy (rows padded to a
// multiple of eight), then each activation block against the weight
// blocks of l.Packed, the weight blocks split across the pool.
func (m *Model) matmulTile(x []float32, T int, l Linear, ws *workspace, out []float32) {
	kt := m.k.tile
	tileCols := kt.cols
	K, N := l.In, l.Out
	Tp := roundUp(T, int8BlockRows)
	xs := ws.xs[:Tp]
	a := ws.vy[:Tp*K]
	src := x
	if Tp != T {
		src = ws.xpad[:Tp*K]
		copy(src, x[:T*K])
		clear(src[T*K:])
	}
	parallelFor(ws.pool, Tp/int8BlockRows, 1, func(glo, ghi int) {
		for g := glo; g < ghi; g++ {
			kt.quantA(&src[g*int8BlockRows*K], K, K, &a[g*int8BlockRows*K], &xs[g*int8BlockRows])
		}
	})
	dst := out
	if Tp != T {
		dst = ws.outpad[:Tp*N]
	}
	parallelFor(ws.pool, N/tileCols, 1, func(blo, bhi int) {
		for g := 0; g < Tp/int8BlockRows; g++ {
			kt.gemm(&a[g*int8BlockRows*K], &l.Packed[blo*tileCols*K], K, bhi-blo, tileCols*K,
				&xs[g*int8BlockRows], &l.Scale[blo*tileCols], &l.Corr[blo*tileCols], &dst[g*int8BlockRows*N+blo*tileCols], N)
		}
	})
	if Tp != T {
		copy(out[:T*N], dst[:T*N])
	}
}
