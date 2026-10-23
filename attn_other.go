//go:build !arm64

package modernbert

import "unsafe"

const hasF32Tiles = false

func dotF32Tile4x4(a, b *float32, k, aStride, bStride int, out *float32) {
	as := unsafe.Slice(a, 3*aStride+k)
	bs := unsafe.Slice(b, 3*bStride+k)
	o := unsafe.Slice(out, 16)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var s float32
			for c := 0; c < k; c++ {
				s += as[i*aStride+c] * bs[j*bStride+c]
			}
			o[i*4+j] = s
		}
	}
}

func axpyTile4x16(p *float32, pStride int, v *float32, vStride int, n int, out *float32, outStride int) {
	ps := unsafe.Slice(p, 3*pStride+n)
	vs := unsafe.Slice(v, (n-1)*vStride+16)
	o := unsafe.Slice(out, 3*outStride+16)
	for i := 0; i < 4; i++ {
		var acc [16]float32
		for j := 0; j < n; j++ {
			w := ps[i*pStride+j]
			row := vs[j*vStride : j*vStride+16]
			for d := 0; d < 16; d++ {
				acc[d] += w * row[d]
			}
		}
		copy(o[i*outStride:i*outStride+16], acc[:])
	}
}
