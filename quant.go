package modernbert

import "math"

// Int8 path: weights with one scale per output row, activations with one
// scale per token row, int32 dot products. With scales outside the inner
// loop the product reduces to integer multiply-accumulates over the whole
// K, which is what int8 dot-product instructions (SDOT / SMMLA, VPDPBUSD)
// run at full rate.

// quantizeRowsInt8 quantizes rows of width in to symmetric int8 with one
// scale per row: q = round(x / s), s = max|x| / levels.|x| / 127. Branch-free on the
// data (sign-dependent branches mispredict on half of random activations):
// |x| and the max are taken on the float bit patterns, and rounding adds
// ±0.5 built from the sign bit before truncation.
func quantizeRowsInt8(x []float32, rows, in int, q []int8, scales []float32, levels float32) {
	for r := 0; r < rows; r++ {
		row := x[r*in : (r+1)*in]
		var amaxBits uint32
		for _, v := range row {
			b := math.Float32bits(v) & 0x7fffffff
			if b > amaxBits {
				amaxBits = b
			}
		}
		s := math.Float32frombits(amaxBits) / levels
		inv := float32(0)
		if s > 0 {
			inv = 1 / s
		}
		scales[r] = s
		qr := q[r*in : (r+1)*in]
		for i, v := range row {
			f := v * inv
			half := math.Float32frombits(math.Float32bits(f)&0x80000000 | 0x3f000000) // ±0.5
			qr[i] = int8(int32(f + half))
		}
	}
}

// quantizeLinear derives the int8 form of l from its float32 weights.
func quantizeLinear(l *Linear) {
	l.W8 = make([]int8, l.Out*l.In)
	l.Scale = make([]float32, l.Out)
	quantizeRowsInt8(l.W, l.Out, l.In, l.W8, l.Scale, 127)
	l.RowSum = make([]int32, l.Out)
	for n := 0; n < l.Out; n++ {
		var s int32
		for _, v := range l.W8[n*l.In : (n+1)*l.In] {
			s += int32(v)
		}
		l.RowSum[n] = s
	}
}

// matmulInt8 is x (T rows of l.In, float32) times l on the int8 path:
// activations quantized per row into ws, int32 dot per (row, output),
// scaled back into out (T rows of l.Out).
func matmulInt8(x []float32, T int, l Linear, ws *workspace, out []float32) {
	if l.ActF32 {
		parallelFor(ws.pool, l.Out, 1, func(nlo, nhi int) {
			for t := 0; t < T; t++ {
				xr := x[t*l.In : (t+1)*l.In]
				for n := nlo; n < nhi; n++ {
					w := l.W8[n*l.In : (n+1)*l.In]
					var s float32
					for k, xv := range xr {
						s += float32(w[k]) * xv
					}
					out[t*l.Out+n] = s * l.Scale[n]
				}
			}
		})
		return
	}
	xq := ws.xq[:T*l.In]
	xs := ws.xs[:T]
	parallelFor(ws.pool, T, 1, func(tlo, thi int) {
		quantizeRowsInt8(x[tlo*l.In:thi*l.In], thi-tlo, l.In, xq[tlo*l.In:thi*l.In], xs[tlo:thi], 127)
	})
	gemmInt8(xq, T, l.In, l.W8, l.RowSum, l.Out, ws, xs, l.Scale, out)
}

// gemmInt8 computes out[t*N+n] = xs[t] * ws[n] * Σ_c x[t*k+c] * w[n*k+c]
// for every (row t of x, row n of w), the output rows split across
// threads. The CPU's tile kernel covers the bulk when k is a multiple of
// its alignment; the edges and the fallback run the scalar dot. rowSum (Σw
// per weight row) is needed only by kernels that take unsigned activations.
func gemmInt8(x []int8, T, k int, w []int8, rowSum []int32, N int, wsp *workspace, xs, ws []float32, out []float32) {
	kn := kernel
	useTile := kn.rows > 0 && k > 0 && k%kn.kAlign == 0
	xk := x
	if useTile && kn.unsignedX {
		// x + 128 as unsigned bytes, stored in the int8 slice the kernel
		// reinterprets; the dot product then carries 128·Σw per weight row.
		xk = wsp.xu[:len(x)]
		for i, v := range x {
			xk[i] = int8(uint8(v) ^ 0x80)
		}
	}
	align := 1
	if useTile {
		align = kn.cols
	}
	parallelFor(wsp.pool, N, align, func(nlo, nhi int) {
		t0, n0 := 0, nlo
		if useTile {
			var acc [64]int32
			for t := 0; t+kn.rows <= T; t += kn.rows {
				for n := nlo; n+kn.cols <= nhi; n += kn.cols {
					kn.run(&xk[t*k], &w[n*k], k, k, k, &acc[0])
					for i := 0; i < kn.rows; i++ {
						o := out[(t+i)*N+n : (t+i)*N+n+kn.cols]
						sx := xs[t+i]
						for j := range o {
							a := acc[i*kn.cols+j]
							if kn.unsignedX {
								a -= 128 * rowSum[n+j]
							}
							o[j] = float32(a) * sx * ws[n+j]
						}
					}
				}
			}
			t0, n0 = T/kn.rows*kn.rows, nlo+(nhi-nlo)/kn.cols*kn.cols
		}
		// Right edge (all t, n ≥ n0) and bottom edge (t ≥ t0, nlo ≤ n < n0).
		for t := 0; t < T; t++ {
			xr := x[t*k : (t+1)*k]
			start := n0
			if t >= t0 {
				start = nlo
			}
			for n := start; n < nhi; n++ {
				out[t*N+n] = float32(dotInt8(xr, w[n*k:(n+1)*k])) * xs[t] * ws[n]
			}
		}
	})
}

// dotInt8 is the int32 dot product of two int8 vectors of equal length.
func dotInt8(a, b []int8) int32 {
	var s int32
	b = b[:len(a)]
	for i, av := range a {
		s += int32(av) * int32(b[i])
	}
	return s
}
