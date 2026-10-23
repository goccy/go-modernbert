package modernbert

import (
	"math"
)

// Int8 attention: per head, Q and K rows quantized per token and V per
// dimension over the keys, so that both the scores and the weighted value
// sum run on the int8 tile. The softmax stays in float32; its
// unnormalized weights go to the value product as unsigned bytes with
// 1/(probLevels·sum) folded into the row scale.

// attentionInt8 is attention on the int8 tile: q, k, v are [Tp*NEmbd]
// with zero rows past T (Tp = T rounded up to tileAlign), out is
// [T*NEmbd].
func (m *Model) attentionInt8(q, k, v []float32, T, il int, out []float32, ws *workspace) {
	hd, D := m.HeadDim(), m.NEmbd
	kt := m.k.tile
	tileCols := kt.cols
	Tp := roundUp(T, kt.align)
	nb := Tp / tileRows  // query blocks
	nkb := Tp / tileCols // key blocks
	db := hd / tileCols  // value dimension blocks
	halfWin := -1
	if !m.IsGlobal(il) && m.Window > 0 {
		halfWin = m.Window / 2
	}
	scale := float32(1 / math.Sqrt(float64(hd)))

	// Q (blocks of eight tokens, the score scale folded into the scales)
	// and K (blocks of tileCols tokens, with the offset correction), one
	// key block per unit.
	parallelForWorker(ws.pool, m.NHead*nkb, 1, func(w, lo, hi int) {
		for item := lo; item < hi; item++ {
			hh, g := item/nkb, item%nkb
			off := hh * hd
			t0 := g * tileCols
			for i0 := t0; i0 < t0+tileCols; i0 += tileRows {
				kt.quantA(&q[i0*D+off], hd, D, &ws.qp[hh*Tp*hd+i0*hd], &ws.qs[hh*Tp+i0])
			}
			for r := t0; r < t0+tileCols; r++ {
				ws.qs[hh*Tp+r] *= scale
			}
			kt.quantB(k[t0*D+off:], tileCols, hd, D, kt.actOffset, 127, ws.kp[hh*Tp*hd+t0*hd:], ws.kcorr[hh*Tp+t0:], ws.ks[hh*Tp+t0:], ws.attn[w].tmp)
		}
	})
	// V: transposed per head so a block holds tileCols dimensions over all
	// keys, quantized per dimension; the weights it meets are unsigned
	// with no offset, so its correction is zero.
	parallelForWorker(ws.pool, m.NHead*db, 1, func(w, lo, hi int) {
		for item := lo; item < hi; item++ {
			hh, b := item/db, item%db
			off := hh*hd + b*tileCols
			vt := ws.vt[off*Tp : (off+tileCols)*Tp]
			for t := 0; t < Tp; t++ {
				row := v[t*D+off : t*D+off+tileCols]
				for d, x := range row {
					vt[d*Tp+t] = x
				}
			}
			kt.quantB(vt, tileCols, Tp, Tp, 0, kt.valueLevels, ws.vp[off*Tp:], ws.vcorr[off:], ws.vs[off:], ws.attn[w].tmp)
		}
	})

	parallelForWorker(ws.pool, m.NHead*nb, 1, func(w, lo, hi int) {
		sc := &ws.attn[w]
		scores := sc.scores[:tileRows*Tp]
		probs := sc.probs[:tileRows*Tp]
		otile := sc.otile[:tileRows*hd]
		var ps [tileRows]float32
		for item := lo; item < hi; item++ {
			hh, g := item/nb, item%nb
			i0 := g * tileRows
			off := hh * hd
			// Key blocks covering the union of the block's windows.
			jb0, jb1 := 0, nkb
			if halfWin >= 0 {
				jb0 = max(0, (i0-halfWin)/tileCols)
				jb1 = min(nkb, (i0+tileRows+halfWin+tileCols-1)/tileCols)
			}
			jlo := jb0 * tileCols
			kw := (jb1 - jb0) * tileCols
			kt.gemm(&ws.qp[hh*Tp*hd+i0*hd], &ws.kp[hh*Tp*hd+jlo*hd], hd, jb1-jb0, tileCols*hd,
				&ws.qs[hh*Tp+i0], &ws.ks[hh*Tp+jlo], &ws.kcorr[hh*Tp+jlo], &scores[0], kw)
			for r := 0; r < tileRows; r++ {
				i := i0 + r
				row := scores[r*kw : (r+1)*kw]
				if i >= T {
					clear(row)
					ps[r] = 0
					continue
				}
				rlo, rhi := 0, min(T, jlo+kw)-jlo
				if halfWin >= 0 {
					rlo = max(0, i-halfWin) - jlo
					rhi = min(T, i+halfWin+1) - jlo
				}
				sum := m.k.softmaxExp(row[rlo:rhi])
				clear(row[:rlo])
				clear(row[rhi:])
				ps[r] = 1 / (kt.probLevels * sum)
			}
			kt.quantProbs(&scores[0], kw, kw, &probs[0])
			dst, stride := &otile[0], hd
			if i0+tileRows <= T {
				dst, stride = &out[i0*D+off], D
			}
			kt.gemmProbs(&probs[0], &ws.vp[off*Tp+jlo*tileCols], kw, db, Tp*tileCols,
				&ps[0], &ws.vs[off], &ws.vcorr[off], dst, stride)
			if i0+tileRows > T {
				for r := 0; i0+r < T; r++ {
					copy(out[(i0+r)*D+off:(i0+r)*D+off+hd], otile[r*hd:(r+1)*hd])
				}
			}
		}
	})
}
