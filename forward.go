package modernbert

import (
	"fmt"
	"math"
	"runtime"
)

// workspace holds the buffers of one forward pass, sized for its sequence,
// so the layers reuse them instead of allocating per op, and the pool its
// parallel regions run on.
type workspace struct {
	cap                                       int // rows the buffers hold
	pool                                      *pool
	h, x, qkv, q, k, v, att, o, up, act, down []float32
	xq, xu                                    []int8
	xs                                        []float32
	// q8_0x4 path: activations padded to four rows, their quantized
	// blocks, and the padded output.
	xpad, outpad []float32
	vy           []byte
	// ropeCS[g] is the cos/sin table of RoPE base g (0 global, 1 sliding
	// window): [T * HeadDim] as (cos, sin) pairs per frequency.
	ropeCS [2][]float32
	// Int8 attention: packed Q, K (per head, blocks of eight tokens) and
	// V (per head, blocks of eight dimensions over the keys) with their
	// scales, and the float32 transpose of V.
	qp, kp, vp     []byte
	qs, ks, vs, vt []float32
	kcorr, vcorr   []int32
	// attn[w] is thread w's scratch for one block of queries.
	attn []attnScratch
}

type attnScratch struct {
	scores, otile []float32
	probs         []uint8
	tmp           []int8
}

func (m *Model) newWorkspace(T int) *workspace {
	D, F := m.NEmbd, m.NFF
	kMax := max(D, F)
	Tp := roundUp(T, tileRows)
	if m.k.tile != nil {
		Tp = roundUp(T, max(tileRows, m.k.tile.align))
	}
	ws := &workspace{
		cap:  T,
		pool: newPool(m.threads),
		h:    make([]float32, T*D), x: make([]float32, T*D),
		qkv: make([]float32, T*3*D), q: make([]float32, Tp*D), k: make([]float32, Tp*D), v: make([]float32, Tp*D),
		att: make([]float32, T*D), o: make([]float32, T*D),
		up: make([]float32, T*2*F), act: make([]float32, T*F), down: make([]float32, T*D),
	}
	if m.int8Attention {
		hd := m.HeadDim()
		ws.qp, ws.kp, ws.vp = make([]byte, Tp*D), make([]byte, Tp*D), make([]byte, Tp*D)
		ws.qs, ws.ks = make([]float32, m.NHead*Tp), make([]float32, m.NHead*Tp)
		ws.kcorr, ws.vcorr = make([]int32, m.NHead*Tp), make([]int32, D)
		ws.vs, ws.vt = make([]float32, D), make([]float32, hd*Tp*m.NHead)
		ws.attn = make([]attnScratch, ws.pool.workers+1)
		for w := range ws.attn {
			ws.attn[w] = attnScratch{
				scores: make([]float32, tileRows*Tp),
				probs:  make([]uint8, tileRows*Tp),
				otile:  make([]float32, tileRows*hd),
				tmp:    make([]int8, m.k.tile.cols*max(hd, Tp)),
			}
		}
	}
	ws.xq, ws.xu, ws.xs = make([]int8, Tp*kMax), make([]int8, Tp*kMax), make([]float32, Tp)
	ws.xpad = make([]float32, Tp*kMax)
	ws.outpad = make([]float32, Tp*max(3*D, 2*F))
	ws.vy = make([]byte, Tp*kMax)
	hd := m.HeadDim()
	half := hd / 2
	for g, base := range []float32{m.RopeBase, m.RopeBaseSW} {
		inv := invFreq(hd, base)
		cs := make([]float32, T*hd)
		for t := 0; t < T; t++ {
			for i := 0; i < half; i++ {
				theta := float64(t) * float64(inv[i])
				cs[t*hd+2*i], cs[t*hd+2*i+1] = float32(math.Cos(theta)), float32(math.Sin(theta))
			}
		}
		ws.ropeCS[g] = cs
	}
	runtime.SetFinalizer(ws, func(ws *workspace) { ws.pool.close() })
	return ws
}

// acquireWorkspace returns a workspace holding at least T rows: a free
// one when it is large enough, else a new one sized to the next multiple
// of 64 rows. The caller returns it with releaseWorkspace.
func (m *Model) acquireWorkspace(T int) *workspace {
	m.wsOnce.Do(func() { m.wsFree = make(chan *workspace, 16) })
	select {
	case ws := <-m.wsFree:
		if ws.cap >= T {
			return ws
		}
		// Too small: it gets collected, its workers closed by the finalizer.
	default:
	}
	return m.newWorkspace((T + 63) / 64 * 64)
}

// releaseWorkspace keeps ws for reuse, or drops it when the free list is
// full.
func (m *Model) releaseWorkspace(ws *workspace) {
	select {
	case m.wsFree <- ws:
	default:
	}
}

// Close releases the workspaces and worker goroutines kept for reuse.
// The model stays usable; a later Encode allocates afresh. Without Close
// they are released when the model is garbage collected.
func (m *Model) Close() {
	m.wsOnce.Do(func() { m.wsFree = make(chan *workspace, 16) })
	for {
		select {
		case ws := <-m.wsFree:
			runtime.SetFinalizer(ws, nil)
			ws.pool.close()
		default:
			return
		}
	}
}

// Encode runs the encoder over tokens (ids including BOS/EOS as the tokenizer
// emits them) and returns the final hidden state of every position,
// [len(tokens)*NEmbd]. CLS pooling is the first NEmbd values. Sequences
// longer than the model's context length are rejected. Encode is safe
// for concurrent use.
func (m *Model) Encode(tokens []int32) ([]float32, error) {
	T := len(tokens)
	if T == 0 {
		return nil, fmt.Errorf("modernbert: empty input")
	}
	if m.NCtxTrain > 0 && T > m.NCtxTrain {
		return nil, fmt.Errorf("modernbert: %d tokens exceed the context length %d", T, m.NCtxTrain)
	}
	D := m.NEmbd
	for i, t := range tokens {
		if t < 0 || int(t) >= m.NVocab {
			return nil, fmt.Errorf("modernbert: token #%d = %d outside the vocabulary (%d)", i, t, m.NVocab)
		}
	}
	ws := m.acquireWorkspace(T)
	defer m.releaseWorkspace(ws)
	h, x := ws.h[:T*D], ws.x[:T*D]
	// Embedding lookup + embedding norm.
	for i, t := range tokens {
		copy(h[i*D:(i+1)*D], m.TokEmbd[int(t)*D:(int(t)+1)*D])
	}
	m.layerNorm(h, T, m.TokNorm, ws)

	for il := range m.Layers {
		l := &m.Layers[il]
		// Attention block: norm (identity on layer 0), QKV, RoPE, attention, out, residual.
		copy(x, h)
		if l.AttnNorm != nil {
			m.layerNorm(x, T, l.AttnNorm, ws)
		}
		m.matmul(x, T, l.QKV, ws, ws.qkv)
		cs := ws.ropeCS[1]
		if m.IsGlobal(il) {
			cs = ws.ropeCS[0]
		}
		parallelFor(ws.pool, T, 1, func(tlo, thi int) {
			for t := tlo; t < thi; t++ {
				copy(ws.q[t*D:(t+1)*D], ws.qkv[t*3*D:t*3*D+D])
				copy(ws.k[t*D:(t+1)*D], ws.qkv[t*3*D+D:t*3*D+2*D])
				copy(ws.v[t*D:(t+1)*D], ws.qkv[t*3*D+2*D:t*3*D+3*D])
				m.ropeRow(ws.q[t*D:(t+1)*D], cs[t*m.HeadDim():(t+1)*m.HeadDim()])
				m.ropeRow(ws.k[t*D:(t+1)*D], cs[t*m.HeadDim():(t+1)*m.HeadDim()])
			}
		})
		if m.int8Attention {
			m.attentionInt8(ws.q, ws.k, ws.v, T, il, ws.att, ws)
		} else {
			m.attention(ws.q, ws.k, ws.v, T, il, ws.att, ws)
		}
		m.matmul(ws.att, T, l.Out, ws, ws.o)

		// FFN block: residual, norm, up (GeGLU), down, residual.
		parallelFor(ws.pool, T, 1, func(tlo, thi int) {
			add(h[tlo*D:thi*D], ws.o[tlo*D:thi*D])
			copy(x[tlo*D:thi*D], h[tlo*D:thi*D])
			m.k.layerNormRows(x[tlo*D:thi*D], thi-tlo, D, l.FFNNorm, m.NormEps)
		})
		m.matmul(x, T, l.Up, ws, ws.up) // [T*2*NFF]
		parallelFor(ws.pool, T, 1, func(tlo, thi int) {
			for t := tlo; t < thi; t++ {
				row := ws.up[t*2*m.NFF : (t+1)*2*m.NFF]
				a := ws.act[t*m.NFF : (t+1)*m.NFF]
				gate := row[m.NFF:]
				m.k.geglu(a, row, gate, m.NFF)
			}
		})
		m.matmul(ws.act, T, l.Down, ws, ws.down)
		parallelFor(ws.pool, T, 1, func(tlo, thi int) {
			add(h[tlo*D:thi*D], ws.down[tlo*D:thi*D])
		})
	}
	m.layerNorm(h, T, m.OutputNorm, ws)
	out := make([]float32, T*D)
	copy(out, h)
	return out, nil
}

// layerNorm normalizes the T rows of x in place, in parallel.
func (m *Model) layerNorm(x []float32, T int, w []float32, ws *workspace) {
	D := m.NEmbd
	parallelFor(ws.pool, T, 1, func(tlo, thi int) {
		m.k.layerNormRows(x[tlo*D:thi*D], thi-tlo, D, w, m.NormEps)
	})
}

// matmul computes x (T rows of l.In) times l into out (T rows of l.Out).
func (m *Model) matmul(x []float32, T int, l Linear, ws *workspace, out []float32) {
	if l.Packed != nil {
		m.matmulTile(x, T, l, ws, out)
		return
	}
	if l.W8 != nil {
		m.matmulInt8(x, T, l, ws, out)
		return
	}
	parallelFor(ws.pool, l.Out, 1, func(nlo, nhi int) {
		for t := 0; t < T; t++ {
			xr := x[t*l.In : (t+1)*l.In]
			for n := nlo; n < nhi; n++ {
				w := l.W[n*l.In : (n+1)*l.In]
				var s float32
				for k, xv := range xr {
					s += w[k] * xv
				}
				out[t*l.Out+n] = s
			}
		}
	})
}

// ropeRow applies rotary position embedding in place to one token's row
// (heads side by side) with its (cos, sin) table, NEOX style: the pair for
// frequency i is (i, i+HeadDim/2).
func (m *Model) ropeRow(row, cs []float32) {
	hd := m.HeadDim()
	half := hd / 2
	for hh := 0; hh < m.NHead; hh++ {
		v := row[hh*hd : (hh+1)*hd]
		for i := 0; i < half; i++ {
			c, s := cs[2*i], cs[2*i+1]
			x0, x1 := v[i], v[i+half]
			v[i] = x0*c - x1*s
			v[i+half] = x0*s + x1*c
		}
	}
}

// attention computes softmax(q·kᵀ/√d) v per head into out ([T*NEmbd]),
// with the symmetric sliding window on local layers. Queries are processed
// in blocks of four: the block's scores against the union of its windows
// come from the 4×4 dot tile, each row is masked to its own window and
// normalized, and the weighted sum of values comes from the 4×16 axpy
// tile. A tail block of fewer than four queries runs in scalar Go.
func (m *Model) attention(q, k, v []float32, T, il int, out []float32, ws *workspace) {
	hd := m.HeadDim()
	D := m.NEmbd
	scale := float32(1 / math.Sqrt(float64(hd)))
	halfWin := -1
	if !m.IsGlobal(il) && m.Window > 0 {
		halfWin = m.Window / 2
	}
	tiles := hasF32Tiles && hd%16 == 0 && m.k.kernel != KernelNone
	nBlocks := (T + 3) / 4
	parallelFor(ws.pool, m.NHead*nBlocks, 1, func(lo, hi int) {
		scores := make([]float32, 4*T)
		var tile [16]float32
		for item := lo; item < hi; item++ {
			hh, i0 := item/nBlocks, item%nBlocks*4
			rows := min(4, T-i0)
			off := hh * hd
			// The union of the block's key windows.
			jlo, jhi := 0, T
			if halfWin >= 0 {
				jlo = max(0, i0-halfWin)
				jhi = min(T, i0+rows-1+halfWin+1)
			}
			n := jhi - jlo
			// Scores S[r*n + (j-jlo)] = q_{i0+r} · k_j.
			j := jlo
			if rows == 4 && tiles {
				for ; j+4 <= jhi; j += 4 {
					dotF32Tile4x4(&q[i0*D+off], &k[j*D+off], hd, D, D, &tile[0])
					for r := 0; r < 4; r++ {
						copy(scores[r*n+j-jlo:r*n+j-jlo+4], tile[r*4:r*4+4])
					}
				}
			}
			for ; j < jhi; j++ {
				kj := k[j*D+off : j*D+off+hd]
				for r := 0; r < rows; r++ {
					qi := q[(i0+r)*D+off : (i0+r)*D+off+hd]
					var s float32
					for d := range kj {
						s += qi[d] * kj[d]
					}
					scores[r*n+j-jlo] = s
				}
			}
			// Per row: mask to the row's own window, softmax.
			for r := 0; r < rows; r++ {
				i := i0 + r
				row := scores[r*n : (r+1)*n]
				rlo, rhi := 0, n
				if halfWin >= 0 {
					rlo = max(0, i-halfWin) - jlo
					rhi = min(T, i+halfWin+1) - jlo
				}
				maxS := float32(math.Inf(-1))
				for c := rlo; c < rhi; c++ {
					row[c] *= scale
					if row[c] > maxS {
						maxS = row[c]
					}
				}
				var sum float32
				if rhi > rlo {
					sum = m.k.softmaxExpRow(row[rlo:rhi], maxS)
				}
				inv := 1 / sum
				for c := 0; c < rlo; c++ {
					row[c] = 0
				}
				for c := rlo; c < rhi; c++ {
					row[c] *= inv
				}
				for c := rhi; c < n; c++ {
					row[c] = 0
				}
			}
			// Weighted sum of values.
			if rows == 4 && tiles {
				for c := 0; c < hd; c += 16 {
					axpyTile4x16(&scores[0], n, &v[jlo*D+off+c], D, n, &out[i0*D+off+c], D)
				}
				continue
			}
			for r := 0; r < rows; r++ {
				oi := out[(i0+r)*D+off : (i0+r)*D+off+hd]
				clear(oi)
				row := scores[r*n : (r+1)*n]
				for c, p := range row {
					if p == 0 {
						continue
					}
					vj := v[(jlo+c)*D+off : (jlo+c)*D+off+hd]
					for d := range oi {
						oi[d] += p * vj[d]
					}
				}
			}
		}
	})
}

// layerNormRows normalizes each of the T rows of x (width D) in place with a
// weight and no bias: y = (x - mean) / sqrt(var + eps) * w.
func layerNormRows(x []float32, T, D int, w []float32, eps float32) {
	for t := 0; t < T; t++ {
		row := x[t*D : (t+1)*D]
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(D)
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(D)
		inv := float32(1 / math.Sqrt(float64(variance+eps)))
		for i, v := range row {
			row[i] = (v - mean) * inv * w[i]
		}
	}
}

func add(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}
