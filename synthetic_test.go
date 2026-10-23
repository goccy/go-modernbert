package modernbert

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"
)

// syntheticModel builds a model of the given shape from random weights,
// prepared like a loaded one, for benchmarks that need no model file.
func syntheticModel(tb testing.TB, c Config, opts Options) *Model {
	rng := rand.New(rand.NewSource(1))
	vec := func(n int, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * scale
		}
		return v
	}
	ones := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}
	lin := func(in, out int) Linear {
		return Linear{Out: out, In: in, W: vec(out*in, 1/float32(in))}
	}
	m := &Model{Config: c, TokEmbd: vec(c.NVocab*c.NEmbd, 0.1), TokNorm: ones(c.NEmbd), OutputNorm: ones(c.NEmbd), Metadata: map[string]any{}}
	m.Layers = make([]Layer, c.NLayer)
	for il := range m.Layers {
		l := &m.Layers[il]
		if il != 0 {
			l.AttnNorm = ones(c.NEmbd)
		}
		l.QKV, l.Out = lin(c.NEmbd, 3*c.NEmbd), lin(c.NEmbd, c.NEmbd)
		l.FFNNorm = ones(c.NEmbd)
		l.Up, l.Down = lin(c.NEmbd, 2*c.NFF), lin(c.NFF, c.NEmbd)
	}
	if err := m.prepare(opts); err != nil {
		tb.Fatal(err)
	}
	return m
}

// config130m is the shape of modernbert-ja-130m with a small vocabulary.
var config130m = Config{NEmbd: 512, NLayer: 19, NHead: 8, NFF: 2048, NVocab: 4096, NCtxTrain: 8192, NormEps: 1e-5, RopeBase: 160000, RopeBaseSW: 10000, Window: 128, Pattern: 3}

// BenchmarkEncodeSynthetic times the forward pass of a 130m-shaped
// model with random weights for 32 / 128 / 512 tokens, on
// MODERNBERT_THREADS threads (default 1) and the kernel of
// MODERNBERT_KERNEL.
func BenchmarkEncodeSynthetic(b *testing.B) {
	threads, _ := strconv.Atoi(os.Getenv("MODERNBERT_THREADS"))
	m := syntheticModel(b, config130m, Options{Threads: threads, Kernel: testKernel(b)})
	b.Logf("kernel=%s threads=%d", m.k.kernel, m.threads)
	for _, n := range []int{32, 128, 512} {
		toks := make([]int32, n)
		for i := range toks {
			toks[i] = int32(3 + (i*37)%4000)
		}
		b.Run(fmt.Sprintf("tokens=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := m.Encode(toks); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

// TestEncodeSyntheticKernels checks that every kernel set of this CPU
// computes the same forward pass as the Go paths, within quantization
// error, on the synthetic model.
func TestEncodeSyntheticKernels(t *testing.T) {
	c := config130m
	c.NLayer = 4
	ref := syntheticModel(t, c, Options{Kernel: KernelNone, Quant: QuantF32})
	toks := make([]int32, 77)
	for i := range toks {
		toks[i] = int32(3 + (i*37)%4000)
	}
	want, err := ref.Encode(toks)
	if err != nil {
		t.Fatal(err)
	}
	m := syntheticModel(t, c, Options{Kernel: testKernel(t), Threads: 3})
	got, err := m.Encode(toks)
	if err != nil {
		t.Fatal(err)
	}
	var dot, na, nb float64
	for i := range want {
		dot += float64(got[i]) * float64(want[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(want[i]) * float64(want[i])
	}
	cos := dot / math.Sqrt(na*nb)
	t.Logf("kernel=%s cosine %.6f", m.k.kernel, cos)
	if cos < 0.99 {
		t.Fatalf("kernel %s diverges from the float32 path: cosine %.6f", m.k.kernel, cos)
	}
}
