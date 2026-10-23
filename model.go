// Package modernbert runs the ModernBERT encoder (sbintuitions/modernbert-ja
// and the like) from a GGUF file in pure Go, for one purpose: turning a token
// sequence into the final hidden state (the CLS vector of an embedding or a
// classifier). It is a fixed forward pass over a fixed architecture, not a
// general graph engine.
package modernbert

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Config is the architecture read from the GGUF metadata.
type Config struct {
	NEmbd      int     // hidden size
	NLayer     int     // encoder layers
	NHead      int     // attention heads (head dim = NEmbd / NHead)
	NFF        int     // feed-forward width (the up projection is 2*NFF for GeGLU)
	NVocab     int     // token embedding rows
	NCtxTrain  int     // training context length
	NormEps    float32 // LayerNorm epsilon
	RopeBase   float32 // RoPE theta of the global layers
	RopeBaseSW float32 // RoPE theta of the sliding-window layers
	Window     int     // sliding window: a key within ±Window/2 of the query is visible
	Pattern    int     // every Pattern-th layer (from 0) is global, the rest sliding-window
}

// HeadDim is the per-head width.
func (c Config) HeadDim() int { return c.NEmbd / c.NHead }

// IsGlobal reports whether layer il attends over the whole sequence.
func (c Config) IsGlobal(il int) bool { return c.Pattern == 0 || il%c.Pattern == 0 }

// Linear is a weight matrix of Out rows of In values (y[n] = Σ_k W[n*In+k]
// x[k]), as float32 (W) and, on the int8 path, as per-row symmetric int8
// (W8 with one Scale per row).
type Linear struct {
	Out, In int
	W       []float32
	W8      []int8
	Scale   []float32
	// RowSum is Σ W8 per row, for kernels that take unsigned activations.
	RowSum []int32
	// Packed is the per-row int8 W8 in the tile's block layout (tile.go),
	// with Corr the per-row offset correction.
	Packed []byte
	Corr   []int32
	// ActF32 keeps the activations in float32 against int8 weights
	// (QuantInt8RowWeights).
	ActF32 bool
}

// Quant selects how the linear layers compute.
type Quant int

const (
	// QuantDefault is QuantInt8Row.
	QuantDefault Quant = iota
	// QuantInt8Row runs every linear layer in int8: weights one scale per
	// output row, activations one scale per token. With the int8 tile
	// (tile.go) attention runs in int8 too; a CPU without one runs the
	// linear layers on the dot-product tile of kernel.go, else in Go, and
	// attention in float32.
	QuantInt8Row
	// QuantF32 keeps the linear layers in float32 (the reference path).
	QuantF32
	// QuantInt8RowWeights quantizes the weights per row but keeps the
	// activations in float32 (a diagnostic: it isolates the weight side of
	// QuantInt8Row's error).
	QuantInt8RowWeights
)

// Options tune Load.
type Options struct {
	Quant Quant
	// Threads is how many goroutines a forward pass spreads its matrix
	// products and attention over; 0 or 1 runs single-threaded, and more
	// than GOMAXPROCS is cut to GOMAXPROCS.
	Threads int
	// Kernel selects the SIMD kernels; KernelAuto picks the best the CPU
	// advertises. A kernel named explicitly runs whether or not the CPU
	// advertises it.
	Kernel Kernel
}

// Layer holds one encoder block's weights. AttnNorm is nil on layer 0 (the
// embedding norm feeds it directly).
type Layer struct {
	AttnNorm []float32 // [NEmbd] or nil
	QKV      Linear    // 3*NEmbd × NEmbd
	Out      Linear    // NEmbd × NEmbd
	FFNNorm  []float32 // [NEmbd]
	Up       Linear    // 2*NFF × NEmbd: first NFF rows are activated, the rest gate
	Down     Linear    // NEmbd × NFF
}

// Model is a loaded encoder.
type Model struct {
	Config
	TokEmbd    []float32 // [NVocab*NEmbd]
	TokNorm    []float32 // [NEmbd]
	Layers     []Layer
	OutputNorm []float32 // [NEmbd]
	// Metadata keeps every GGUF key/value, for callers that ship their own
	// data in the file (a classifier head, say).
	Metadata map[string]any

	threads int
	// k is the kernel set the model runs on.
	k *kernels
	// int8Attention runs attention on the int8 tile (QuantInt8Row with a
	// tile present).
	int8Attention bool
	// wsFree keeps forward-pass workspaces (buffers and worker pools) for
	// reuse across calls.
	wsFree chan *workspace
	wsOnce sync.Once
	// tok is the tokenizer built from the metadata on first use.
	tok     *Tokenizer
	tokErr  error
	tokOnce sync.Once
}

// Load reads a ModernBERT GGUF (f32, f16 or q8_0 tensors) with the default
// options (QuantDefault, single-threaded).
func Load(path string) (*Model, error) { return LoadWithOptions(path, Options{}) }

// LoadWithOptions reads a ModernBERT GGUF with opts.
func LoadWithOptions(path string, opts Options) (*Model, error) {
	f, err := readGGUF(path)
	if err != nil {
		return nil, err
	}
	arch, err := f.stringKV("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "modern-bert" {
		return nil, fmt.Errorf("modernbert: %s is a %q model", path, arch)
	}
	var c Config
	if err := readConfig(f, &c); err != nil {
		return nil, err
	}
	m := &Model{Config: c, Metadata: f.kv}
	if m.TokEmbd, err = loadVector(f, "token_embd.weight", c.NEmbd, c.NVocab); err != nil {
		return nil, err
	}
	if m.TokNorm, err = loadVector(f, "token_embd_norm.weight", c.NEmbd, 1); err != nil {
		return nil, err
	}
	if m.OutputNorm, err = loadVector(f, "output_norm.weight", c.NEmbd, 1); err != nil {
		return nil, err
	}
	m.Layers = make([]Layer, c.NLayer)
	for il := range m.Layers {
		l := &m.Layers[il]
		pfx := fmt.Sprintf("blk.%d.", il)
		if il != 0 {
			if l.AttnNorm, err = loadVector(f, pfx+"attn_norm.weight", c.NEmbd, 1); err != nil {
				return nil, err
			}
		}
		if l.QKV, err = loadLinear(f, pfx+"attn_qkv.weight", c.NEmbd, 3*c.NEmbd); err != nil {
			return nil, err
		}
		if l.Out, err = loadLinear(f, pfx+"attn_output.weight", c.NEmbd, c.NEmbd); err != nil {
			return nil, err
		}
		if l.FFNNorm, err = loadVector(f, pfx+"ffn_norm.weight", c.NEmbd, 1); err != nil {
			return nil, err
		}
		if l.Up, err = loadLinear(f, pfx+"ffn_up.weight", c.NEmbd, 2*c.NFF); err != nil {
			return nil, err
		}
		if l.Down, err = loadLinear(f, pfx+"ffn_down.weight", c.NFF, c.NEmbd); err != nil {
			return nil, err
		}
	}
	if err := m.prepare(opts); err != nil {
		return nil, err
	}
	return m, nil
}

// Kernel reports the kernel set the model runs on (the resolved choice
// when Options.Kernel was KernelAuto).
func (m *Model) Kernel() Kernel { return m.k.kernel }

// prepare selects the kernels and quantizes the float32 weights of m for
// opts.
func (m *Model) prepare(opts Options) error {
	k, err := selectKernels(opts.Kernel)
	if err != nil {
		return err
	}
	m.k = k
	// More threads than the runtime can run at once only contend: the
	// workers spin between regions.
	m.threads = max(1, min(opts.Threads, runtime.GOMAXPROCS(0)))
	if opts.Quant == QuantDefault {
		opts.Quant = QuantInt8Row
	}
	hd := m.HeadDim()
	m.int8Attention = opts.Quant == QuantInt8Row && k.tile != nil && hd%k.tile.cols == 0 && hd%tileRows == 0
	if opts.Quant != QuantInt8Row && opts.Quant != QuantInt8RowWeights {
		return nil
	}
	for il := range m.Layers {
		l := &m.Layers[il]
		for _, lin := range []*Linear{&l.QKV, &l.Out, &l.Up, &l.Down} {
			quantizeLinear(lin)
			lin.W = nil
			lin.ActF32 = opts.Quant == QuantInt8RowWeights
			if opts.Quant == QuantInt8Row && k.tile != nil && lin.In%8 == 0 && lin.Out%k.tile.cols == 0 {
				lin.Packed = make([]byte, lin.Out*lin.In)
				lin.Corr = make([]int32, lin.Out)
				k.tile.packB(lin.W8, lin.Out, lin.In, k.tile.actOffset, lin.Packed, lin.Corr)
			}
		}
	}
	return nil
}

func readConfig(f *ggufFile, c *Config) error {
	u := func(key string, dst *int) error {
		v, err := f.uint32KV("modern-bert." + key)
		if err != nil {
			return err
		}
		*dst = int(v)
		return nil
	}
	for key, dst := range map[string]*int{
		"embedding_length":     &c.NEmbd,
		"block_count":          &c.NLayer,
		"attention.head_count": &c.NHead,
		"feed_forward_length":  &c.NFF,
		"vocab_size":           &c.NVocab,
		"context_length":       &c.NCtxTrain,
	} {
		if err := u(key, dst); err != nil {
			return err
		}
	}
	var err error
	if c.NormEps, err = f.float32KV("modern-bert.attention.layer_norm_epsilon"); err != nil {
		return err
	}
	if c.RopeBase, err = f.float32KV("modern-bert.rope.freq_base"); err != nil {
		return err
	}
	// Sliding-window attention is optional in the format; without it every
	// layer is global.
	if w, werr := f.uint32KV("modern-bert.attention.sliding_window"); werr == nil && w > 0 {
		c.Window = int(w)
		if c.RopeBaseSW, err = f.float32KV("modern-bert.rope.freq_base_swa"); err != nil {
			c.RopeBaseSW = c.RopeBase
		}
		p, perr := f.uint32KV("modern-bert.attention.sliding_window_pattern")
		if perr != nil {
			p = 3
		}
		c.Pattern = int(p)
	}
	for name, v := range map[string]int{"hidden size": c.NEmbd, "head count": c.NHead, "feed-forward width": c.NFF, "vocabulary": c.NVocab} {
		if v <= 0 {
			return fmt.Errorf("modernbert: %s is %d", name, v)
		}
	}
	// Each layer has at least four tensors; more layers than that cannot
	// be in the file.
	if c.NLayer < 0 || c.NLayer > len(f.tensors)/4 {
		return fmt.Errorf("modernbert: %d layers declared but the file holds %d tensors", c.NLayer, len(f.tensors))
	}
	if c.NEmbd%c.NHead != 0 {
		return fmt.Errorf("modernbert: hidden size %d is not a multiple of %d heads", c.NEmbd, c.NHead)
	}
	if c.HeadDim()%2 != 0 {
		return fmt.Errorf("modernbert: head dim %d is odd", c.HeadDim())
	}
	return nil
}

// loadVector loads a tensor of shape [in, rows] as float32 and checks it.
func loadVector(f *ggufFile, name string, in, rows int) ([]float32, error) {
	v, t, err := f.f32(name)
	if err != nil {
		return nil, err
	}
	if len(v) != in*rows || int(t.shape[0]) != in {
		return nil, fmt.Errorf("modernbert: tensor %s has shape %v, want [%d %d]", name, t.shape, in, rows)
	}
	return v, nil
}

func loadLinear(f *ggufFile, name string, in, out int) (Linear, error) {
	w, err := loadVector(f, name, in, out)
	if err != nil {
		return Linear{}, err
	}
	return Linear{Out: out, In: in, W: w}, nil
}

// ropeBase returns the RoPE theta of layer il.
func (m *Model) ropeBase(il int) float32 {
	if m.IsGlobal(il) {
		return m.RopeBase
	}
	return m.RopeBaseSW
}

// invFreq returns the per-pair inverse frequencies of RoPE for base theta.
func invFreq(headDim int, base float32) []float32 {
	half := headDim / 2
	out := make([]float32, half)
	for i := range out {
		out[i] = float32(math.Pow(float64(base), -2*float64(i)/float64(headDim)))
	}
	return out
}
