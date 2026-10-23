package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// A ModernBERT GGUF with random weights, in the layout llama.cpp's
// converter produces for the released checkpoints (q8_0 matrices, float32
// norms, a SentencePiece Unigram vocabulary), for benchmarks that cannot
// ship a model file.

type shape struct {
	embd, layers, heads, ff, vocab, ctx int
}

var shape130m = shape{embd: 512, layers: 19, heads: 8, ff: 2048, vocab: 4096, ctx: 8192}

// GGUF value types.
const (
	ggufUint32  = 4
	ggufInt32   = 5
	ggufFloat32 = 6
	ggufBool    = 7
	ggufString  = 8
	ggufArray   = 9

	ggmlF32  = 0
	ggmlQ8_0 = 8

	alignment = 32
)

type tensor struct {
	name string
	dims []uint64 // ne0 first
	typ  uint32
	data []byte
}

type kv struct {
	key string
	typ uint32
	val any // uint32, int32, float32, bool, string, []string, []float32, []int32
}

func writeSyntheticGGUF(path string, s shape, seed int64) error {
	rng := rand.New(rand.NewSource(seed))
	tokens := make([]string, s.vocab)
	scores := make([]float32, s.vocab)
	types := make([]int32, s.vocab)
	for i := range tokens {
		tokens[i] = fmt.Sprintf("tok%d", i)
		scores[i] = -float32(i) / float32(s.vocab)
		types[i] = 1
	}
	for i, sp := range []string{"<unk>", "<s>", "</s>", "<pad>", "<sep>", "<mask>"} {
		tokens[i] = sp
		scores[i] = 0
		types[i] = 3
	}
	types[0] = 2
	kvs := []kv{
		{"general.architecture", ggufString, "modern-bert"},
		{"general.type", ggufString, "model"},
		{"general.name", ggufString, "synthetic"},
		{"general.file_type", ggufUint32, uint32(7)},
		{"general.quantization_version", ggufUint32, uint32(2)},
		{"modern-bert.block_count", ggufUint32, uint32(s.layers)},
		{"modern-bert.context_length", ggufUint32, uint32(s.ctx)},
		{"modern-bert.embedding_length", ggufUint32, uint32(s.embd)},
		{"modern-bert.feed_forward_length", ggufUint32, uint32(s.ff)},
		{"modern-bert.attention.head_count", ggufUint32, uint32(s.heads)},
		{"modern-bert.rope.freq_base", ggufFloat32, float32(160000)},
		{"modern-bert.rope.freq_base_swa", ggufFloat32, float32(10000)},
		{"modern-bert.attention.layer_norm_rms_epsilon", ggufFloat32, float32(1e-5)},
		{"modern-bert.attention.layer_norm_epsilon", ggufFloat32, float32(1e-5)},
		{"modern-bert.attention.causal", ggufBool, false},
		{"modern-bert.classifier.pooling_type", ggufUint32, uint32(2)},
		{"modern-bert.classifier.activation", ggufString, "gelu"},
		{"modern-bert.attention.sliding_window", ggufUint32, uint32(128)},
		{"modern-bert.attention.sliding_window_pattern", ggufUint32, uint32(3)},
		{"modern-bert.rope.scaling.type", ggufString, "none"},
		{"modern-bert.vocab_size", ggufUint32, uint32(s.vocab)},
		{"modern-bert.hidden_activation", ggufString, "gelu"},
		{"tokenizer.ggml.model", ggufString, "t5"},
		{"tokenizer.ggml.pre", ggufString, "default"},
		{"tokenizer.ggml.tokens", ggufArray, tokens},
		{"tokenizer.ggml.scores", ggufArray, scores},
		{"tokenizer.ggml.token_type", ggufArray, types},
		{"tokenizer.ggml.add_space_prefix", ggufBool, false},
		{"tokenizer.ggml.remove_extra_whitespaces", ggufBool, false},
		{"tokenizer.ggml.bos_token_id", ggufUint32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufUint32, uint32(2)},
		{"tokenizer.ggml.unknown_token_id", ggufUint32, uint32(0)},
		{"tokenizer.ggml.seperator_token_id", ggufUint32, uint32(4)},
		{"tokenizer.ggml.padding_token_id", ggufUint32, uint32(3)},
		{"tokenizer.ggml.mask_token_id", ggufUint32, uint32(5)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
		{"tokenizer.ggml.add_eos_token", ggufBool, true},
		{"tokenizer.ggml.add_sep_token", ggufBool, false},
	}
	norm := func(name string) tensor {
		v := make([]float32, s.embd)
		for i := range v {
			v[i] = 1
		}
		return tensor{name: name, dims: []uint64{uint64(s.embd)}, typ: ggmlF32, data: f32Bytes(v)}
	}
	mat := func(name string, in, out int, scale float32) tensor {
		w := make([]float32, out*in)
		for i := range w {
			w[i] = float32(rng.NormFloat64()) * scale
		}
		return tensor{name: name, dims: []uint64{uint64(in), uint64(out)}, typ: ggmlQ8_0, data: q8_0Bytes(w, in)}
	}
	tensors := []tensor{
		norm("token_embd_norm.weight"),
		mat("token_embd.weight", s.embd, s.vocab, 0.1),
		norm("output_norm.weight"),
	}
	for il := 0; il < s.layers; il++ {
		p := fmt.Sprintf("blk.%d.", il)
		if il > 0 {
			tensors = append(tensors, norm(p+"attn_norm.weight"))
		}
		tensors = append(tensors,
			mat(p+"attn_qkv.weight", s.embd, 3*s.embd, 1/float32(math.Sqrt(float64(s.embd)))),
			mat(p+"attn_output.weight", s.embd, s.embd, 1/float32(math.Sqrt(float64(s.embd)))),
			norm(p+"ffn_norm.weight"),
			mat(p+"ffn_up.weight", s.embd, 2*s.ff, 1/float32(math.Sqrt(float64(s.embd)))),
			mat(p+"ffn_down.weight", s.ff, s.embd, 1/float32(math.Sqrt(float64(s.ff)))),
		)
	}
	return writeGGUF(path, kvs, tensors)
}

func writeGGUF(path string, kvs []kv, tensors []tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	le := binary.LittleEndian
	u32 := func(v uint32) { binary.Write(w, le, v) }
	u64 := func(v uint64) { binary.Write(w, le, v) }
	str := func(s string) { u64(uint64(len(s))); w.WriteString(s) }
	w.WriteString("GGUF")
	u32(3)
	u64(uint64(len(tensors)))
	u64(uint64(len(kvs)))
	for _, e := range kvs {
		str(e.key)
		u32(e.typ)
		switch v := e.val.(type) {
		case uint32:
			u32(v)
		case float32:
			binary.Write(w, le, v)
		case bool:
			if v {
				w.WriteByte(1)
			} else {
				w.WriteByte(0)
			}
		case string:
			str(v)
		case []string:
			u32(ggufString)
			u64(uint64(len(v)))
			for _, s := range v {
				str(s)
			}
		case []float32:
			u32(ggufFloat32)
			u64(uint64(len(v)))
			binary.Write(w, le, v)
		case []int32:
			u32(ggufInt32)
			u64(uint64(len(v)))
			binary.Write(w, le, v)
		default:
			return fmt.Errorf("gguf: unsupported value %T for %s", e.val, e.key)
		}
	}
	var off uint64
	for _, t := range tensors {
		str(t.name)
		u32(uint32(len(t.dims)))
		for _, d := range t.dims {
			u64(d)
		}
		u32(t.typ)
		u64(off)
		off = (off + uint64(len(t.data)) + alignment - 1) / alignment * alignment
	}
	w.Flush()
	pos, _ := f.Seek(0, 1)
	pad := (alignment - pos%alignment) % alignment
	w.Write(make([]byte, pad))
	for _, t := range tensors {
		w.Write(t.data)
		pad := (alignment - len(t.data)%alignment) % alignment
		w.Write(make([]byte, pad))
	}
	return w.Flush()
}

func f32Bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

// q8_0Bytes quantizes rows of ne0 values in blocks of 32: an f16 scale
// (max|x|/127) and 32 int8.
func q8_0Bytes(w []float32, ne0 int) []byte {
	const blk = 32
	nb := len(w) / blk
	out := make([]byte, nb*34)
	for b := 0; b < nb; b++ {
		x := w[b*blk : (b+1)*blk]
		var amax float32
		for _, v := range x {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		d := amax / 127
		inv := float32(0)
		if d != 0 {
			inv = 1 / d
		}
		o := out[b*34:]
		binary.LittleEndian.PutUint16(o, f32ToF16(d))
		for i, v := range x {
			o[2+i] = byte(int8(math.Round(float64(v * inv))))
		}
	}
	return out
}

// f32ToF16 converts to IEEE half, round to nearest even.
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		h := mant >> shift
		if rem := mant & (1<<shift - 1); rem > 1<<(shift-1) || (rem == 1<<(shift-1) && h&1 == 1) {
			h++
		}
		return sign | uint16(h)
	case exp >= 0x1f:
		return sign | 0x7c00
	}
	h := uint32(exp)<<10 | mant>>13
	if rem := mant & 0x1fff; rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
		h++
	}
	return sign | uint16(h)
}
