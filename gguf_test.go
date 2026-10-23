package modernbert

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// ggufBuilder assembles a GGUF image for tests.
type ggufBuilder struct {
	kv      []byte
	nKV     uint64
	tensors []byte
	nT      uint64
	data    []byte
}

func (b *ggufBuilder) str(dst []byte, s string) []byte {
	dst = binary.LittleEndian.AppendUint64(dst, uint64(len(s)))
	return append(dst, s...)
}

func (b *ggufBuilder) u32(key string, v uint32) *ggufBuilder {
	b.kv = b.str(b.kv, key)
	b.kv = binary.LittleEndian.AppendUint32(b.kv, uint32(ggufUint32))
	b.kv = binary.LittleEndian.AppendUint32(b.kv, v)
	b.nKV++
	return b
}

func (b *ggufBuilder) f32(key string, v float32) *ggufBuilder {
	b.kv = b.str(b.kv, key)
	b.kv = binary.LittleEndian.AppendUint32(b.kv, uint32(ggufFloat32))
	b.kv = binary.LittleEndian.AppendUint32(b.kv, math.Float32bits(v))
	b.nKV++
	return b
}

func (b *ggufBuilder) s(key, v string) *ggufBuilder {
	b.kv = b.str(b.kv, key)
	b.kv = binary.LittleEndian.AppendUint32(b.kv, uint32(ggufString))
	b.kv = b.str(b.kv, v)
	b.nKV++
	return b
}

// raw appends an already encoded key/value.
func (b *ggufBuilder) raw(key string, enc []byte) *ggufBuilder {
	b.kv = b.str(b.kv, key)
	b.kv = append(b.kv, enc...)
	b.nKV++
	return b
}

// tensor adds an info entry; data is placed at the next 32-byte offset.
func (b *ggufBuilder) tensor(name string, typ uint32, shape []uint64, data []byte) *ggufBuilder {
	return b.tensorAt(name, typ, shape, uint64(len(b.data)), data)
}

func (b *ggufBuilder) tensorAt(name string, typ uint32, shape []uint64, offset uint64, data []byte) *ggufBuilder {
	b.tensors = b.str(b.tensors, name)
	b.tensors = binary.LittleEndian.AppendUint32(b.tensors, uint32(len(shape)))
	for _, d := range shape {
		b.tensors = binary.LittleEndian.AppendUint64(b.tensors, d)
	}
	b.tensors = binary.LittleEndian.AppendUint32(b.tensors, typ)
	b.tensors = binary.LittleEndian.AppendUint64(b.tensors, offset)
	b.nT++
	b.data = append(b.data, data...)
	b.data = append(b.data, make([]byte, (32-len(b.data)%32)%32)...)
	return b
}

func (b *ggufBuilder) bytes() []byte {
	out := []byte("GGUF")
	out = binary.LittleEndian.AppendUint32(out, 3)
	out = binary.LittleEndian.AppendUint64(out, b.nT)
	out = binary.LittleEndian.AppendUint64(out, b.nKV)
	out = append(out, b.kv...)
	out = append(out, b.tensors...)
	out = append(out, make([]byte, (32-len(out)%32)%32)...)
	return append(out, b.data...)
}

// patchU64 overwrites the uint64 at off.
func patchU64(img []byte, off int, v uint64) []byte {
	out := append([]byte(nil), img...)
	binary.LittleEndian.PutUint64(out[off:], v)
	return out
}

func f32s(n int) []byte {
	out := make([]byte, 4*n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(1))
	}
	return out
}

// TestGGUFMalformed feeds the reader images that must be rejected with an
// error, never a panic or an unbounded allocation.
func TestGGUFMalformed(t *testing.T) {
	// A small valid image: one 8×4 f32 tensor.
	valid := func() *ggufBuilder {
		b := &ggufBuilder{}
		b.s("general.architecture", "modern-bert")
		b.tensor("w", ggmlF32, []uint64{8, 4}, f32s(32))
		return b
	}
	// [array[array[array[...]]]]: the value type, then per level the
	// element type and a count of one.
	deep := binary.LittleEndian.AppendUint32(nil, uint32(ggufArray))
	for i := 0; i < 1000; i++ {
		deep = binary.LittleEndian.AppendUint32(deep, uint32(ggufArray))
		deep = binary.LittleEndian.AppendUint64(deep, 1)
	}
	for _, tc := range []struct {
		name string
		img  []byte
		want string // substring of the error
	}{
		{"truncated header", valid().bytes()[:20], "truncated"},
		{"absurd kv count", patchU64(valid().bytes(), 16, 1<<30), "cannot fit"},
		{"absurd tensor count", patchU64(valid().bytes(), 8, 1<<40), "cannot fit"},
		{"nested arrays", (&ggufBuilder{}).raw("a", deep).bytes(), "nested"},
		{"huge string", (&ggufBuilder{}).raw("a", append(binary.LittleEndian.AppendUint32(nil, uint32(ggufString)), binary.LittleEndian.AppendUint64(nil, 1<<40)...)).bytes(), "runs past"},
		{"huge array", (&ggufBuilder{}).raw("a", append(binary.LittleEndian.AppendUint32(nil, uint32(ggufArray)), append(binary.LittleEndian.AppendUint32(nil, uint32(ggufUint8)), binary.LittleEndian.AppendUint64(nil, 1<<40)...)...)).bytes(), "runs past"},
		{"zero dims", (&ggufBuilder{}).tensor("w", ggmlF32, nil, f32s(0)).bytes(), "0 dims"},
		{"zero dim", (&ggufBuilder{}).tensor("w", ggmlF32, []uint64{0, 4}, nil).bytes(), "has dim 0"},
		{"duplicate tensor", valid().tensor("w", ggmlF32, []uint64{8, 4}, f32s(32)).bytes(), "twice"},
		{"offset past end", (&ggufBuilder{}).tensorAt("w", ggmlF32, []uint64{8}, 1<<40, f32s(8)).bytes(), "past the end"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseGGUF(tc.img)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %v, want one containing %q", err, tc.want)
			}
		})
	}
}

// TestGGUFTensorBounds covers size arithmetic that must not wrap.
func TestGGUFTensorBounds(t *testing.T) {
	for _, tc := range []struct {
		name   string
		shape  []uint64
		typ    uint32
		offset uint64
		want   string
	}{
		{"dim beyond file", []uint64{1 << 20, 1 << 20}, ggmlF32, 0, "has dim"},
		{"dim wraps size", []uint64{1 << 62}, ggmlF32, 0, "has dim"},
		{"product beyond file", []uint64{64, 64, 64, 64}, ggmlF32, 0, "more than the file"},
		{"offset wraps", []uint64{8}, ggmlF32, math.MaxUint64 - 4, "past the end"},
		{"offset before data", []uint64{8}, ggmlF32, math.MaxUint64, "past the end"},
		{"q8_0 row", []uint64{8}, ggmlQ8_0, 0, "multiple of 32"},
		{"unknown type", []uint64{8}, 99, 0, "unsupported type"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Dims are checked against the file length at parse time, so
			// an oversized one is rejected there; the rest reach tensor().
			img := (&ggufBuilder{}).tensorAt("w", tc.typ, tc.shape, tc.offset, f32s(8)).bytes()
			f, err := parseGGUF(img)
			if err == nil {
				_, _, err = f.tensor("w")
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %v, want one containing %q", err, tc.want)
			}
		})
	}
}

// TestGGUFTruncations cuts a valid image at every length; none may panic.
func TestGGUFTruncations(t *testing.T) {
	b := &ggufBuilder{}
	b.s("general.architecture", "modern-bert").u32("modern-bert.embedding_length", 8).f32("x", 1)
	b.tensor("w", ggmlF32, []uint64{8, 4}, f32s(32))
	img := b.bytes()
	if _, err := parseGGUF(img); err != nil {
		t.Fatal(err)
	}
	for n := 0; n < len(img); n++ {
		f, err := parseGGUF(img[:n])
		if err == nil {
			_, _, err = f.tensor("w")
		}
		if err == nil {
			t.Fatalf("image cut at %d of %d bytes was accepted", n, len(img))
		}
	}
}

// TestReadConfigRejects covers configurations Load must refuse.
func TestReadConfigRejects(t *testing.T) {
	base := func() *ggufBuilder {
		b := &ggufBuilder{}
		b.s("general.architecture", "modern-bert")
		b.u32("modern-bert.embedding_length", 16).u32("modern-bert.block_count", 1).u32("modern-bert.attention.head_count", 2)
		b.u32("modern-bert.feed_forward_length", 16).u32("modern-bert.vocab_size", 8).u32("modern-bert.context_length", 64)
		b.f32("modern-bert.attention.layer_norm_epsilon", 1e-5).f32("modern-bert.rope.freq_base", 10000)
		for i := 0; i < 4; i++ {
			b.tensor("t"+string(rune('a'+i)), ggmlF32, []uint64{8}, f32s(8))
		}
		return b
	}
	for _, tc := range []struct {
		name string
		key  string
		v    uint32
		want string
	}{
		{"zero heads", "modern-bert.attention.head_count", 0, "head count is 0"},
		{"zero hidden", "modern-bert.embedding_length", 0, "hidden size is 0"},
		{"zero ffn", "modern-bert.feed_forward_length", 0, "feed-forward width is 0"},
		{"zero vocab", "modern-bert.vocab_size", 0, "vocabulary is 0"},
		{"too many layers", "modern-bert.block_count", 1 << 24, "layers declared"},
		{"heads do not divide", "modern-bert.attention.head_count", 3, "not a multiple"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, err := parseGGUF(base().u32(tc.key, tc.v).bytes())
			if err != nil {
				t.Fatal(err)
			}
			// The later key wins in the map, as a duplicate would in a file.
			var c Config
			err = readConfig(f, &c)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %v, want one containing %q", err, tc.want)
			}
		})
	}
}
