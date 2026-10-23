package modernbert

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// A minimal GGUF reader: the metadata (key/value) section, the tensor infos
// and the tensor data of the types this package consumes (F32, F16, Q8_0).
// Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

const ggufMagic = "GGUF"

type ggufType uint32

const (
	ggufUint8   ggufType = 0
	ggufInt8    ggufType = 1
	ggufUint16  ggufType = 2
	ggufInt16   ggufType = 3
	ggufUint32  ggufType = 4
	ggufInt32   ggufType = 5
	ggufFloat32 ggufType = 6
	ggufBool    ggufType = 7
	ggufString  ggufType = 8
	ggufArray   ggufType = 9
	ggufUint64  ggufType = 10
	ggufInt64   ggufType = 11
	ggufFloat64 ggufType = 12
)

// ggml tensor types this reader decodes.
const (
	ggmlF32  uint32 = 0
	ggmlF16  uint32 = 1
	ggmlQ8_0 uint32 = 8
)

// q8_0 block: an f16 scale and 32 int8 values.
const q8_0BlockSize = 32
const q8_0BlockBytes = 2 + q8_0BlockSize

type ggufTensor struct {
	name   string
	typ    uint32
	shape  []uint64 // ne[0] (the row length) first
	offset uint64   // from the start of the data section
}

type ggufFile struct {
	data      []byte // the whole file
	kv        map[string]any
	tensors   map[string]*ggufTensor
	dataStart uint64
}

// readGGUF reads the whole file at path into memory and parses its header.
func readGGUF(path string) (*ggufFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	r := &ggufReader{b: data}
	if string(r.take(4)) != ggufMagic {
		return nil, fmt.Errorf("gguf: %s is not a GGUF file", path)
	}
	version := r.u32()
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}
	nTensors := r.u64()
	nKV := r.u64()
	if r.err != nil {
		return nil, r.err
	}
	f := &ggufFile{data: data, kv: make(map[string]any, nKV), tensors: make(map[string]*ggufTensor, nTensors)}
	for i := uint64(0); i < nKV && r.err == nil; i++ {
		key := r.str()
		typ := ggufType(r.u32())
		f.kv[key] = r.value(typ)
	}
	alignment := uint64(32)
	if a, ok := f.kv["general.alignment"].(uint32); ok && a != 0 {
		alignment = uint64(a)
	}
	for i := uint64(0); i < nTensors && r.err == nil; i++ {
		t := &ggufTensor{name: r.str()}
		nd := r.u32()
		if nd > 4 {
			return nil, fmt.Errorf("gguf: tensor %s has %d dims", t.name, nd)
		}
		for d := uint32(0); d < nd; d++ {
			t.shape = append(t.shape, r.u64())
		}
		t.typ = r.u32()
		t.offset = r.u64()
		f.tensors[t.name] = t
	}
	if r.err != nil {
		return nil, r.err
	}
	f.dataStart = (uint64(r.pos) + alignment - 1) / alignment * alignment
	return f, nil
}

// tensor returns the named tensor's info and raw bytes.
func (f *ggufFile) tensor(name string) (*ggufTensor, []byte, error) {
	t, ok := f.tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("gguf: tensor %s missing", name)
	}
	n := uint64(1)
	for _, d := range t.shape {
		n *= d
	}
	var size uint64
	switch t.typ {
	case ggmlF32:
		size = n * 4
	case ggmlF16:
		size = n * 2
	case ggmlQ8_0:
		if t.shape[0]%q8_0BlockSize != 0 {
			return nil, nil, fmt.Errorf("gguf: tensor %s: q8_0 row length %d is not a multiple of %d", name, t.shape[0], q8_0BlockSize)
		}
		size = n / q8_0BlockSize * q8_0BlockBytes
	default:
		return nil, nil, fmt.Errorf("gguf: tensor %s has unsupported type %d", name, t.typ)
	}
	start := f.dataStart + t.offset
	if start+size > uint64(len(f.data)) {
		return nil, nil, fmt.Errorf("gguf: tensor %s runs past the end of the file", name)
	}
	return t, f.data[start : start+size], nil
}

// f32 returns the named tensor as float32 values (row-major, ne[0] fastest),
// dequantizing q8_0 and converting f16.
func (f *ggufFile) f32(name string) ([]float32, *ggufTensor, error) {
	t, raw, err := f.tensor(name)
	if err != nil {
		return nil, nil, err
	}
	n := 1
	for _, d := range t.shape {
		n *= int(d)
	}
	out := make([]float32, n)
	switch t.typ {
	case ggmlF32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case ggmlF16:
		for i := range out {
			out[i] = f16ToF32(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case ggmlQ8_0:
		for b := 0; b < n/q8_0BlockSize; b++ {
			blk := raw[b*q8_0BlockBytes:]
			d := f16ToF32(binary.LittleEndian.Uint16(blk))
			for i := 0; i < q8_0BlockSize; i++ {
				out[b*q8_0BlockSize+i] = d * float32(int8(blk[2+i]))
			}
		}
	}
	return out, t, nil
}

// uint32KV / float32KV / stringKV fetch typed metadata.
func (f *ggufFile) uint32KV(key string) (uint32, error) {
	v, ok := f.kv[key].(uint32)
	if !ok {
		return 0, fmt.Errorf("gguf: metadata %s missing or not uint32 (%T)", key, f.kv[key])
	}
	return v, nil
}

func (f *ggufFile) float32KV(key string) (float32, error) {
	v, ok := f.kv[key].(float32)
	if !ok {
		return 0, fmt.Errorf("gguf: metadata %s missing or not float32 (%T)", key, f.kv[key])
	}
	return v, nil
}

func (f *ggufFile) stringKV(key string) (string, error) {
	v, ok := f.kv[key].(string)
	if !ok {
		return "", fmt.Errorf("gguf: metadata %s missing or not string (%T)", key, f.kv[key])
	}
	return v, nil
}

// f16ToF32 converts an IEEE half to float32.
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: normalize.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}

// ggufReader decodes little-endian primitives from a byte slice; the first
// error sticks and later reads return zero values.
type ggufReader struct {
	b   []byte
	pos int
	err error
}

func (r *ggufReader) take(n int) []byte {
	if r.err != nil {
		return nil
	}
	if r.pos+n > len(r.b) {
		r.err = fmt.Errorf("gguf: truncated at offset %d", r.pos)
		return nil
	}
	s := r.b[r.pos : r.pos+n]
	r.pos += n
	return s
}

func (r *ggufReader) u32() uint32 {
	b := r.take(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (r *ggufReader) u64() uint64 {
	b := r.take(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (r *ggufReader) str() string {
	n := r.u64()
	if n > 1<<24 {
		r.err = fmt.Errorf("gguf: string length %d too large", n)
		return ""
	}
	return string(r.take(int(n)))
}

func (r *ggufReader) value(typ ggufType) any {
	switch typ {
	case ggufUint8:
		b := r.take(1)
		if b == nil {
			return nil
		}
		return b[0]
	case ggufInt8:
		b := r.take(1)
		if b == nil {
			return nil
		}
		return int8(b[0])
	case ggufUint16:
		b := r.take(2)
		if b == nil {
			return nil
		}
		return binary.LittleEndian.Uint16(b)
	case ggufInt16:
		b := r.take(2)
		if b == nil {
			return nil
		}
		return int16(binary.LittleEndian.Uint16(b))
	case ggufUint32:
		return r.u32()
	case ggufInt32:
		return int32(r.u32())
	case ggufFloat32:
		return math.Float32frombits(r.u32())
	case ggufBool:
		b := r.take(1)
		if b == nil {
			return nil
		}
		return b[0] != 0
	case ggufString:
		return r.str()
	case ggufUint64:
		return r.u64()
	case ggufInt64:
		return int64(r.u64())
	case ggufFloat64:
		return math.Float64frombits(r.u64())
	case ggufArray:
		et := ggufType(r.u32())
		n := r.u64()
		if n > 1<<26 {
			r.err = fmt.Errorf("gguf: array length %d too large", n)
			return nil
		}
		switch et {
		case ggufString:
			out := make([]string, 0, min(n, 1<<16))
			for i := uint64(0); i < n && r.err == nil; i++ {
				out = append(out, r.str())
			}
			return out
		case ggufFloat32:
			out := make([]float32, 0, min(n, 1<<16))
			for i := uint64(0); i < n && r.err == nil; i++ {
				out = append(out, math.Float32frombits(r.u32()))
			}
			return out
		case ggufInt32:
			out := make([]int32, 0, min(n, 1<<16))
			for i := uint64(0); i < n && r.err == nil; i++ {
				out = append(out, int32(r.u32()))
			}
			return out
		case ggufUint32:
			out := make([]uint32, 0, min(n, 1<<16))
			for i := uint64(0); i < n && r.err == nil; i++ {
				out = append(out, r.u32())
			}
			return out
		default:
			// Other element types are skipped value by value.
			out := make([]any, 0, min(n, 1<<16))
			for i := uint64(0); i < n && r.err == nil; i++ {
				out = append(out, r.value(et))
			}
			return out
		}
	}
	r.err = fmt.Errorf("gguf: unknown value type %d", typ)
	return nil
}
