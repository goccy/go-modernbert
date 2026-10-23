# go-modernbert

A pure Go runtime for the ModernBERT encoder (sbintuitions/modernbert-ja
and the like) from a GGUF file. It does one thing: turn a token sequence
into the final hidden state, for the CLS vector of an embedding or a
classifier. It is a fixed forward pass over a fixed architecture, not a
graph engine.

```go
m, err := modernbert.LoadWithOptions("model.gguf", modernbert.Options{Threads: 4})
h, err := m.EncodeText("東京の天気を教えてください。") // [tokens * NEmbd]; the CLS vector is h[:m.NEmbd]
```

`EncodeText` tokenizes with the model's own vocabulary: a SentencePiece
Unigram tokenizer (`Tokenizer`) built from the GGUF metadata, with byte
fallback, literal special tokens and the BOS/EOS the model's config
specifies; it reproduces the HuggingFace fast tokenizer of the released
checkpoints (testdata/tokenizer-reference.json). `Encode` takes token ids
directly.

## How it computes

Every linear layer and both attention products run on one int8 tile:
eight activation rows against blocks of weight rows, int32 accumulation
over the whole K, one scale per row and per column applied at the end.
Weights are quantized per output row at load, activations per token at
run time, attention keys per token and values per dimension; the softmax
runs in float32 and its weights go to the value product as unsigned bytes.

| Architecture | Tile | Operands |
|---|---|---|
| arm64 with i8mm | SMMLA 8×8 | int8 × int8 |
| amd64 with AVX-512 VNNI | VPDPBUSD 8×16 | u8 (offset 128) × int8 |
| amd64 with AVX2 | VPMADDUBSW/VPMADDWD 8×8 | 7-bit (activations and values) × int8 |

LayerNorm, the row quantizers, the softmax and GeGLU have NEON and AVX2
versions. A CPU without a tile runs the linear layers on a smaller
dot-product tile (SDOT, or AVX2 with the sign trick) or in Go, and
attention in float32. The only dependency is `golang.org/x/sys` for CPU
feature detection.

`Options.Quant` selects the path explicitly (`QuantInt8Row`, the default;
`QuantF32` for a scalar float32 reference). `Options.Kernel` selects the
kernels (`KernelAuto` picks the best the CPU advertises; `KernelNone`,
`KernelSMMLA`, `KernelAVX2`, `KernelVNNI` name one, which then runs
whether or not the CPU advertises it). `Options.Threads` is the number of
goroutines a forward pass spreads over; workspaces and worker pools are
reused across calls.

## Testing

```sh
go test ./...                                    # kernels against scalar references
MODERNBERT_GGUF=model.gguf go test -run TestParity ./...   # against testdata/*-reference.json (PyTorch)
MODERNBERT_GGUF=model.gguf go test -run NONE -bench BenchmarkEncode ./...
MODERNBERT_KERNEL=avx2 GOARCH=amd64 go test ./...         # the AVX2 path under Rosetta
```

`MODERNBERT_THREADS`, `MODERNBERT_QUANT` (`f32`, `int8`, `w8`) and
`MODERNBERT_KERNEL` (`none`, `smmla`, `avx2`, `vnni`) select the thread
count, path and kernels of the tests and benchmarks; `MODERNBERT_COS_MIN`
relaxes the parity cosine threshold for the int8 paths. The environment
is read by the tests only. `BenchmarkEncodeSynthetic` times a
130m-shaped model with random weights and needs no file.
