package modernbert

import (
	"fmt"
	"os"
	"strconv"
	"testing"
)

// BenchmarkEncode times one forward pass for 32 / 128 / 512 synthetic
// tokens. Opt-in via MODERNBERT_GGUF.
func BenchmarkEncode(b *testing.B) {
	path := os.Getenv("MODERNBERT_GGUF")
	if path == "" {
		b.Skip("set MODERNBERT_GGUF to run")
	}
	threads, _ := strconv.Atoi(os.Getenv("MODERNBERT_THREADS"))
	quant := QuantDefault
	switch os.Getenv("MODERNBERT_QUANT") {
	case "f32":
		quant = QuantF32
	case "int8":
		quant = QuantInt8Row
	}
	m, err := LoadWithOptions(path, Options{Threads: threads, Quant: quant, Kernel: testKernel(b)})
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("threads=%d quant=%d", max(threads, 1), quant)
	for _, n := range []int{32, 128, 512} {
		toks := make([]int32, n)
		toks[0], toks[n-1] = 1, 2
		for i := 1; i < n-1; i++ {
			toks[i] = int32(1000 + (i*37)%50000)
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
