// Command benchmarks times a ModernBERT forward pass in go-modernbert and
// in go-llama on the same GGUF, for 32 / 128 / 512 synthetic tokens.
//
//	go run . -gen model.gguf            # a 130m-shaped model with random weights
//	go run . -gguf model.gguf -threads 1
package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"runtime"
	"time"

	llama "github.com/goccy/go-llama"
	modernbert "github.com/goccy/go-modernbert"
)

func main() {
	gen := flag.String("gen", "", "write a synthetic GGUF to this path and exit")
	gguf := flag.String("gguf", "", "GGUF to benchmark")
	threads := flag.Int("threads", 1, "threads")
	reps := flag.Int("reps", 5, "timed repetitions per length")
	only := flag.String("only", "", "run one runtime: modernbert or llama")
	flag.Parse()
	if *gen != "" {
		if err := writeSyntheticGGUF(*gen, shape130m, 1); err != nil {
			log.Fatal(err)
		}
		fmt.Println("wrote", *gen)
		return
	}
	if *gguf == "" {
		log.Fatal("set -gguf or -gen")
	}
	fmt.Printf("%s/%s GOMAXPROCS=%d threads=%d\n", runtime.GOOS, runtime.GOARCH, runtime.GOMAXPROCS(0), *threads)
	if *only != "llama" {
		benchModernbert(*gguf, *threads, *reps)
	}
	if *only != "modernbert" {
		benchLlama(*gguf, *threads, *reps)
	}
}

func tokens(n int) []int32 {
	toks := make([]int32, n)
	toks[0], toks[n-1] = 1, 2
	for i := 1; i < n-1; i++ {
		toks[i] = int32(10 + (i*37)%4000)
	}
	return toks
}

func report(name string, n int, ds []time.Duration) {
	var total, best time.Duration
	best = time.Hour
	for _, d := range ds {
		total += d
		best = min(best, d)
	}
	avg := total / time.Duration(len(ds))
	fmt.Printf("%-14s T=%4d  avg %8.1f ms  best %8.1f ms  (%5.0f tok/s)\n", name, n, float64(avg.Microseconds())/1000, float64(best.Microseconds())/1000, float64(n)/avg.Seconds())
}

func benchModernbert(path string, threads, reps int) {
	m, err := modernbert.LoadWithOptions(path, modernbert.Options{Threads: threads})
	if err != nil {
		log.Fatal(err)
	}
	for _, n := range []int{32, 128, 512} {
		toks := tokens(n)
		for i := 0; i < 2; i++ {
			if _, err := m.Encode(toks); err != nil {
				log.Fatal(err)
			}
		}
		ds := make([]time.Duration, reps)
		for i := range ds {
			t0 := time.Now()
			if _, err := m.Encode(toks); err != nil {
				log.Fatal(err)
			}
			ds[i] = time.Since(t0)
		}
		report("go-modernbert", n, ds)
	}
}

func benchLlama(path string, threads, reps int) {
	dir, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		log.Fatal(err)
	}
	inst, err := llama.New(llama.WithPreopenDir(dir))
	if err != nil {
		log.Fatal(err)
	}
	defer inst.Close()
	m, err := inst.LoadModel(filepath.Base(path))
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 1024, NBatch: 1024, NUBatch: 1024, NThreads: uint32(threads), Embeddings: true})
	if err != nil {
		log.Fatal(err)
	}
	defer ctx.Close()
	for _, n := range []int{32, 128, 512} {
		toks := tokens(n)
		for i := 0; i < 2; i++ {
			if _, err := ctx.EmbedTokens(toks, false); err != nil {
				log.Fatal(err)
			}
		}
		ds := make([]time.Duration, reps)
		for i := range ds {
			t0 := time.Now()
			if _, err := ctx.EmbedTokens(toks, false); err != nil {
				log.Fatal(err)
			}
			ds[i] = time.Since(t0)
		}
		report("go-llama", n, ds)
	}
}
