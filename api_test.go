package modernbert

import (
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestOptionsRejected(t *testing.T) {
	c := config130m
	c.NLayer = 1
	m := syntheticModel(t, c, Options{})
	if err := m.prepare(Options{Quant: Quant(99)}); err == nil || !strings.Contains(err.Error(), "unknown quantization") {
		t.Fatalf("Quant(99): error %v", err)
	}
	if err := m.prepare(Options{Kernel: Kernel(99)}); err == nil {
		t.Fatal("Kernel(99) accepted")
	}
}

func TestEncodeTooLong(t *testing.T) {
	c := config130m
	c.NLayer = 1
	c.NCtxTrain = 16
	m := syntheticModel(t, c, Options{Kernel: testKernel(t)})
	if _, err := m.Encode(make([]int32, 17)); err == nil || !strings.Contains(err.Error(), "context length") {
		t.Fatalf("17 tokens over a context of 16: error %v", err)
	}
	if _, err := m.Encode(make([]int32, 16)); err != nil {
		t.Fatal(err)
	}
}

// TestClose checks that Close releases the pooled workers.
func TestClose(t *testing.T) {
	c := config130m
	c.NLayer = 1
	runtime.GC()
	base := runtime.NumGoroutine()
	m := syntheticModel(t, c, Options{Kernel: testKernel(t), Threads: 4})
	if _, err := m.Encode([]int32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if runtime.NumGoroutine() <= base {
		t.Fatal("no worker goroutines to release")
	}
	m.Close()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) && runtime.NumGoroutine() > base {
		time.Sleep(20 * time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > base {
		t.Fatalf("%d goroutines remain after Close (base %d)", n, base)
	}
	// Still usable.
	if _, err := m.Encode([]int32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	m.Close()
}

func TestTokenizerVocabularyChecks(t *testing.T) {
	kv := func() map[string]any {
		return map[string]any{
			"tokenizer.ggml.model":            "t5",
			"tokenizer.ggml.tokens":           []string{"<unk>", "<s>", "</s>", "a"},
			"tokenizer.ggml.scores":           []float32{0, 0, 0, -1},
			"tokenizer.ggml.token_type":       []int32{2, 3, 3, 1},
			"tokenizer.ggml.unknown_token_id": uint32(0),
			"tokenizer.ggml.add_bos_token":    true,
			"tokenizer.ggml.bos_token_id":     uint32(1),
		}
	}
	if _, err := newTokenizer(kv()); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ key, want string }{
		{"tokenizer.ggml.bos_token_id", "outside the vocabulary"},
		{"tokenizer.ggml.unknown_token_id", "outside the vocabulary"},
	} {
		m := kv()
		m[tc.key] = uint32(1 << 31)
		if _, err := newTokenizer(m); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s = 2^31: error %v", tc.key, err)
		}
	}
}
