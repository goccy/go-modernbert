package modernbert

import (
	"runtime"
	"testing"
	"time"
)

// TestEncodeRepeatable checks that a result does not depend on what the
// reused workspace held before: a short input encoded fresh, after a
// longer input, and after a call that left the padding rows dirty, must
// give identical bytes, for every length around the row padding.
func TestEncodeRepeatable(t *testing.T) {
	c := config130m
	c.NLayer = 3
	m := syntheticModel(t, c, Options{Kernel: testKernel(t), Threads: 2})
	toks := func(n, seed int) []int32 {
		out := make([]int32, n)
		for i := range out {
			out[i] = int32(3 + (i*37+seed*11)%4000)
		}
		return out
	}
	encode := func(n, seed int) []float32 {
		h, err := m.Encode(toks(n, seed))
		if err != nil {
			t.Fatal(err)
		}
		return h
	}
	for _, n := range []int{1, 2, 7, 8, 9, 15, 16, 17, 33, 77} {
		fresh := encode(n, 1)
		encode(200, 2) // leaves rows past n in the workspace
		encode(n+1, 3)
		again := encode(n, 1)
		for i := range fresh {
			if fresh[i] != again[i] {
				t.Fatalf("T=%d: output[%d] = %v after other calls, %v fresh", n, i, again[i], fresh[i])
			}
		}
	}
}

// TestModelReclaimed checks that a dropped model with a worker pool is
// collected: its goroutines exit and its memory is returned.
func TestModelReclaimed(t *testing.T) {
	c := config130m
	c.NLayer = 2
	runtime.GC()
	base := runtime.NumGoroutine()
	for i := 0; i < 4; i++ {
		m := syntheticModel(t, c, Options{Kernel: testKernel(t), Threads: 4})
		if _, err := m.Encode([]int32{1, 5, 9, 2}); err != nil {
			t.Fatal(err)
		}
	}
	// Finalizers run after a collection and the workers park a little
	// later.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		time.Sleep(50 * time.Millisecond)
		if runtime.NumGoroutine() <= base {
			return
		}
	}
	t.Fatalf("%d goroutines remain (base %d): the models' worker pools were not released", runtime.NumGoroutine(), base)
}
