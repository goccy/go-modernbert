package modernbert

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// pool runs the parallel regions of one forward pass on a fixed set of
// workers. A pass is a few hundred short regions back to back, so the
// workers spin on the job counter between regions instead of parking: a
// park/unpark per region per worker costs more (a futex or condition
// variable round trip each) than the regions themselves. A worker that has
// spun for a while without a job parks on its own wake channel, and the
// pool signals every channel with the next job; close releases the workers.
type pool struct {
	workers int // besides the caller
	job     atomic.Uint64
	done    atomic.Int64
	wake    []chan struct{} // one per worker, capacity 1
	quit    atomic.Bool
	wg      sync.WaitGroup

	mu sync.Mutex // the current job
	fn func(worker int)
}

// spinFor is how long a worker (or the caller waiting for the workers)
// spins without a job before parking.
const spinFor = 200 * time.Microsecond

func newPool(threads int) *pool {
	p := &pool{workers: max(threads, 1) - 1}
	p.wake = make([]chan struct{}, p.workers)
	for w := range p.wake {
		p.wake[w] = make(chan struct{}, 1)
		p.wg.Add(1)
		go p.worker(w + 1)
	}
	return p
}

func (p *pool) worker(id int) {
	defer p.wg.Done()
	wake := p.wake[id-1]
	seen := uint64(0)
	for {
		spins := 0
		var since time.Time
		for p.job.Load() == seen {
			if p.quit.Load() {
				return
			}
			spins++
			if spins&1023 != 0 {
				continue
			}
			if since.IsZero() {
				since = time.Now()
				continue
			}
			if time.Since(since) < spinFor {
				continue
			}
			// A stale token (sent for a job seen while spinning) costs one
			// more check of the counter.
			<-wake
			spins, since = 0, time.Time{}
		}
		seen = p.job.Load()
		p.mu.Lock()
		fn := p.fn
		p.mu.Unlock()
		fn(id)
		p.done.Add(1)
	}
}

// signal gives every worker a wake token; a worker whose token from the
// previous job is still pending keeps that one.
func (p *pool) signal() {
	for _, wake := range p.wake {
		select {
		case wake <- struct{}{}:
		default:
		}
	}
}

// run executes fn(worker) on the caller (worker 0) and every worker, and
// returns when all are done.
func (p *pool) run(fn func(worker int)) {
	if p.workers == 0 {
		fn(0)
		return
	}
	p.mu.Lock()
	p.fn = fn
	p.mu.Unlock()
	p.done.Store(0)
	p.job.Add(1)
	p.signal()
	fn(0)
	spins := 0
	var since time.Time
	for p.done.Load() < int64(p.workers) {
		spins++
		if spins&1023 != 0 {
			continue
		}
		if since.IsZero() {
			since = time.Now()
		} else if time.Since(since) >= spinFor {
			runtime.Gosched()
		}
	}
	// The parked workers keep the pool reachable; the pool must not keep
	// the last job (and whatever it captured) reachable in turn.
	p.mu.Lock()
	p.fn = nil
	p.mu.Unlock()
}

func (p *pool) close() {
	p.quit.Store(true)
	p.signal()
	p.wg.Wait()
}

// parallelFor splits [0, n) into contiguous chunks whose bounds are
// multiples of align (the last one takes the remainder), about four per
// thread, which the caller and the workers of p take in order until none
// are left; it returns when all are done. A nil pool, or n ≤ align, runs
// fn(0, n) inline.
func parallelFor(p *pool, n, align int, fn func(lo, hi int)) {
	parallelForWorker(p, n, align, func(_, lo, hi int) { fn(lo, hi) })
}

// parallelForWorker is parallelFor with the running thread's index (0
// for the caller, 1.. for the workers) passed to fn, for per-thread
// scratch.
func parallelForWorker(p *pool, n, align int, fn func(w, lo, hi int)) {
	if p == nil || p.workers == 0 || n <= align {
		fn(0, 0, n)
		return
	}
	threads := p.workers + 1
	chunk := (n + 4*threads - 1) / (4 * threads)
	chunk = (chunk + align - 1) / align * align
	var next atomic.Int64
	p.run(func(w int) {
		for {
			lo := int(next.Add(int64(chunk))) - chunk
			if lo >= n {
				return
			}
			fn(w, lo, min(lo+chunk, n))
		}
	})
}
