package modernbert

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
)

// TestParity compares the CLS vector with the PyTorch reference in
// testdata/modernbert-ja-130m-reference.json (sbintuitions/modernbert-ja-130m,
// transformers AutoModel, last_hidden_state[:, 0]). Opt-in via
// MODERNBERT_GGUF (an f32, f16 or q8_0 conversion of that checkpoint).
// MODERNBERT_REL_TOL (default 1e-2) bounds the largest element-wise
// difference relative to the largest reference magnitude; MODERNBERT_COS_MIN
// (default 0.999) the cosine similarity.
func TestParity(t *testing.T) {
	path := os.Getenv("MODERNBERT_GGUF")
	if path == "" {
		t.Skip("set MODERNBERT_GGUF to run")
	}
	relTol := envFloat(t, "MODERNBERT_REL_TOL", 1e-2)
	cosMin := envFloat(t, "MODERNBERT_COS_MIN", 0.999)
	var ref struct {
		Model string `json:"model"`
		NEmbd int    `json:"n_embd"`
		Tests []struct {
			Text   string    `json:"text"`
			Tokens []int32   `json:"tokens"`
			CLS    []float32 `json:"cls"`
		} `json:"tests"`
	}
	b, err := os.ReadFile("testdata/modernbert-ja-130m-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	quant := QuantDefault
	switch os.Getenv("MODERNBERT_QUANT") {
	case "f32":
		quant = QuantF32
	case "w8":
		quant = QuantInt8RowWeights
	case "int8":
		quant = QuantInt8Row
	}
	threads, _ := strconv.Atoi(os.Getenv("MODERNBERT_THREADS"))
	m, err := LoadWithOptions(path, Options{Quant: quant, Threads: threads})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("quant=%d %+v", quant, m.Config)
	if m.NEmbd != ref.NEmbd {
		t.Fatalf("n_embd %d, reference %d", m.NEmbd, ref.NEmbd)
	}
	for _, tc := range ref.Tests {
		h, err := m.Encode(tc.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		cls := h[:m.NEmbd]
		var maxd, maxr, dot, na, nb float64
		for i := range cls {
			a, b := float64(cls[i]), float64(tc.CLS[i])
			maxd = math.Max(maxd, math.Abs(a-b))
			maxr = math.Max(maxr, math.Abs(b))
			dot += a * b
			na += a * a
			nb += b * b
		}
		cos := dot / math.Sqrt(na*nb)
		rel := maxd / maxr
		t.Logf("%3d tokens  cls max|diff| %.2e (%.2e of max|ref| %.2e)  cos %.6f  %q", len(tc.Tokens), maxd, rel, maxr, cos, tc.Text[:min(24, len(tc.Text))])
		if rel > relTol && quant == QuantF32 {
			t.Errorf("CLS diverges from PyTorch: relative max|diff| %.3e > %.1e", rel, relTol)
		}
		if cos < cosMin {
			t.Errorf("CLS diverges from PyTorch: cosine %.6f < %.4f", cos, cosMin)
		}
	}
}

func envFloat(t *testing.T, key string, def float64) float64 {
	t.Helper()
	s := os.Getenv(key)
	if s == "" {
		return def
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("%s: %v", key, err)
	}
	return v
}
