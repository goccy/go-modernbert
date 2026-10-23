package modernbert

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// TestTokenizer compares Encode with the HuggingFace fast tokenizer's ids
// (testdata/tokenizer-reference.json). Opt-in via MODERNBERT_GGUF, a
// conversion of the same checkpoint.
func TestTokenizer(t *testing.T) {
	path := os.Getenv("MODERNBERT_GGUF")
	if path == "" {
		t.Skip("set MODERNBERT_GGUF to run")
	}
	raw, err := os.ReadFile("testdata/tokenizer-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := LoadWithOptions(path, Options{Quant: QuantInt8RowWeights})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := m.Tokenizer()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range ref.Cases {
		got := tok.Encode(c.Text)
		if !slices.Equal(got, c.IDs) {
			t.Errorf("%q:\n got  %v\n want %v", c.Text, got, c.IDs)
		}
	}
}
