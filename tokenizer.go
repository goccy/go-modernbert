package modernbert

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
)

// Tokenizer is the SentencePiece Unigram tokenizer of a model, read from
// the GGUF vocabulary (tokenizer.ggml.tokens / scores / token_type): the
// text has every space replaced by "▁", the vocabulary pieces are laid
// over it by Viterbi for the highest total score, characters no piece
// covers become byte pieces (<0xXX>), and BOS / EOS wrap the sequence as
// the model's config says. There is no normalization, as in the released
// tokenizer.json of the models this runtime targets.
type Tokenizer struct {
	pieces   []string
	scores   []float32
	trie     trieNode
	special  trieNode   // control and unknown pieces, matched on the raw text
	byteID   [256]int32 // id of <0xXX>, or -1
	unk      int32
	bos      int32 // -1 when not added
	eos      int32 // -1 when not added
	minScore float32
}

// Token types of the GGUF vocabulary.
const (
	tokenNormal  = 1
	tokenUnknown = 2
	tokenControl = 3
	tokenByte    = 6
)

// unkPenalty is subtracted from the lowest piece score for a character no
// piece covers (the value the reference implementation uses).
const unkPenalty = 10

const metaspace = "▁"

// Tokenizer returns the model's tokenizer, built on first use.
func (m *Model) Tokenizer() (*Tokenizer, error) {
	m.tokOnce.Do(func() { m.tok, m.tokErr = newTokenizer(m.Metadata) })
	return m.tok, m.tokErr
}

func newTokenizer(kv map[string]any) (*Tokenizer, error) {
	model, _ := kv["tokenizer.ggml.model"].(string)
	if model != "t5" {
		return nil, fmt.Errorf("modernbert: tokenizer model %q is not a SentencePiece Unigram (t5) vocabulary", model)
	}
	pieces, _ := kv["tokenizer.ggml.tokens"].([]string)
	scores, _ := kv["tokenizer.ggml.scores"].([]float32)
	types, _ := kv["tokenizer.ggml.token_type"].([]int32)
	if len(pieces) == 0 || len(scores) != len(pieces) || len(types) != len(pieces) {
		return nil, fmt.Errorf("modernbert: tokenizer vocabulary incomplete (%d tokens, %d scores, %d types)", len(pieces), len(scores), len(types))
	}
	t := &Tokenizer{pieces: pieces, scores: scores, unk: -1, bos: -1, eos: -1, minScore: float32(math.Inf(1))}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	for id, p := range pieces {
		switch types[id] {
		case tokenNormal:
			t.trie.insert(p, int32(id))
			t.minScore = min(t.minScore, scores[id])
		case tokenUnknown:
			t.unk = int32(id)
			t.special.insert(p, int32(id))
		case tokenControl:
			t.special.insert(p, int32(id))
		case tokenByte:
			var b int
			if _, err := fmt.Sscanf(p, "<0x%02X>", &b); err == nil && b < 256 {
				t.byteID[b] = int32(id)
			}
		}
	}
	if id, ok := kv["tokenizer.ggml.unknown_token_id"].(uint32); ok {
		t.unk = int32(id)
	}
	if t.unk < 0 {
		return nil, fmt.Errorf("modernbert: tokenizer has no unknown token")
	}
	if add, _ := kv["tokenizer.ggml.add_bos_token"].(bool); add {
		if id, ok := kv["tokenizer.ggml.bos_token_id"].(uint32); ok {
			t.bos = int32(id)
		}
	}
	if add, _ := kv["tokenizer.ggml.add_eos_token"].(bool); add {
		if id, ok := kv["tokenizer.ggml.eos_token_id"].(uint32); ok {
			t.eos = int32(id)
		}
	}
	return t, nil
}

// trieNode is a byte trie over the pieces; id is the piece ending here
// (-1 for none).
type trieNode struct {
	id   int32
	next map[byte]*trieNode
}

func (n *trieNode) insert(p string, id int32) {
	for i := 0; i < len(p); i++ {
		if n.next == nil {
			n.next = make(map[byte]*trieNode)
		}
		c := n.next[p[i]]
		if c == nil {
			c = &trieNode{id: -1}
			n.next[p[i]] = c
		}
		n = c
	}
	n.id = id
}

// Encode tokenizes text into ids, with BOS and EOS when the model adds
// them.
func (t *Tokenizer) Encode(text string) []int32 {
	var out []int32
	if t.bos >= 0 {
		out = append(out, t.bos)
	}
	// Special pieces written literally in the text are tokens of their
	// own (the longest match at each position) and split the text into
	// segments tokenized separately.
	seg := 0
	for i := 0; i < len(text); {
		id, size := t.special.longest(text[i:])
		if id < 0 {
			_, r := utf8.DecodeRuneInString(text[i:])
			i += r
			continue
		}
		out = t.appendSegment(out, text[seg:i])
		out = append(out, id)
		i += size
		seg = i
	}
	out = t.appendSegment(out, text[seg:])
	if t.eos >= 0 {
		out = append(out, t.eos)
	}
	return out
}

// longest returns the id and byte length of the longest piece of the
// trie that prefixes s, or -1.
func (n *trieNode) longest(s string) (int32, int) {
	id, size := int32(-1), 0
	for i := 0; i < len(s); i++ {
		n = n.next[s[i]]
		if n == nil {
			break
		}
		if n.id >= 0 {
			id, size = n.id, i+1
		}
	}
	return id, size
}

// appendSegment appends the Unigram tokenization of text to out.
func (t *Tokenizer) appendSegment(out []int32, text string) []int32 {
	if text == "" {
		return out
	}
	s := strings.ReplaceAll(text, " ", metaspace)
	n := len(s)
	// Viterbi over byte positions: best[i] is the best score of a path
	// covering s[:i], from[i] / id[i] its last piece (id -1: an unknown
	// character whose bytes become byte pieces).
	best := make([]float32, n+1)
	from := make([]int, n+1)
	ids := make([]int32, n+1)
	for i := 1; i <= n; i++ {
		best[i] = float32(math.Inf(-1))
	}
	for i := 0; i < n; i++ {
		if best[i] == float32(math.Inf(-1)) {
			continue
		}
		_, size := utf8.DecodeRuneInString(s[i:])
		single := false
		node := &t.trie
		for j := i; j < n; j++ {
			node = node.next[s[j]]
			if node == nil {
				break
			}
			if node.id >= 0 {
				if j+1 == i+size {
					single = true
				}
				if sc := best[i] + t.scores[node.id]; sc > best[j+1] {
					best[j+1], from[j+1], ids[j+1] = sc, i, node.id
				}
			}
		}
		if !single {
			if sc := best[i] + t.minScore - unkPenalty; sc > best[i+size] {
				best[i+size], from[i+size], ids[i+size] = sc, i, -1
			}
		}
	}
	start := len(out)
	for i := n; i > 0; i = from[i] {
		if ids[i] >= 0 {
			out = append(out, ids[i])
			continue
		}
		// Byte pieces, in reverse since the path is walked backwards.
		for b := i - 1; b >= from[i]; b-- {
			if id := t.byteID[s[b]]; id >= 0 {
				out = append(out, id)
			} else {
				out = append(out, t.unk)
			}
		}
	}
	for l, r := start, len(out)-1; l < r; l, r = l+1, r-1 {
		out[l], out[r] = out[r], out[l]
	}
	return out
}

// Piece returns the vocabulary string of id.
func (t *Tokenizer) Piece(id int32) string {
	if id < 0 || int(id) >= len(t.pieces) {
		return ""
	}
	return t.pieces[id]
}

// EncodeText tokenizes text and runs the encoder; see Encode.
func (m *Model) EncodeText(text string) ([]float32, error) {
	t, err := m.Tokenizer()
	if err != nil {
		return nil, err
	}
	return m.Encode(t.Encode(text))
}
