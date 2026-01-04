package tokenizer

import (
	"testing"
)

func TestBPE(t *testing.T) {
	bpe := NewBPE()
	text := "hello hello hello world world, Hey what's your name? 1234"
	// Vocab size: 256 (base) + merges.
	// "hello " might become a token eventually.
	// Let's train to a small size.
	targetVocabSize := 260
	err := bpe.Train(text, targetVocabSize)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	if bpe.GetVocabSize() != targetVocabSize {
		t.Errorf("Expected vocab size %d, got %d", targetVocabSize, bpe.GetVocabSize())
	}

	// Test Round Trip
	encoded := bpe.Encode(text)
	decoded := bpe.Decode(encoded)

	if decoded != text {
		t.Errorf("Decode(Encode(text)) != text. Got %q, want %q", decoded, text)
	}

	t.Logf("Ids: %v", encoded)

	// Test generalization (simple)
	test2 := "hello world"
	enc2 := bpe.Encode(test2)
	dec2 := bpe.Decode(enc2)
	if dec2 != test2 {
		t.Errorf("Round trip failed for %q. Got %q", test2, dec2)
	}
}
