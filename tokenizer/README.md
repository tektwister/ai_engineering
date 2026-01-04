# Tokenizer (BPE Implementation)

A byte-level Byte Pair Encoding (BPE) tokenizer implementation in Go. This module demonstrates how modern LLMs like GPT-2 and GPT-3 tokenize text into subword units.

## About BPE Tokenizer

**Byte Pair Encoding (BPE)** is a simple yet powerful data compression algorithm that has become the de facto standard for tokenizing text in large language models. Unlike traditional word tokenizers that split on whitespace, BPE learns subword units, allowing it to handle rare words, misspellings, and new words efficiently.

### Why BPE?

1. **Vocabulary Size**: Reduces vocabulary from potentially millions of unique words to a manageable size (e.g., 50k tokens for GPT-2).
2. **Out-of-Vocabulary**: No unknown tokens—any text can be encoded since we start with byte-level primitives.
3. **Semantic Grouping**: Learns that "playing" and "played" share the token "play", enabling better generalization.
4. **Compression**: Frequently occurring character sequences are merged into single tokens, reducing sequence length.

## How BPE Works

BPE operates in two phases:

### Phase 1: Training

Given a corpus, BPE learns merge rules:

1. **Initialize**: Start with a vocabulary of all 256 possible bytes (0-255).
2. **Iterate**:
   - Find the most frequent adjacent pair of tokens in the corpus.
   - Merge that pair into a new token (e.g., tokens 120 and 121 merge into token 256).
   - Add the new token to the vocabulary.
   - Repeat until the vocabulary reaches the target size (e.g., 50,000 tokens).

**Example**:
```
Initial text: "hello hello world"
Bytes: [104, 101, 108, 108, 111, 32, ...]

Step 1: Most common pair is (104, 101) = "he", merge → token 256
Step 2: Most common pair is (108, 108) = "ll", merge → token 257
Step 3: Most common pair is (256, 257) = "hell", merge → token 258
... and so on
```

### Phase 2: Encoding

To encode new text:

1. **Initialize**: Convert text to bytes (initial tokens).
2. **Apply Merges**: Repeatedly find and apply the learned merge rules in priority order (order learned).
3. **Result**: A sequence of token IDs representing the original text.

## Project Structure

```
tokenizer/
├── bpe.go          # BPE implementation (Tokenizer interface, BPE struct)
├── bpe_test.go     # Unit tests for BPE training and encoding/decoding
├── go.mod          # Go module definition
└── README.md       # This file
```

## Architecture

### BPE Struct

```go
type BPE struct {
    vocab     map[string]int    // "hello" -> 256
    vocabInv  map[int]string    // 256 -> "hello"
    merges    map[string]int    // "104,101" (pair) -> 256 (new token ID)
    mergesInv map[int]string    // 256 -> "104,101" (for reference)
}
```

### Tokenizer Interface

All tokenizers implement this interface:

```go
type Tokenizer interface {
    Encode(text string) []int      // Text → Token IDs
    Decode(ids []int) string       // Token IDs → Text
    Train(text string, vocabSize int) error  // Learn merges
}
```

## Usage

### Training a Tokenizer

```go
import "github.com/tektwister/ai_engineering/tokenizer"

// Create a new BPE tokenizer
bpe := tokenizer.NewBPE()

// Train on a corpus
corpus := "The quick brown fox jumps over the lazy dog. The dog ran away."
vocabSize := 1000  // Desired vocabulary size (256 base + 744 merges)

err := bpe.Train(corpus, vocabSize)
if err != nil {
    log.Fatalf("Training failed: %v", err)
}

fmt.Printf("Vocabulary size: %d\n", bpe.GetVocabSize())
```

### Encoding Text

```go
text := "Hello, world!"
tokenIDs := bpe.Encode(text)
fmt.Printf("Text: %q\nTokens: %v\n", text, tokenIDs)
// Output: Text: "Hello, world!"
//         Tokens: [72 101 108 108 111 44 32 119 111 114 108 100 33]
```

### Decoding Text

```go
tokenIDs := []int{72, 101, 108, 108, 111}
decoded := bpe.Decode(tokenIDs)
fmt.Printf("Decoded: %q\n", decoded)
// Output: Decoded: "hello"
```

### Round-Trip (Encode → Decode)

```go
original := "Hello, world!"
encoded := bpe.Encode(original)
decoded := bpe.Decode(encoded)
assert(original == decoded)  // Always true for BPE
```

## Example: Training on a Small Corpus

```go
package main

import (
    "fmt"
    "log"
    "github.com/tektwister/ai_engineering/tokenizer"
)

func main() {
    // Create tokenizer
    bpe := tokenizer.NewBPE()

    // Small corpus
    corpus := "hello hello hello world world, Hey what's your name? 1234"

    // Train with vocab size 260 (256 bytes + 4 merges)
    targetSize := 260
    if err := bpe.Train(corpus, targetSize); err != nil {
        log.Fatalf("Training failed: %v", err)
    }

    // Check vocabulary size
    fmt.Printf("Trained vocab size: %d\n", bpe.GetVocabSize())  // 260

    // Encode the corpus
    tokens := bpe.Encode(corpus)
    fmt.Printf("Encoded tokens: %v\n", tokens)
    fmt.Printf("Number of tokens: %d\n", len(tokens))

    // Decode back
    decoded := bpe.Decode(tokens)
    fmt.Printf("Decoded text: %q\n", decoded)
    fmt.Printf("Match: %v\n", decoded == corpus)  // true
}
```

## Implementation Details

### Training Algorithm

1. Initialize tokens as bytes (0-255) for the entire corpus.
2. Repeat until vocab size reached:
   - **Find Pairs**: Count all adjacent token pairs in the corpus.
   - **Select Best**: Choose the pair with the highest frequency.
   - **Merge**: Create a new token ID for this pair and update all occurrences.

**Time Complexity**: O(N × M), where N is corpus length and M is merges to learn.
**Space Complexity**: O(V), where V is vocabulary size.

### Encoding Algorithm

1. Convert text to bytes.
2. Iteratively apply learned merges in order (prioritized by their merge order—earlier merges have lower token IDs).
3. Stop when no more merges apply.

### Decoding Algorithm

For each token ID, look up its string representation and concatenate. Since BPE starts with bytes and builds up, all tokens can always be decoded.

## Testing

Run the provided test suite:

```bash
cd tokenizer
go test -v
```

**Test Coverage**:
- Training on small corpus and reaching target vocabulary size.
- Encode → Decode round-trip (lossless compression).
- Encoding of unseen (but similar) text.

Example test output:
```
=== RUN   TestBPE
    bpe_test.go:18: Ids: [104 101 108 108 111 32 104 101 108 108 111 32 104 101 108 108 111 32 119 111 114 108 100 32 119 111 114 108 100 44 32 72 101 121 32 119 104 97 116 39 115 32 121 111 117 114 32 110 97 109 101 63 32 49 50 51 52]
--- PASS: TestBPE (0.01s)
```

## Comparison with Other Tokenizers

| Tokenizer | Level | Vocab Size | Example |
|-----------|-------|-----------|---------|
| Character | Character | ~100 | "hello" → ['h','e','l','l','o'] |
| Word | Word | ~100k | "hello" → ['hello'] |
| **BPE** | **Subword** | **~50k-100k** | **"hello" → ['he','llo']** |
| WordPiece | Subword | ~30k | "hello" → ['hello'] |
| SentencePiece | Subword | ~32k | "hello" → ['▁he','llo'] |

## Enhancements & Future Work

- **Vocabulary Pruning**: Remove low-frequency tokens.
- **Custom Pre-tokenization**: Split on whitespace/punctuation before byte encoding.
- **Merge Order Serialization**: Save/load trained tokenizer state.
- **Byte Fallback**: Handle bytes not in vocabulary (for robustness).
- **Performance**: Cache frequent pairs during training.
- **Unicode Support**: Extend beyond ASCII/UTF-8 bytes.

## References

- **BPE Paper**: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich, Haddow, Birch, 2016)
- **GPT-2 Tokenizer**: OpenAI's implementation and vocabulary.
- **SentencePiece**: Google's language-agnostic tokenizer (alternative approach).

## Running the Module

### As a Standalone Program

Extend `cmd/main.go` to demonstrate training and usage:

```bash
cd tokenizer
go run cmd/main.go  # (if main.go exists)
```

### In Your Project

```go
import "github.com/tektwister/ai_engineering/tokenizer"

bpe := tokenizer.NewBPE()
bpe.Train(corpus, 50000)
tokens := bpe.Encode("Your text here")
```

## Module in the AI Engineering Roadmap

The tokenizer module is part of **Phase 1: Fundamentals** and prepares you for:
- **Next**: Transformer from scratch — understands how text is fed into neural networks.
- **Later**: Small Language Model training — uses this tokenizer for preprocessing.

## License

See the root repository LICENSE file.
