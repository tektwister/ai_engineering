package tokenizer

import (
	"fmt"
	"strings"
)

// Tokenizer defines the interface for a tokenizer
type Tokenizer interface {
	Encode(text string) []int
	Decode(ids []int) string
	Train(text string, vocabSize int) error
}

// BPE implements Byte Pair Encoding tokenizer
type BPE struct {
	vocab     map[string]int
	vocabInv  map[int]string
	merges    map[string]int // pair "t1 t2" -> rank/order
	mergesInv map[int]string
}

// NewBPE creates a new BPE tokenizer
func NewBPE() *BPE {
	return &BPE{
		vocab:     make(map[string]int),
		vocabInv:  make(map[int]string),
		merges:    make(map[string]int),
		mergesInv: make(map[int]string),
	}
}

// Train learns BPE merges from the provided text until vocabSize is reached
func (b *BPE) Train(text string, vocabSize int) error {
	// 1. Initialize vocabulary with all unique characters (bytes for simplicity in BPE usually, but here likely runes or bytes)
	// For standard BPE, we often start with bytes. Let's use characters (runes) for a more text-focused approach, 
	// or bytes to be properly robust like GPT-2. Let's stick to byte-level BPE for robustness.
	
	// Convert text to bytes
	data := []byte(text)
	
	// Initial tokens are just the bytes
	// We need to represent the current state of the data as a list of integers (token IDs)
	// Since we are doing byte-level BPE, the initial vocab is 0-255.
	
	ids := make([]int, len(data))
	for i, b := range data {
		ids[i] = int(b)
	}

	// Initialize vocab with 0-255
	for i := 0; i < 256; i++ {
		b.vocab[string(byte(i))] = i
		b.vocabInv[i] = string(byte(i))
	}

	nextID := 256

	for len(b.vocab) < vocabSize {
		// Find most frequent pair
		pairs := make(map[string]int)
		for i := 0; i < len(ids)-1; i++ {
			pair := fmt.Sprintf("%d,%d", ids[i], ids[i+1])
			pairs[pair]++
		}

		if len(pairs) == 0 {
			break
		}

		bestPair := ""
		maxCount := -1
		for p, count := range pairs {
			if count > maxCount {
				maxCount = count
				bestPair = p
			}
		}

		// Merge the best pair
		// Create new token
		var part1, part2 int
		fmt.Sscanf(bestPair, "%d,%d", &part1, &part2)
		
		 newTokenStr := b.vocabInv[part1] + b.vocabInv[part2]
		 b.vocab[newTokenStr] = nextID
		 b.vocabInv[nextID] = newTokenStr
		 b.merges[bestPair] = nextID
		 
		 // Update ids
		 newIds := make([]int, 0, len(ids))
		 for i := 0; i < len(ids); {
			 if i < len(ids)-1 && ids[i] == part1 && ids[i+1] == part2 {
				 newIds = append(newIds, nextID)
				 i += 2
			 } else {
				 newIds = append(newIds, ids[i])
				 i++
			 }
		 }
		 ids = newIds
		 nextID++
	}

	return nil
}

// Encode converts text to token IDs
func (b *BPE) Encode(text string) []int {
	// Start with bytes
	data := []byte(text)
	ids := make([]int, len(data))
	for i, v := range data {
		ids[i] = int(v)
	}

	// Apply merges
	// We need to apply merges in the order they were learned (or just verify presence)
	// Actually, optimization: iterate through merges until no changes?
	// Or store merges by priority.
	
	// Since we didn't store simple priority list in `merges` map (we stored rank as value effectively, or we should),
	// we should iterate. But standard BPE inference is:
	// Find any pair in current `ids` that exists in `merges`.
	// If multiple, pick the one with lowest rank (earliest merge).
	// But our `merges` map value IS the new ID which acts as rank if we trained sequentially.
	// Actually, we need to know the order of merges. 
	// The `nextID` strictly increases, so smaller `nextID` means earlier merge.
	
	for {
		minRank := 1000000000 // Arbitrary large number
		var bestPairPart1, bestPairPart2 int
		found := false

		// Naive scan for the best next merge
		for i := 0; i < len(ids)-1; i++ {
			pair := fmt.Sprintf("%d,%d", ids[i], ids[i+1])
			if id, ok := b.merges[pair]; ok {
				if id < minRank {
					minRank = id
					bestPairPart1 = ids[i]
					bestPairPart2 = ids[i+1]
					found = true
				}
			}
		}

		if !found {
			break
		}

		// Apply the merge
		newIds := make([]int, 0, len(ids))
		for i := 0; i < len(ids); {
			if i < len(ids)-1 && ids[i] == bestPairPart1 && ids[i+1] == bestPairPart2 {
				newIds = append(newIds, minRank)
				i += 2
			} else {
				newIds = append(newIds, ids[i])
				i++
			}
		}
		ids = newIds
	}

	return ids
}

// Decode converts token IDs back to text
func (b *BPE) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if s, ok := b.vocabInv[id]; ok {
			sb.WriteString(s)
		} else {
			// fallback/unknown, though BPE shouldn't have unknowns if initialized with bytes
			sb.WriteByte(0) // or replace
		}
	}
	return sb.String()
}

// GetVocabSize returns the current size of the vocabulary
func (b *BPE) GetVocabSize() int {
	return len(b.vocab)
}
