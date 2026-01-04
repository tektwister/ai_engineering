package small_language_model

import (
	"math"
	"math/rand"

	"github.com/tektwister/ai_engineering/transformer"
)

// GPTConfig holds configuration for the GPT model
type GPTConfig struct {
	VocabSize int     // Vocabulary size
	BlockSize int     // Maximum sequence length
	NLayer    int     // Number of transformer blocks
	NHead     int     // Number of attention heads
	NEmbd     int     // Embedding dimension
	Dropout   float64 // Dropout rate
	Bias      bool    // Whether to use bias in linear layers
}

// DefaultGPTConfig returns a default GPT configuration
func DefaultGPTConfig() *GPTConfig {
	return &GPTConfig{
		VocabSize: 50257, // GPT-2 vocabulary size
		BlockSize: 1024,
		NLayer:    12,
		NHead:     12,
		NEmbd:     768,
		Dropout:   0.0,
		Bias:      true,
	}
}

// SmallGPTConfig returns a small GPT configuration for testing
func SmallGPTConfig() *GPTConfig {
	return &GPTConfig{
		VocabSize: 1000,
		BlockSize: 128,
		NLayer:    4,
		NHead:     4,
		NEmbd:     128,
		Dropout:   0.1,
		Bias:      true,
	}
}

// GPT represents a decoder-only transformer model (GPT architecture)
type GPT struct {
	config   *GPTConfig
	tokenEmb *transformer.Embedding          // Token embeddings
	posEmb   *transformer.Embedding          // Position embeddings
	blocks   []*transformer.TransformerBlock // Transformer blocks
	lnF      *transformer.LayerNorm          // Final layer norm
	lmHead   *transformer.Linear             // Language model head
}

// NewGPT creates a new GPT model
func NewGPT(config *GPTConfig) *GPT {
	// Initialize embeddings
	tokenEmb := transformer.NewEmbedding(config.VocabSize, config.NEmbd)
	posEmb := transformer.NewEmbedding(config.BlockSize, config.NEmbd)

	// Initialize transformer blocks
	blocks := make([]*transformer.TransformerBlock, config.NLayer)
	for i := 0; i < config.NLayer; i++ {
		blocks[i] = transformer.NewTransformerBlock(config.NEmbd, config.NHead, config.NEmbd*4, config.Dropout, config.Bias)
	}

	// Final layer norm and language model head
	lnF := transformer.NewLayerNorm(config.NEmbd)
	lmHead := transformer.NewLinearWithBias(config.NEmbd, config.VocabSize, config.Bias)

	return &GPT{
		config:   config,
		tokenEmb: tokenEmb,
		posEmb:   posEmb,
		blocks:   blocks,
		lnF:      lnF,
		lmHead:   lmHead,
	}
}

// Forward performs a forward pass through the GPT model
// x: input token indices of shape (batch, seq_len)
// targets: optional target token indices for computing loss of shape (batch, seq_len)
// Returns: logits of shape (batch, seq_len, vocab_size) and optional loss
func (g *GPT) Forward(x [][]int, targets [][]int) (*transformer.Tensor, float64) {
	batchSize := len(x)
	seqLen := len(x[0])

	// Create token embeddings: (batch, seq_len, n_embd)
	tokenEmbeddings := g.tokenEmb.ForwardBatch(x)

	// Create position embeddings: (batch, seq_len, n_embd)
	posIndices := make([][]int, batchSize)
	for b := 0; b < batchSize; b++ {
		posIndices[b] = make([]int, seqLen)
		for t := 0; t < seqLen; t++ {
			posIndices[b][t] = t
		}
	}
	posEmbeddings := g.posEmb.ForwardBatch(posIndices)

	// Add token and position embeddings
	embeddings := tokenEmbeddings.Add(posEmbeddings)

	// Apply transformer blocks
	hidden := embeddings
	for _, block := range g.blocks {
		hidden = block.Forward(hidden, nil) // No mask for language modeling
	}

	// Apply final layer norm
	hidden = g.lnF.Forward(hidden)

	// Apply language model head
	logits := g.lmHead.Forward(hidden)

	// Compute loss if targets provided
	var loss float64
	if targets != nil {
		loss = g.computeLoss(logits, targets)
	}

	return logits, loss
}

// Generate generates new tokens autoregressively
// idx: input token indices of shape (batch, seq_len)
// maxNewTokens: maximum number of new tokens to generate
// temperature: sampling temperature (0.0 = greedy, 1.0 = normal, >1.0 = more random)
// topK: top-k sampling (0 = disabled)
// topP: nucleus sampling (1.0 = disabled)
// Returns: generated token indices including input
func (g *GPT) Generate(idx [][]int, maxNewTokens int, temperature, topP float64, topK int) [][]int {
	for _ = range maxNewTokens {
		// Crop idx to block_size if needed
		if len(idx[0]) > g.config.BlockSize {
			for b := range idx {
				idx[b] = idx[b][len(idx[b])-g.config.BlockSize:]
			}
		}

		// Forward pass to get logits
		logits, _ := g.Forward(idx, nil)

		// Get logits for the last position of each sequence in batch
		batchSize := len(idx)
		lastLogits := make([][]float32, batchSize)
		for b := 0; b < batchSize; b++ {
			seqLen := len(idx[b])
			lastLogits[b] = make([]float32, g.config.VocabSize)
			for v := 0; v < g.config.VocabSize; v++ {
				lastLogits[b][v] = float32(logits.At(b, seqLen-1, v))
			}
		}

		// Sample next tokens
		nextTokens := make([]int, batchSize)
		for b := 0; b < batchSize; b++ {
			nextTokens[b] = g.sampleToken(lastLogits[b], temperature, topP, topK)
		}

		// Append sampled tokens to sequences
		for b := 0; b < batchSize; b++ {
			idx[b] = append(idx[b], nextTokens[b])
		}
	}

	return idx
}

// sampleToken samples a token from logits using temperature, top-k, and top-p
func (g *GPT) sampleToken(logits []float32, temperature, topP float64, topK int) int {
	// Apply temperature
	if temperature != 1.0 {
		for i := range logits {
			logits[i] /= float32(temperature)
		}
	}

	// Apply top-k filtering
	if topK > 0 && topK < len(logits) {
		logits = g.applyTopK(logits, topK)
	}

	// Apply top-p (nucleus) filtering
	if topP < 1.0 {
		logits = g.applyTopP(logits, topP)
	}

	// Convert to probabilities
	probs := g.softmax(logits)

	// Sample from distribution
	r := rand.Float64()
	cumProb := 0.0
	for i, prob := range probs {
		cumProb += prob
		if r <= cumProb {
			return i
		}
	}

	// Fallback (should not happen)
	return len(logits) - 1
}

// applyTopK keeps only the top-k highest probability tokens
func (g *GPT) applyTopK(logits []float32, k int) []float32 {
	if k >= len(logits) {
		return logits
	}

	// Create index-value pairs
	type kv struct {
		idx int
		val float32
	}

	pairs := make([]kv, len(logits))
	for i, v := range logits {
		pairs[i] = kv{i, v}
	}

	// Sort by value (descending)
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].val < pairs[j].val {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Zero out everything except top-k
	result := make([]float32, len(logits))
	for i := 0; i < k; i++ {
		result[pairs[i].idx] = pairs[i].val
	}

	// Set others to -inf
	for i := range result {
		if result[i] == 0 {
			result[i] = float32(math.Inf(-1))
		}
	}

	return result
}

// applyTopP keeps only tokens that make up the top-p probability mass
func (g *GPT) applyTopP(logits []float32, p float64) []float32 {
	probs := g.softmax(logits)

	// Create index-probability pairs
	type kv struct {
		idx  int
		prob float64
	}

	pairs := make([]kv, len(probs))
	for i, prob := range probs {
		pairs[i] = kv{i, prob}
	}

	// Sort by probability (descending)
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].prob < pairs[j].prob {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Find cutoff point
	cumProb := 0.0
	cutoff := len(pairs)
	for i, pair := range pairs {
		cumProb += pair.prob
		if cumProb >= p {
			cutoff = i + 1
			break
		}
	}

	// Zero out everything except top-p tokens
	result := make([]float32, len(logits))
	for i := 0; i < cutoff; i++ {
		result[pairs[i].idx] = logits[pairs[i].idx]
	}

	// Set others to -inf
	for i := range result {
		if result[i] == 0 {
			result[i] = float32(math.Inf(-1))
		}
	}

	return result
}

// softmax computes softmax of logits
func (g *GPT) softmax(logits []float32) []float64 {
	// Find max for numerical stability
	maxVal := float64(logits[0])
	for _, v := range logits {
		if float64(v) > maxVal {
			maxVal = float64(v)
		}
	}

	// Compute exp and sum
	sum := 0.0
	expVals := make([]float64, len(logits))
	for i, v := range logits {
		if v == float32(math.Inf(-1)) {
			expVals[i] = 0.0
		} else {
			expVals[i] = math.Exp(float64(v) - maxVal)
			sum += expVals[i]
		}
	}

	// Normalize
	probs := make([]float64, len(logits))
	for i, v := range expVals {
		probs[i] = v / sum
	}

	return probs
}

// computeLoss computes cross-entropy loss
func (g *GPT) computeLoss(logits *transformer.Tensor, targets [][]int) float64 {
	batchSize := len(targets)
	seqLen := len(targets[0])

	totalLoss := 0.0
	count := 0

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			target := targets[b][t]

			// Skip padding tokens (assuming 0 is padding)
			if target == 0 {
				continue
			}

			// Get logits for this position
			logit := logits.At(b, t, target)

			// Compute log softmax
			maxLogit := math.Inf(-1)
			sumExp := 0.0

			for v := 0; v < g.config.VocabSize; v++ {
				l := logits.At(b, t, v)
				if l > maxLogit {
					maxLogit = l
				}
			}

			for v := 0; v < g.config.VocabSize; v++ {
				sumExp += math.Exp(logits.At(b, t, v) - maxLogit)
			}

			logProb := logit - maxLogit - math.Log(sumExp)
			totalLoss -= logProb
			count++
		}
	}

	if count == 0 {
		return 0.0
	}
	return totalLoss / float64(count)
}

// GetConfig returns the model configuration
func (g *GPT) GetConfig() *GPTConfig {
	return g.config
}
