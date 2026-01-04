package transformer

import (
	"math"
)

// Transformer implements the full Transformer model from "Attention is All You Need".
type Transformer struct {
	// Embeddings
	SrcEmbedding *Embedding
	TgtEmbedding *Embedding

	// Positional Encoding
	PosEncoder *PositionalEncoding

	// Encoder and Decoder
	Encoder *Encoder
	Decoder *Decoder

	// Output projection
	OutputLinear *Linear

	// Model config
	Config *TransformerConfig
}

// TransformerConfig holds the configuration for the Transformer model.
type TransformerConfig struct {
	SrcVocabSize int     // Source vocabulary size
	TgtVocabSize int     // Target vocabulary size
	DModel       int     // Model dimension (embedding dimension)
	NumHeads     int     // Number of attention heads
	NumEncoderLayers int // Number of encoder layers
	NumDecoderLayers int // Number of decoder layers
	DFF          int     // Feed-forward hidden dimension
	MaxSeqLen    int     // Maximum sequence length
	Dropout      float64 // Dropout rate
}

// DefaultConfig returns a default Transformer configuration.
func DefaultConfig() *TransformerConfig {
	return &TransformerConfig{
		SrcVocabSize:     10000,
		TgtVocabSize:     10000,
		DModel:           512,
		NumHeads:         8,
		NumEncoderLayers: 6,
		NumDecoderLayers: 6,
		DFF:              2048,
		MaxSeqLen:        512,
		Dropout:          0.1,
	}
}

// SmallConfig returns a smaller Transformer configuration for testing.
func SmallConfig() *TransformerConfig {
	return &TransformerConfig{
		SrcVocabSize:     1000,
		TgtVocabSize:     1000,
		DModel:           64,
		NumHeads:         4,
		NumEncoderLayers: 2,
		NumDecoderLayers: 2,
		DFF:              256,
		MaxSeqLen:        128,
		Dropout:          0.1,
	}
}

// NewTransformer creates a new Transformer model with the given configuration.
func NewTransformer(config *TransformerConfig) *Transformer {
	return &Transformer{
		SrcEmbedding: NewEmbedding(config.SrcVocabSize, config.DModel),
		TgtEmbedding: NewEmbedding(config.TgtVocabSize, config.DModel),
		PosEncoder:   NewPositionalEncoding(config.DModel, config.MaxSeqLen, config.Dropout),
		Encoder:      NewEncoder(config.NumEncoderLayers, config.DModel, config.NumHeads, config.DFF, config.Dropout),
		Decoder:      NewDecoder(config.NumDecoderLayers, config.DModel, config.NumHeads, config.DFF, config.Dropout),
		OutputLinear: NewLinear(config.DModel, config.TgtVocabSize),
		Config:       config,
	}
}

// Encode encodes the source sequence.
// srcTokens: [][]int of shape (batch, srcLen)
// Returns: (batch, srcLen, dModel)
func (t *Transformer) Encode(srcTokens [][]int, srcMask *Tensor) *Tensor {
	// Embed source tokens
	srcEmbed := t.SrcEmbedding.ForwardBatch(srcTokens)

	// Scale embeddings
	srcEmbed = srcEmbed.MulScalar(math.Sqrt(float64(t.Config.DModel)))

	// Add positional encoding
	srcEmbed = t.PosEncoder.Forward(srcEmbed)

	// Pass through encoder
	return t.Encoder.Forward(srcEmbed, srcMask)
}

// Decode decodes the target sequence given encoder output.
// tgtTokens: [][]int of shape (batch, tgtLen)
// encOutput: (batch, srcLen, dModel)
// Returns: (batch, tgtLen, tgtVocabSize) - logits
func (t *Transformer) Decode(tgtTokens [][]int, encOutput *Tensor, tgtMask, srcMask *Tensor) *Tensor {
	// Embed target tokens
	tgtEmbed := t.TgtEmbedding.ForwardBatch(tgtTokens)

	// Scale embeddings
	tgtEmbed = tgtEmbed.MulScalar(math.Sqrt(float64(t.Config.DModel)))

	// Add positional encoding
	tgtEmbed = t.PosEncoder.Forward(tgtEmbed)

	// Pass through decoder (mask handling is done inside attention layers)
	decOutput := t.Decoder.Forward(tgtEmbed, encOutput, tgtMask, srcMask)

	// Project to vocabulary
	return t.OutputLinear.Forward(decOutput)
}

// Forward performs a full forward pass (encode + decode).
// Returns logits: (batch, tgtLen, tgtVocabSize)
func (t *Transformer) Forward(srcTokens, tgtTokens [][]int, srcMask, tgtMask *Tensor) *Tensor {
	encOutput := t.Encode(srcTokens, srcMask)
	return t.Decode(tgtTokens, encOutput, tgtMask, srcMask)
}

// Generate performs autoregressive generation (greedy decoding).
// srcTokens: source sequence
// maxLen: maximum length to generate
// startToken: token to start generation with
// endToken: token that signals end of generation
func (t *Transformer) Generate(srcTokens [][]int, maxLen, startToken, endToken int) [][]int {
	batch := len(srcTokens)

	// Encode source
	encOutput := t.Encode(srcTokens, nil)

	// Initialize target with start token
	generated := make([][]int, batch)
	for b := range generated {
		generated[b] = []int{startToken}
	}

	for step := 0; step < maxLen; step++ {
		// Decode current sequence
		logits := t.Decode(generated, encOutput, nil, nil)

		// Get predictions for the last position
		// logits: (batch, currLen, vocabSize)
		currLen := len(generated[0])
		
		for b := 0; b < batch; b++ {
			// Find argmax for last position
			maxIdx := 0
			maxVal := math.Inf(-1)
			for v := 0; v < t.Config.TgtVocabSize; v++ {
				val := logits.At(b, currLen-1, v)
				if val > maxVal {
					maxVal = val
					maxIdx = v
				}
			}

			// Append predicted token
			generated[b] = append(generated[b], maxIdx)
		}

		// Check if all sequences have ended
		allEnded := true
		for b := 0; b < batch; b++ {
			if generated[b][len(generated[b])-1] != endToken {
				allEnded = false
				break
			}
		}
		if allEnded {
			break
		}
	}

	return generated
}

// Softmax converts logits to probabilities.
func Softmax(logits *Tensor) *Tensor {
	return logits.Softmax()
}

// CrossEntropyLoss computes cross-entropy loss.
// logits: (batch, seqLen, vocabSize)
// targets: [][]int of shape (batch, seqLen)
// Returns: scalar loss value
func CrossEntropyLoss(logits *Tensor, targets [][]int) float64 {
	batch, seqLen, vocabSize := logits.Shape[0], logits.Shape[1], logits.Shape[2]
	
	totalLoss := 0.0
	count := 0

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			// Get logits for this position
			maxLogit := math.Inf(-1)
			for v := 0; v < vocabSize; v++ {
				if logits.At(b, s, v) > maxLogit {
					maxLogit = logits.At(b, s, v)
				}
			}

			// Compute log-softmax
			sumExp := 0.0
			for v := 0; v < vocabSize; v++ {
				sumExp += math.Exp(logits.At(b, s, v) - maxLogit)
			}
			logSumExp := maxLogit + math.Log(sumExp)

			// Cross-entropy: -log(softmax[target])
			targetIdx := targets[b][s]
			logProb := logits.At(b, s, targetIdx) - logSumExp
			totalLoss -= logProb
			count++
		}
	}

	return totalLoss / float64(count)
}
