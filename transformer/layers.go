package transformer

import (
	"math"
)

// Linear represents a fully connected layer: y = xW^T + b
type Linear struct {
	Weight  *Tensor // (outFeatures, inFeatures)
	Bias    *Tensor // (outFeatures)
	InFeat  int
	OutFeat int
}

// NewLinear creates a new Linear layer with Xavier initialization.
func NewLinear(inFeatures, outFeatures int) *Linear {
	return NewLinearWithBias(inFeatures, outFeatures, true)
}

// NewLinearWithBias creates a new Linear layer with optional bias.
func NewLinearWithBias(inFeatures, outFeatures int, bias bool) *Linear {
	// Xavier initialization: scale by sqrt(2 / (in + out))
	scale := math.Sqrt(2.0 / float64(inFeatures+outFeatures))
	var biasTensor *Tensor
	if bias {
		biasTensor = Zeros(outFeatures)
	}
	return &Linear{
		Weight:  RandnScaled(scale, outFeatures, inFeatures),
		Bias:    biasTensor,
		InFeat:  inFeatures,
		OutFeat: outFeatures,
	}
}

// Forward applies the linear transformation.
// Input: (seqLen, inFeatures) or (batch, seqLen, inFeatures)
// Output: (seqLen, outFeatures) or (batch, seqLen, outFeatures)
func (l *Linear) Forward(x *Tensor) *Tensor {
	if len(x.Shape) == 2 {
		// x: (seqLen, inFeatures), W: (outFeatures, inFeatures)
		// y = x @ W^T + b
		wT := l.Weight.Transpose()
		result := x.MatMul(wT)
		// Add bias (broadcast) if bias exists
		if l.Bias != nil {
			for i := 0; i < result.Shape[0]; i++ {
				for j := 0; j < result.Shape[1]; j++ {
					result.Set(result.At(i, j)+l.Bias.Data[j], i, j)
				}
			}
		}
		return result
	} else if len(x.Shape) == 3 {
		// x: (batch, seqLen, inFeatures)
		batch, seqLen := x.Shape[0], x.Shape[1]
		result := NewTensor(batch, seqLen, l.OutFeat)
		wT := l.Weight.Transpose()

		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				for o := 0; o < l.OutFeat; o++ {
					sum := 0.0
					if l.Bias != nil {
						sum = l.Bias.Data[o]
					}
					for i := 0; i < l.InFeat; i++ {
						sum += x.At(b, s, i) * wT.At(i, o)
					}
					result.Set(sum, b, s, o)
				}
			}
		}
		return result
	}
	panic("Linear.Forward requires 2D or 3D input")
}

// Embedding represents an embedding layer that maps token IDs to vectors.
type Embedding struct {
	Weight    *Tensor // (vocabSize, embedDim)
	VocabSize int
	EmbedDim  int
}

// NewEmbedding creates a new Embedding layer.
func NewEmbedding(vocabSize, embedDim int) *Embedding {
	scale := math.Sqrt(1.0 / float64(embedDim))
	return &Embedding{
		Weight:    RandnScaled(scale, vocabSize, embedDim),
		VocabSize: vocabSize,
		EmbedDim:  embedDim,
	}
}

// Forward maps token IDs to embeddings.
// Input: []int of length seqLen
// Output: (seqLen, embedDim)
func (e *Embedding) Forward(tokenIDs []int) *Tensor {
	seqLen := len(tokenIDs)
	result := NewTensor(seqLen, e.EmbedDim)
	for i, tokID := range tokenIDs {
		for j := 0; j < e.EmbedDim; j++ {
			result.Set(e.Weight.At(tokID, j), i, j)
		}
	}
	return result
}

// ForwardBatch maps batched token IDs to embeddings.
// Input: [][]int of shape (batch, seqLen)
// Output: (batch, seqLen, embedDim)
func (e *Embedding) ForwardBatch(tokenIDs [][]int) *Tensor {
	batch := len(tokenIDs)
	seqLen := len(tokenIDs[0])
	result := NewTensor(batch, seqLen, e.EmbedDim)
	for b := 0; b < batch; b++ {
		for i, tokID := range tokenIDs[b] {
			for j := 0; j < e.EmbedDim; j++ {
				result.Set(e.Weight.At(tokID, j), b, i, j)
			}
		}
	}
	return result
}

// LayerNorm implements Layer Normalization.
type LayerNorm struct {
	Gamma   *Tensor // scale parameter (normalized_shape)
	Beta    *Tensor // shift parameter (normalized_shape)
	Epsilon float64
	NormDim int
}

// NewLayerNorm creates a new LayerNorm layer.
func NewLayerNorm(normalizedShape int) *LayerNorm {
	return &LayerNorm{
		Gamma:   Ones(normalizedShape),
		Beta:    Zeros(normalizedShape),
		Epsilon: 1e-5,
		NormDim: normalizedShape,
	}
}

// Forward applies layer normalization.
// Input: (seqLen, dim) or (batch, seqLen, dim)
// Normalizes over the last dimension.
func (ln *LayerNorm) Forward(x *Tensor) *Tensor {
	result := x.Clone()

	if len(x.Shape) == 2 {
		// (seqLen, dim)
		seqLen, dim := x.Shape[0], x.Shape[1]
		for i := 0; i < seqLen; i++ {
			// Compute mean and variance for this position
			mean := 0.0
			for j := 0; j < dim; j++ {
				mean += x.At(i, j)
			}
			mean /= float64(dim)

			variance := 0.0
			for j := 0; j < dim; j++ {
				diff := x.At(i, j) - mean
				variance += diff * diff
			}
			variance /= float64(dim)

			// Normalize
			for j := 0; j < dim; j++ {
				normalized := (x.At(i, j) - mean) / math.Sqrt(variance+ln.Epsilon)
				result.Set(normalized*ln.Gamma.Data[j]+ln.Beta.Data[j], i, j)
			}
		}
	} else if len(x.Shape) == 3 {
		// (batch, seqLen, dim)
		batch, seqLen, dim := x.Shape[0], x.Shape[1], x.Shape[2]
		for b := 0; b < batch; b++ {
			for i := 0; i < seqLen; i++ {
				// Compute mean and variance for this position
				mean := 0.0
				for j := 0; j < dim; j++ {
					mean += x.At(b, i, j)
				}
				mean /= float64(dim)

				variance := 0.0
				for j := 0; j < dim; j++ {
					diff := x.At(b, i, j) - mean
					variance += diff * diff
				}
				variance /= float64(dim)

				// Normalize
				for j := 0; j < dim; j++ {
					normalized := (x.At(b, i, j) - mean) / math.Sqrt(variance+ln.Epsilon)
					result.Set(normalized*ln.Gamma.Data[j]+ln.Beta.Data[j], b, i, j)
				}
			}
		}
	} else {
		panic("LayerNorm.Forward requires 2D or 3D input")
	}

	return result
}

// Dropout applies dropout during training (for now, just identity since we don't track training mode).
type Dropout struct {
	P float64 // dropout probability
}

// NewDropout creates a new Dropout layer.
func NewDropout(p float64) *Dropout {
	return &Dropout{P: p}
}

// Forward applies dropout (currently identity for inference).
func (d *Dropout) Forward(x *Tensor) *Tensor {
	// For simplicity, dropout is disabled (inference mode)
	return x
}

// TransformerBlock implements a single transformer block with self-attention and feed-forward.
type TransformerBlock struct {
	Attn    *MultiHeadAttention
	LN1     *LayerNorm
	LN2     *LayerNorm
	FFN     *FeedForward
	Dropout *Dropout
}

// FeedForward implements the feed-forward network in transformer blocks.
type FeedForward struct {
	Linear1 *Linear
	Linear2 *Linear
	Dropout *Dropout
	UseGELU bool
}

// NewFeedForward creates a new feed-forward network.
// dModel: input/output dimension
// dFF: hidden layer dimension
// dropout: dropout probability
// useGELU: if true use GELU activation, else use ReLU
func NewFeedForward(dModel, dFF int, dropout float64, useGELU bool) *FeedForward {
	return &FeedForward{
		Linear1: NewLinear(dModel, dFF),
		Linear2: NewLinear(dFF, dModel),
		Dropout: NewDropout(dropout),
		UseGELU: useGELU,
	}
}

// Forward applies the feed-forward network.
// x -> Linear1 -> Activation -> Dropout -> Linear2
func (ff *FeedForward) Forward(x *Tensor) *Tensor {
	hidden := ff.Linear1.Forward(x)
	if ff.UseGELU {
		hidden = hidden.GELU()
	} else {
		hidden = hidden.ReLU()
	}
	hidden = ff.Dropout.Forward(hidden)
	return ff.Linear2.Forward(hidden)
}

// NewTransformerBlock creates a new transformer block.
func NewTransformerBlock(dModel, numHeads, dFF int, dropout float64, useGELU bool) *TransformerBlock {
	return &TransformerBlock{
		Attn:    NewMultiHeadAttention(dModel, numHeads),
		LN1:     NewLayerNorm(dModel),
		LN2:     NewLayerNorm(dModel),
		FFN:     NewFeedForward(dModel, dFF, dropout, useGELU),
		Dropout: NewDropout(dropout),
	}
}

// Forward applies the transformer block.
// x: (batch, seqLen, dModel)
// mask: optional attention mask
// Returns: (batch, seqLen, dModel)
func (tb *TransformerBlock) Forward(x *Tensor, mask *Tensor) *Tensor {
	// Self-attention with residual connection and layer norm
	attnOut := tb.Attn.Forward(x, x, x, mask)
	attnOut = tb.Dropout.Forward(attnOut)
	x = x.Add(attnOut) // residual
	x = tb.LN1.Forward(x)

	// Feed-forward with residual connection and layer norm
	ffnOut := tb.FFN.Forward(x)
	ffnOut = tb.Dropout.Forward(ffnOut)
	x = x.Add(ffnOut) // residual
	x = tb.LN2.Forward(x)

	return x
}
