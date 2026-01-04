package transformer

import (
	"math"
)

// PositionalEncoding adds positional information to embeddings using sinusoidal functions.
// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
type PositionalEncoding struct {
	MaxLen   int
	DModel   int
	Encoding *Tensor // Precomputed positional encodings (maxLen, dModel)
	Dropout  *Dropout
}

// NewPositionalEncoding creates a new PositionalEncoding layer.
func NewPositionalEncoding(dModel, maxLen int, dropout float64) *PositionalEncoding {
	pe := NewTensor(maxLen, dModel)

	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < dModel; i += 2 {
			angle := float64(pos) / math.Pow(10000, float64(i)/float64(dModel))
			pe.Set(math.Sin(angle), pos, i)
			if i+1 < dModel {
				pe.Set(math.Cos(angle), pos, i+1)
			}
		}
	}

	return &PositionalEncoding{
		MaxLen:   maxLen,
		DModel:   dModel,
		Encoding: pe,
		Dropout:  NewDropout(dropout),
	}
}

// Forward adds positional encoding to the input embeddings.
// Input: (seqLen, dModel) or (batch, seqLen, dModel)
// Output: same shape as input
func (pe *PositionalEncoding) Forward(x *Tensor) *Tensor {
	if len(x.Shape) == 2 {
		seqLen := x.Shape[0]
		result := x.Clone()
		for i := 0; i < seqLen; i++ {
			for j := 0; j < pe.DModel; j++ {
				result.Set(result.At(i, j)+pe.Encoding.At(i, j), i, j)
			}
		}
		return pe.Dropout.Forward(result)
	} else if len(x.Shape) == 3 {
		batch, seqLen := x.Shape[0], x.Shape[1]
		result := x.Clone()
		for b := 0; b < batch; b++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < pe.DModel; j++ {
					result.Set(result.At(b, i, j)+pe.Encoding.At(i, j), b, i, j)
				}
			}
		}
		return pe.Dropout.Forward(result)
	}
	panic("PositionalEncoding.Forward requires 2D or 3D input")
}

// EncoderLayer implements a single Transformer encoder layer.
type EncoderLayer struct {
	SelfAttn    *MultiHeadAttention
	FeedForward *FeedForward
	Norm1       *LayerNorm
	Norm2       *LayerNorm
	Dropout1    *Dropout
	Dropout2    *Dropout
}

// NewEncoderLayer creates a new EncoderLayer.
func NewEncoderLayer(dModel, numHeads, dFF int, dropout float64) *EncoderLayer {
	return &EncoderLayer{
		SelfAttn:    NewMultiHeadAttention(dModel, numHeads),
		FeedForward: NewFeedForward(dModel, dFF, dropout, false),
		Norm1:       NewLayerNorm(dModel),
		Norm2:       NewLayerNorm(dModel),
		Dropout1:    NewDropout(dropout),
		Dropout2:    NewDropout(dropout),
	}
}

// Forward processes input through the encoder layer.
// x: (batch, seqLen, dModel)
// srcMask: optional mask
func (el *EncoderLayer) Forward(x *Tensor, srcMask *Tensor) *Tensor {
	// Self-attention with residual connection and layer norm
	attnOutput := el.SelfAttn.Forward(x, x, x, srcMask)
	attnOutput = el.Dropout1.Forward(attnOutput)
	x = el.Norm1.Forward(x.Add(attnOutput))

	// Feed-forward with residual connection and layer norm
	ffOutput := el.FeedForward.Forward(x)
	ffOutput = el.Dropout2.Forward(ffOutput)
	x = el.Norm2.Forward(x.Add(ffOutput))

	return x
}

// DecoderLayer implements a single Transformer decoder layer.
type DecoderLayer struct {
	SelfAttn    *MultiHeadAttention
	CrossAttn   *MultiHeadAttention
	FeedForward *FeedForward
	Norm1       *LayerNorm
	Norm2       *LayerNorm
	Norm3       *LayerNorm
	Dropout1    *Dropout
	Dropout2    *Dropout
	Dropout3    *Dropout
}

// NewDecoderLayer creates a new DecoderLayer.
func NewDecoderLayer(dModel, numHeads, dFF int, dropout float64) *DecoderLayer {
	return &DecoderLayer{
		SelfAttn:    NewMultiHeadAttention(dModel, numHeads),
		CrossAttn:   NewMultiHeadAttention(dModel, numHeads),
		FeedForward: NewFeedForward(dModel, dFF, dropout, false),
		Norm1:       NewLayerNorm(dModel),
		Norm2:       NewLayerNorm(dModel),
		Norm3:       NewLayerNorm(dModel),
		Dropout1:    NewDropout(dropout),
		Dropout2:    NewDropout(dropout),
		Dropout3:    NewDropout(dropout),
	}
}

// Forward processes input through the decoder layer.
// x: (batch, seqLen, dModel) - decoder input
// encOutput: (batch, srcLen, dModel) - encoder output
// tgtMask: causal mask for decoder self-attention
// srcMask: mask for cross-attention
func (dl *DecoderLayer) Forward(x, encOutput *Tensor, tgtMask, srcMask *Tensor) *Tensor {
	// Masked self-attention with residual and layer norm
	selfAttnOutput := dl.SelfAttn.Forward(x, x, x, tgtMask)
	selfAttnOutput = dl.Dropout1.Forward(selfAttnOutput)
	x = dl.Norm1.Forward(x.Add(selfAttnOutput))

	// Cross-attention with residual and layer norm
	crossAttnOutput := dl.CrossAttn.Forward(x, encOutput, encOutput, srcMask)
	crossAttnOutput = dl.Dropout2.Forward(crossAttnOutput)
	x = dl.Norm2.Forward(x.Add(crossAttnOutput))

	// Feed-forward with residual and layer norm
	ffOutput := dl.FeedForward.Forward(x)
	ffOutput = dl.Dropout3.Forward(ffOutput)
	x = dl.Norm3.Forward(x.Add(ffOutput))

	return x
}

// Encoder implements the Transformer encoder (stack of encoder layers).
type Encoder struct {
	Layers []*EncoderLayer
}

// NewEncoder creates a new Encoder with numLayers.
func NewEncoder(numLayers, dModel, numHeads, dFF int, dropout float64) *Encoder {
	layers := make([]*EncoderLayer, numLayers)
	for i := range layers {
		layers[i] = NewEncoderLayer(dModel, numHeads, dFF, dropout)
	}
	return &Encoder{Layers: layers}
}

// Forward processes input through all encoder layers.
func (e *Encoder) Forward(x *Tensor, srcMask *Tensor) *Tensor {
	for _, layer := range e.Layers {
		x = layer.Forward(x, srcMask)
	}
	return x
}

// Decoder implements the Transformer decoder (stack of decoder layers).
type Decoder struct {
	Layers []*DecoderLayer
}

// NewDecoder creates a new Decoder with numLayers.
func NewDecoder(numLayers, dModel, numHeads, dFF int, dropout float64) *Decoder {
	layers := make([]*DecoderLayer, numLayers)
	for i := range layers {
		layers[i] = NewDecoderLayer(dModel, numHeads, dFF, dropout)
	}
	return &Decoder{Layers: layers}
}

// Forward processes input through all decoder layers.
func (d *Decoder) Forward(x, encOutput *Tensor, tgtMask, srcMask *Tensor) *Tensor {
	for _, layer := range d.Layers {
		x = layer.Forward(x, encOutput, tgtMask, srcMask)
	}
	return x
}
