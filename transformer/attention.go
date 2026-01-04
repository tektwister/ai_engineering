package transformer

import (
	"math"
)

// ScaledDotProductAttention computes attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
// Q: (seqLen, d_k) or (batch, seqLen, d_k)
// K: (seqLen, d_k) or (batch, seqLen, d_k)
// V: (seqLen, d_v) or (batch, seqLen, d_v)
// mask: optional attention mask
func ScaledDotProductAttention(Q, K, V *Tensor, mask *Tensor) *Tensor {
	if len(Q.Shape) == 2 {
		return scaledDotProductAttention2D(Q, K, V, mask)
	} else if len(Q.Shape) == 3 {
		return scaledDotProductAttention3D(Q, K, V, mask)
	}
	panic("ScaledDotProductAttention requires 2D or 3D tensors")
}

func scaledDotProductAttention2D(Q, K, V *Tensor, mask *Tensor) *Tensor {
	dK := float64(Q.Shape[1])
	scale := 1.0 / math.Sqrt(dK)

	// Q @ K^T: (seqLen, d_k) @ (d_k, seqLen) = (seqLen, seqLen)
	kT := K.Transpose()
	scores := Q.MatMul(kT).MulScalar(scale)

	// Apply mask if provided (set masked positions to -inf)
	if mask != nil {
		for i := 0; i < scores.Shape[0]; i++ {
			for j := 0; j < scores.Shape[1]; j++ {
				if mask.At(i, j) == 0 {
					scores.Set(math.Inf(-1), i, j)
				}
			}
		}
	}

	// Softmax
	attnWeights := scores.Softmax()

	// attnWeights @ V: (seqLen, seqLen) @ (seqLen, d_v) = (seqLen, d_v)
	return attnWeights.MatMul(V)
}

func scaledDotProductAttention3D(Q, K, V *Tensor, mask *Tensor) *Tensor {
	batch, seqLenQ, dK := Q.Shape[0], Q.Shape[1], Q.Shape[2]
	seqLenK := K.Shape[1]
	dV := V.Shape[2]
	scale := 1.0 / math.Sqrt(float64(dK))

	// Q @ K^T: (batch, seqLenQ, d_k) @ (batch, d_k, seqLenK) = (batch, seqLenQ, seqLenK)
	kT := K.TransposeLast2D()
	
	// Manual batch matmul for different sequence lengths
	scores := NewTensor(batch, seqLenQ, seqLenK)
	for b := 0; b < batch; b++ {
		for i := 0; i < seqLenQ; i++ {
			for j := 0; j < seqLenK; j++ {
				sum := 0.0
				for k := 0; k < dK; k++ {
					sum += Q.At(b, i, k) * kT.At(b, k, j)
				}
				scores.Set(sum*scale, b, i, j)
			}
		}
	}

	// Apply mask if provided
	if mask != nil {
		maskBatch := mask.Shape[0]
		maskSeqQ := mask.Shape[1]
		maskSeqK := mask.Shape[2]
		for b := 0; b < batch; b++ {
			// Handle mask broadcasting: if mask has fewer batches, wrap around
			mb := b % maskBatch
			for i := 0; i < seqLenQ; i++ {
				mi := i % maskSeqQ
				for j := 0; j < seqLenK; j++ {
					mj := j % maskSeqK
					if mask.At(mb, mi, mj) == 0 {
						scores.Set(math.Inf(-1), b, i, j)
					}
				}
			}
		}
	}

	// Softmax
	attnWeights := scores.Softmax()

	// attnWeights @ V: (batch, seqLenQ, seqLenK) @ (batch, seqLenK, d_v) = (batch, seqLenQ, d_v)
	result := NewTensor(batch, seqLenQ, dV)
	for b := 0; b < batch; b++ {
		for i := 0; i < seqLenQ; i++ {
			for j := 0; j < dV; j++ {
				sum := 0.0
				for k := 0; k < seqLenK; k++ {
					sum += attnWeights.At(b, i, k) * V.At(b, k, j)
				}
				result.Set(sum, b, i, j)
			}
		}
	}
	return result
}

// MultiHeadAttention implements multi-head attention mechanism.
type MultiHeadAttention struct {
	NumHeads int
	DModel   int
	DK       int
	DV       int

	WQ *Linear // Query projection
	WK *Linear // Key projection
	WV *Linear // Value projection
	WO *Linear // Output projection
}

// NewMultiHeadAttention creates a new MultiHeadAttention layer.
func NewMultiHeadAttention(dModel, numHeads int) *MultiHeadAttention {
	if dModel%numHeads != 0 {
		panic("dModel must be divisible by numHeads")
	}
	dK := dModel / numHeads
	dV := dModel / numHeads

	return &MultiHeadAttention{
		NumHeads: numHeads,
		DModel:   dModel,
		DK:       dK,
		DV:       dV,
		WQ:       NewLinear(dModel, dModel),
		WK:       NewLinear(dModel, dModel),
		WV:       NewLinear(dModel, dModel),
		WO:       NewLinear(dModel, dModel),
	}
}

// Forward computes multi-head attention.
// Query, Key, Value: (batch, seqLen, dModel)
// Returns: (batch, seqLen, dModel)
func (mha *MultiHeadAttention) Forward(query, key, value *Tensor, mask *Tensor) *Tensor {
	batch := query.Shape[0]
	seqLenQ := query.Shape[1]
	seqLenK := key.Shape[1]

	// Linear projections
	Q := mha.WQ.Forward(query) // (batch, seqLen, dModel)
	K := mha.WK.Forward(key)
	V := mha.WV.Forward(value)

	// Reshape for multi-head: (batch, seqLen, numHeads, dK) -> (batch, numHeads, seqLen, dK)
	Q = mha.splitHeads(Q, batch, seqLenQ)
	K = mha.splitHeads(K, batch, seqLenK)
	V = mha.splitHeads(V, batch, seqLenK)

	// Scaled dot-product attention for each head
	// Q, K, V are now (batch * numHeads, seqLen, dK/dV)
	attnOutput := ScaledDotProductAttention(Q, K, V, mask)

	// Concatenate heads: (batch * numHeads, seqLen, dV) -> (batch, seqLen, dModel)
	attnOutput = mha.concatHeads(attnOutput, batch, seqLenQ)

	// Final linear projection
	return mha.WO.Forward(attnOutput)
}

// splitHeads reshapes (batch, seqLen, dModel) -> (batch * numHeads, seqLen, dK)
func (mha *MultiHeadAttention) splitHeads(x *Tensor, batch, seqLen int) *Tensor {
	// x: (batch, seqLen, dModel)
	// Reshape to (batch, seqLen, numHeads, dK)
	// Then transpose to (batch, numHeads, seqLen, dK)
	// Then reshape to (batch * numHeads, seqLen, dK)
	
	result := NewTensor(batch*mha.NumHeads, seqLen, mha.DK)
	for b := 0; b < batch; b++ {
		for h := 0; h < mha.NumHeads; h++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < mha.DK; d++ {
					// Original index: (b, s, h * dK + d)
					val := x.At(b, s, h*mha.DK+d)
					// New index: (b * numHeads + h, s, d)
					result.Set(val, b*mha.NumHeads+h, s, d)
				}
			}
		}
	}
	return result
}

// concatHeads reshapes (batch * numHeads, seqLen, dK) -> (batch, seqLen, dModel)
func (mha *MultiHeadAttention) concatHeads(x *Tensor, batch, seqLen int) *Tensor {
	result := NewTensor(batch, seqLen, mha.DModel)
	for b := 0; b < batch; b++ {
		for h := 0; h < mha.NumHeads; h++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < mha.DK; d++ {
					// Original index: (b * numHeads + h, s, d)
					val := x.At(b*mha.NumHeads+h, s, d)
					// New index: (b, s, h * dK + d)
					result.Set(val, b, s, h*mha.DK+d)
				}
			}
		}
	}
	return result
}

// CreateCausalMask creates a causal (look-ahead) mask for decoder self-attention.
// Returns a mask of shape (batch, seqLen, seqLen) where mask[b][i][j] = 1 if i >= j, else 0.
func CreateCausalMask(batch, seqLen int) *Tensor {
	mask := NewTensor(batch, seqLen, seqLen)
	for b := 0; b < batch; b++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if i >= j {
					mask.Set(1.0, b, i, j)
				}
			}
		}
	}
	return mask
}
