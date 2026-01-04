package transformer

import (
	"testing"
)

func TestTensorBasics(t *testing.T) {
	// Test tensor creation
	tensor := NewTensor(2, 3)
	if len(tensor.Data) != 6 {
		t.Errorf("Expected size 6, got %d", len(tensor.Data))
	}
	if tensor.Shape[0] != 2 || tensor.Shape[1] != 3 {
		t.Errorf("Expected shape [2, 3], got %v", tensor.Shape)
	}

	// Test set and get
	tensor.Set(5.0, 1, 2)
	if tensor.At(1, 2) != 5.0 {
		t.Errorf("Expected 5.0, got %f", tensor.At(1, 2))
	}
}

func TestMatMul(t *testing.T) {
	// (2, 3) @ (3, 2) = (2, 2)
	a := NewTensorFromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := NewTensorFromData([]float64{1, 2, 3, 4, 5, 6}, 3, 2)

	c := a.MatMul(b)
	if c.Shape[0] != 2 || c.Shape[1] != 2 {
		t.Errorf("Expected shape [2, 2], got %v", c.Shape)
	}

	// Manual calculation:
	// c[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
	// c[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
	// c[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
	// c[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
	expected := []float64{22, 28, 49, 64}
	for i, exp := range expected {
		if c.Data[i] != exp {
			t.Errorf("Expected c.Data[%d] = %f, got %f", i, exp, c.Data[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	tensor := NewTensorFromData([]float64{1, 2, 3, 1, 2, 3}, 2, 3)
	softmaxed := tensor.Softmax()

	// Check that each row sums to 1
	for i := 0; i < 2; i++ {
		sum := 0.0
		for j := 0; j < 3; j++ {
			sum += softmaxed.At(i, j)
		}
		if sum < 0.999 || sum > 1.001 {
			t.Errorf("Expected row %d sum to be 1, got %f", i, sum)
		}
	}
}

func TestLinear(t *testing.T) {
	linear := NewLinear(4, 2)
	input := Randn(3, 4) // (seqLen, inFeatures)

	output := linear.Forward(input)
	if output.Shape[0] != 3 || output.Shape[1] != 2 {
		t.Errorf("Expected shape [3, 2], got %v", output.Shape)
	}
}

func TestLayerNorm(t *testing.T) {
	ln := NewLayerNorm(4)
	input := NewTensorFromData([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)

	output := ln.Forward(input)
	if output.Shape[0] != 2 || output.Shape[1] != 4 {
		t.Errorf("Expected shape [2, 4], got %v", output.Shape)
	}

	// Check that each row has approximately zero mean after normalization
	for i := 0; i < 2; i++ {
		mean := 0.0
		for j := 0; j < 4; j++ {
			mean += output.At(i, j)
		}
		mean /= 4.0
		if mean < -0.1 || mean > 0.1 {
			t.Errorf("Expected row %d mean to be ~0, got %f", i, mean)
		}
	}
}

func TestEmbedding(t *testing.T) {
	emb := NewEmbedding(100, 16)
	tokens := []int{1, 5, 10}

	output := emb.Forward(tokens)
	if output.Shape[0] != 3 || output.Shape[1] != 16 {
		t.Errorf("Expected shape [3, 16], got %v", output.Shape)
	}
}

func TestPositionalEncoding(t *testing.T) {
	pe := NewPositionalEncoding(16, 100, 0.0)
	input := Randn(10, 16) // (seqLen, dModel)

	output := pe.Forward(input)
	if output.Shape[0] != 10 || output.Shape[1] != 16 {
		t.Errorf("Expected shape [10, 16], got %v", output.Shape)
	}
}

func TestMultiHeadAttention(t *testing.T) {
	mha := NewMultiHeadAttention(16, 4) // dModel=16, numHeads=4
	
	// (batch, seqLen, dModel)
	batch, seqLen := 2, 5
	query := Randn(batch, seqLen, 16)
	key := Randn(batch, seqLen, 16)
	value := Randn(batch, seqLen, 16)

	output := mha.Forward(query, key, value, nil)
	if output.Shape[0] != batch || output.Shape[1] != seqLen || output.Shape[2] != 16 {
		t.Errorf("Expected shape [%d, %d, 16], got %v", batch, seqLen, output.Shape)
	}
}

func TestEncoderLayer(t *testing.T) {
	el := NewEncoderLayer(16, 4, 64, 0.0)
	
	batch, seqLen := 2, 5
	input := Randn(batch, seqLen, 16)

	output := el.Forward(input, nil)
	if output.Shape[0] != batch || output.Shape[1] != seqLen || output.Shape[2] != 16 {
		t.Errorf("Expected shape [%d, %d, 16], got %v", batch, seqLen, output.Shape)
	}
}

func TestTransformerForward(t *testing.T) {
	config := &TransformerConfig{
		SrcVocabSize:     100,
		TgtVocabSize:     100,
		DModel:           32,
		NumHeads:         4,
		NumEncoderLayers: 2,
		NumDecoderLayers: 2,
		DFF:              128,
		MaxSeqLen:        50,
		Dropout:          0.0,
	}
	transformer := NewTransformer(config)

	srcTokens := [][]int{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}
	tgtTokens := [][]int{{1, 2, 3}, {4, 5, 6}}

	logits := transformer.Forward(srcTokens, tgtTokens, nil, nil)
	
	batch := len(srcTokens)
	tgtLen := len(tgtTokens[0])
	if logits.Shape[0] != batch || logits.Shape[1] != tgtLen || logits.Shape[2] != config.TgtVocabSize {
		t.Errorf("Expected shape [%d, %d, %d], got %v", batch, tgtLen, config.TgtVocabSize, logits.Shape)
	}

	t.Logf("Transformer forward pass completed successfully!")
	t.Logf("Output logits shape: %v", logits.Shape)
}

func TestCrossEntropyLoss(t *testing.T) {
	// Simple test: create logits where target has highest value
	logits := NewTensor(1, 2, 5) // (batch, seqLen, vocabSize)
	// Position 0: target is 2
	logits.Set(0.1, 0, 0, 0)
	logits.Set(0.1, 0, 0, 1)
	logits.Set(5.0, 0, 0, 2) // highest
	logits.Set(0.1, 0, 0, 3)
	logits.Set(0.1, 0, 0, 4)
	// Position 1: target is 4
	logits.Set(0.1, 0, 1, 0)
	logits.Set(0.1, 0, 1, 1)
	logits.Set(0.1, 0, 1, 2)
	logits.Set(0.1, 0, 1, 3)
	logits.Set(5.0, 0, 1, 4) // highest

	targets := [][]int{{2, 4}}
	loss := CrossEntropyLoss(logits, targets)

	// Loss should be low since targets have highest logits
	if loss > 1.0 {
		t.Errorf("Expected low loss, got %f", loss)
	}
	t.Logf("Cross-entropy loss: %f", loss)
}

func TestGeneration(t *testing.T) {
	config := &TransformerConfig{
		SrcVocabSize:     100,
		TgtVocabSize:     100,
		DModel:           32,
		NumHeads:         4,
		NumEncoderLayers: 1,
		NumDecoderLayers: 1,
		DFF:              64,
		MaxSeqLen:        50,
		Dropout:          0.0,
	}
	transformer := NewTransformer(config)

	srcTokens := [][]int{{1, 2, 3, 4, 5}}
	startToken := 1
	endToken := 99

	generated := transformer.Generate(srcTokens, 10, startToken, endToken)

	if len(generated) != 1 {
		t.Errorf("Expected 1 generated sequence, got %d", len(generated))
	}
	if len(generated[0]) < 2 {
		t.Errorf("Expected at least 2 tokens (including start), got %d", len(generated[0]))
	}
	if generated[0][0] != startToken {
		t.Errorf("Expected first token to be start token %d, got %d", startToken, generated[0][0])
	}

	t.Logf("Generated sequence: %v", generated[0])
}
