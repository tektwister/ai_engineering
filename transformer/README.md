# Transformer - "Attention is All You Need" Implementation

A complete implementation of the Transformer architecture from the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) in Go.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Transformer                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │  Source Tokens   │          │  Target Tokens   │        │
│  └────────┬─────────┘          └────────┬─────────┘        │
│           │                              │                  │
│           ▼                              ▼                  │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │    Embedding     │          │    Embedding     │        │
│  │  + Positional    │          │  + Positional    │        │
│  │    Encoding      │          │    Encoding      │        │
│  └────────┬─────────┘          └────────┬─────────┘        │
│           │                              │                  │
│           ▼                              ▼                  │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │                  │          │                  │        │
│  │     ENCODER      │─────────▶│     DECODER      │        │
│  │  (N x Layers)    │          │  (N x Layers)    │        │
│  │                  │          │                  │        │
│  └──────────────────┘          └────────┬─────────┘        │
│                                          │                  │
│                                          ▼                  │
│                                ┌──────────────────┐        │
│                                │  Linear + Softmax│        │
│                                │    (Vocab Size)  │        │
│                                └────────┬─────────┘        │
│                                          │                  │
│                                          ▼                  │
│                                ┌──────────────────┐        │
│                                │  Output Logits   │        │
│                                └──────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Core Tensor Operations (`tensor.go`)
- Multi-dimensional tensor with shape metadata
- Matrix multiplication (2D and batched 3D)
- Element-wise operations (add, sub, mul, div)
- Softmax, ReLU, GELU activations
- Transpose operations

### Neural Network Layers (`layers.go`)
- **Linear**: Fully connected layer with Xavier initialization
- **Embedding**: Token embedding lookup table
- **LayerNorm**: Layer normalization
- **Dropout**: Regularization (inference mode only)

### Attention Mechanism (`attention.go`)
- **Scaled Dot-Product Attention**: Core attention computation
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k) V
  ```
- **Multi-Head Attention**: Parallel attention with multiple heads
- **Causal Masking**: For autoregressive decoding

### Encoder/Decoder (`encoder_decoder.go`)
- **Positional Encoding**: Sinusoidal position embeddings
- **Feed-Forward Network**: Position-wise FFN
- **Encoder Layer**: Self-attention + FFN
- **Decoder Layer**: Masked self-attention + Cross-attention + FFN
- **Encoder/Decoder Stacks**: N-layer stacks

### Transformer Model (`transformer.go`)
- **Complete Transformer**: Full encoder-decoder architecture
- **Forward Pass**: Encode + Decode pipeline
- **Generation**: Autoregressive greedy decoding
- **Cross-Entropy Loss**: Training objective

## Usage

### Create a Transformer

```go
import "github.com/tektwister/ai_engineering/transformer"

// Use default configuration
config := transformer.DefaultConfig()

// Or create custom config
config := &transformer.TransformerConfig{
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

model := transformer.NewTransformer(config)
```

### Forward Pass

```go
// Batch of source and target token sequences
srcTokens := [][]int{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}
tgtTokens := [][]int{{1, 2, 3}, {4, 5, 6}}

// Get logits: (batch, tgtLen, vocabSize)
logits := model.Forward(srcTokens, tgtTokens, nil, nil)
```

### Text Generation

```go
srcTokens := [][]int{{1, 2, 3, 4, 5}}
startToken := 1    // <BOS>
endToken := 2      // <EOS>
maxLen := 50

generated := model.Generate(srcTokens, maxLen, startToken, endToken)
// generated[0] = [1, 45, 23, 67, ...]
```

### Compute Loss

```go
logits := model.Forward(srcTokens, tgtTokens, nil, nil)
targets := [][]int{{2, 3, 4}, {5, 6, 7}} // shifted targets
loss := transformer.CrossEntropyLoss(logits, targets)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SrcVocabSize` | 10000 | Source vocabulary size |
| `TgtVocabSize` | 10000 | Target vocabulary size |
| `DModel` | 512 | Model dimension |
| `NumHeads` | 8 | Number of attention heads |
| `NumEncoderLayers` | 6 | Number of encoder layers |
| `NumDecoderLayers` | 6 | Number of decoder layers |
| `DFF` | 2048 | Feed-forward hidden dimension |
| `MaxSeqLen` | 512 | Maximum sequence length |
| `Dropout` | 0.1 | Dropout rate |

## Testing

```bash
cd transformer
go test -v
```

## Key Features

✅ **Full Encoder-Decoder Architecture**  
✅ **Multi-Head Self-Attention**  
✅ **Cross-Attention for Encoder-Decoder**  
✅ **Sinusoidal Positional Encoding**  
✅ **Layer Normalization**  
✅ **Residual Connections**  
✅ **Causal Masking for Decoding**  
✅ **Greedy Decoding Generation**  
✅ **Cross-Entropy Loss**  
✅ **Xavier Weight Initialization**  

## Paper Reference

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## Future Enhancements

- [ ] Backward pass and gradient computation
- [ ] Training loop with optimizer
- [ ] Beam search decoding
- [ ] Nucleus (top-p) sampling
- [ ] Temperature-based sampling  
- [ ] Weight loading/saving
- [ ] Pre-LayerNorm variant
- [ ] Rotary Position Embeddings (RoPE)
