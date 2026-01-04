# Small Language Model

A complete implementation of a small language model in Go, featuring:

- **Decoder-only Transformer (GPT-like architecture)**
- **Byte Pair Encoding (BPE) tokenizer**
- **Advanced sampling methods** (temperature, top-k, top-p)
- **Training framework** with gradient verification
- **GPT-2 weight loading** capability

## Features

### 🏗️ Architecture
- GPT-style decoder-only transformer
- Configurable model sizes (small, medium, large)
- Multi-head attention with causal masking
- Feed-forward networks with GELU activation
- Layer normalization and residual connections

### 🔤 Tokenization
- Byte Pair Encoding (BPE) implementation
- Trainable vocabulary from any text corpus
- Efficient encoding/decoding

### 🎯 Sampling Methods
- **Temperature sampling** - Controls randomness
- **Top-k sampling** - Limits to k highest probability tokens
- **Top-p (nucleus) sampling** - Limits to tokens comprising top-p probability mass
- **Greedy sampling** - Deterministic, highest probability token

### 🏋️ Training
- Gradient verification using finite differences
- Batch training with configurable sequence length
- Loss monitoring and checkpointing
- Support for custom datasets

### 📦 GPT-2 Compatibility
- Weight loading from GPT-2 format (mock implementation)
- Architecture matching GPT-2 specifications
- Easy extension to load real GPT-2 weights

## Quick Start

### Build and Run

```bash
cd small_language_model
go build ./cmd
./cmd
```

### Generate Text

```bash
# Generate text with default settings
go run ./cmd -mode=generate -prompt="Once upon a time"

# Generate with custom sampling
go run ./cmd -mode=generate -prompt="The cat" -temperature=0.8 -top_k=40 -top_p=0.9 -max_tokens=100
```

### Train a Model

```bash
# Quick training on Shakespeare
go run ./cmd -mode=train -data=shakespeare -max_iters=50

# Train on TinyStories
go run ./cmd -mode=train -data=tinystories -size=medium
```

### Gradient Checking

```bash
# Verify gradients are working correctly
go run ./cmd -mode=gradient_check
```

## Architecture Details

### GPT Model
```go
config := SmallGPTConfig() // vocab=1000, layers=4, heads=4, embd=128
gpt := NewGPT(config)
```

### Language Model Pipeline
```go
model := NewLanguageModel(&ModelConfig{
    GPTConfig:   gptConfig,
    SamplerType: domain.SamplingTopP,
    Temperature: 1.0,
    TopK:        50,
    TopP:        0.9,
})

// Train tokenizer
model.TrainTokenizer(text, vocabSize)

// Generate text
generated, _ := model.GenerateText("Hello world", 50)
```

### Training Loop
```go
trainer := NewTrainer(model, &TrainingConfig{
    LearningRate: 1e-3,
    MaxIters:     100,
    BatchSize:    8,
    SeqLength:    64,
})

trainer.TrainOnData(text, config)
```

## Model Configurations

| Size | Layers | Heads | Embedding | Vocab | Context |
|------|--------|-------|-----------|-------|---------|
| Small | 4 | 4 | 128 | 1000 | 128 |
| Medium | 6 | 6 | 384 | 2000 | 256 |
| Large | 12 | 12 | 768 | 5000 | 512 |

## Dependencies

- `transformer` - Tensor operations and transformer layers
- `tokenizer` - BPE tokenization
- `logit_processor` - Advanced sampling strategies

## File Structure

```
small_language_model/
├── cmd/
│   └── main.go              # CLI application
├── gpt.go                   # GPT model implementation
├── model.go                 # Language model wrapper
├── trainer.go               # Training framework
├── gpt2_loader.go           # GPT-2 weight loading
├── go.mod                   # Module dependencies
└── README.md               # This file
```

## Implementation Notes

### Gradient Verification
The trainer includes gradient checking using finite differences to verify that gradients are computed correctly. This is essential for debugging custom transformer implementations.

### Sampling Implementation
Sampling is handled by the `logit_processor` package, which provides:
- Temperature scaling: `logits /= temperature`
- Top-k filtering: Keep only k highest probability tokens
- Top-p filtering: Keep tokens comprising top-p cumulative probability

### GPT-2 Compatibility
The loader provides a framework for loading GPT-2 weights. In practice, you'd need to:
1. Parse actual GPT-2 weight files (safetensors, pickle, etc.)
2. Map weight matrices to the correct tensor positions
3. Handle different architectures (GPT-2 small/medium/large)

## Performance

- **Small model**: ~1M parameters, trains on CPU in reasonable time
- **Memory usage**: Scales linearly with model size
- **Inference speed**: Fast autoregressive generation
- **Training**: Supports batch processing for efficiency

## Future Enhancements

- [ ] Real GPT-2 weight loading from Hugging Face
- [ ] Optimizer implementations (Adam, etc.)
- [ ] Mixed precision training
- [ ] Distributed training support
- [ ] Model parallelism
- [ ] Custom dataset loading
- [ ] Evaluation metrics (perplexity, BLEU, etc.)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
