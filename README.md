# AI Engineering Roadmap using Golang

A comprehensive collection of AI Engineering implementations in Go, demonstrating modern architectures and patterns. Each module is designed to be modular, well-tested, and production-ready.

## Setup

### Prerequisites

- **Go 1.20+**: Download from [golang.org](https://golang.org/dl/)
- **API Keys** (optional, for LLM providers):
  - OpenAI: [platform.openai.com](https://platform.openai.com/)
  - Google Gemini: [makersuite.google.com](https://makersuite.google.com/)
  - Groq: [console.groq.com](https://console.groq.com/)
  - HuggingFace: [huggingface.co](https://huggingface.co/)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/tektwister/ai_engineering.git
cd ai_engineering
```

2. **Configure environment variables**:
Copy the env example file and add your API keys:

cp chain_of_thoughts/env.example .env

Edit `.env`:
```env
LLM_API_KEY=sk-...
LLM_PROVIDER=openai # or google, groq, huggingface
LLM_BASE_URL=      # (optional) custom base URL
LLM_ORG_ID=        # (optional) organization ID
```

### Adding a New Provider

Implement the `llm.Provider` interface in `pkg/llm/providers/<yourprovider>/` and register it in the factory.

### Adding a New Module

1. Create a new directory at the root level.
2. Add a `go.mod` file.
3. Implement domain layer, application layer, and adapters (following hexagonal architecture).
4. Include a comprehensive `README.md` with usage examples.

## Completed

### Phase 1: Fundamentals
- [x] [Autograd engine (like Micrograd)](autograd_engine/README.md)
- [x] [Tokenizer (BPE implementation)](tokenizer/README.md)
- [x] [Matrix Multiplication kernel](matrix_multiplication_kernel/README.md)
- [x] [Transformer from scratch (Attention is all you need)](transformer/README.md)
- [x] [Logit Processor](logit_processor/README.md)

### Applications & Systems
- [x] [Reasoner (Chain of Thought implementation)](chain_of_thoughts/README.md)
- [x] [Small Language Model (SLM)](small_language_model/README.md)

## To Do

### Phase 2: Model Optimization
- [ ] Flash Attention kernel (CUDA)
- [ ] Quantization library (Int8/FP4 implementation)
- [ ] LoRA (Low-Rank Adaptation) trainer
- [ ] Parameter Efficient Fine-Tuning (PEFT) library

### Phase 3: Training at Scale
- [ ] Distributed training loop (FSDP/Tensor Parallelism)
- [ ] RLHF pipeline (PPO implementation)
- [ ] DPO (Direct Preference Optimization) loss function
- [ ] Model Distillation pipeline

### Phase 4: Inference & Production
- [ ] KV Cache paging system (like vLLM)
- [ ] Speculative Decoding system
- [ ] Inference Server (in C++/Rust)
- [ ] Prompt caching mechanism

### Phase 5: Advanced Architectures
- [ ] Vision Transformer (ViT)
- [ ] State Space Model (Mamba implementation)
- [ ] Mixture of Experts (MoE) routing layer
- [ ] Multi-modal projector (CLIP implementation)
- [ ] Diffusion model (UNet + Scheduler)

### Phase 6: Applications & Systems
- [ ] Agent loop (ReAct pattern)
- [ ] RAG pipeline
- [ ] Vector Database (HNSW index)
- [ ] Function Calling router
- [ ] Structured Output parser (Context Free Grammars)
- [ ] LLM Eval harness
- [ ] Guardrails system (Input/Output filtering)
- [ ] Code interpreter sandbox
- [ ] Graph RAG system
- [ ] Model merger (Model Soups/Spherical Linear Interpolation)
- [ ] Interpretability tool (SAE - Sparse Autoencoders)
- [ ] Synthetic data generator
- [ ] Semantic Router
- [ ] Knowledge Graph builder
- [ ] Data curation pipeline (MinHash/Deduplication)
- [ ] AI Gateway (Load balancing/Failover)
- [ ] Text-to-SQL engine
- [ ] Recommendation system (Two-tower architecture)
- [ ] Embedding model
- [ ] Softmax kernel optimization
- [ ] Adversarial attack generator
- [ ] Audio Spectrogram transformer
- [ ] Whisper-style ASR model
- [ ] Text-to-Speech pipeline
- [ ] Neural Architecture Search
- [ ] Feature Store
- [ ] Database driver (for Vectors)
