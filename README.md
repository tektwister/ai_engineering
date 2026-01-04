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

- [x] [Reasoner (Chain of Thought implementation)](chain_of_thoughts/README.md)

## To Do

- [ ] Agent loop (ReAct pattern)
- [ ] Inference Server (in C++/Rust)
- [ ] Transformer from scratch (Attention is all you need)
- [ ] Vector Database (HNSW index)
- [ ] RAG pipeline
- [ ] Flash Attention kernel (CUDA)
- [ ] Quantization library (Int8/FP4 implementation)
- [ ] Mixture of Experts (MoE) routing layer
- [ ] Distributed training loop (FSDP/Tensor Parallelism)
- [ ] KV Cache paging system (like vLLM)
- [ ] Speculative Decoding system
- [ ] State Space Model (Mamba implementation)
- [ ] RLHF pipeline (PPO implementation)
- [ ] Small Language Model (SLM)
- [ ] Matrix Multiplication kernel
- [ ] LoRA (Low-Rank Adaptation) trainer
- [ ] Code interpreter sandbox
- [ ] DPO (Direct Preference Optimization) loss function
- [ ] Graph RAG system
- [ ] Model merger (Model Soups/Spherical Linear Interpolation)
- [ ] Interpretability tool (SAE - Sparse Autoencoders)
- [ ] Synthetic data generator
- [ ] Function Calling router
- [ ] Structured Output parser (Context Free Grammars)
- [ ] Multi-modal projector (CLIP implementation)
- [ ] LLM Eval harness
- [ ] Guardrails system (Input/Output filtering)
- [ ] Prompt caching mechanism
- [ ] Tokenizer (BPE implementation)
- [ ] Autograd engine (like Micrograd)
- [ ] Diffusion model (UNet + Scheduler)
- [ ] Vision Transformer (ViT)
- [ ] Whisper-style ASR model
- [ ] Text-to-Speech pipeline
- [ ] Semantic Router
- [ ] Knowledge Graph builder
- [ ] Data curation pipeline (MinHash/Deduplication)
- [ ] AI Gateway (Load balancing/Failover)
- [ ] Parameter Efficient Fine-Tuning (PEFT) library
- [ ] Text-to-SQL engine
- [ ] Recommendation system (Two-tower architecture)
- [ ] Embedding model
- [ ] Logit Processor
- [ ] Softmax kernel optimization
- [ ] Adversarial attack generator
- [ ] Audio Spectrogram transformer
- [ ] Neural Architecture Search
- [ ] Model Distillation pipeline
- [ ] Feature Store
- [ ] Database driver (for Vectors)
