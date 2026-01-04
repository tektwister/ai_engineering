# Multimodal Chain of Thought (CoT) Framework in Go

This project implements a flexible Chain of Thought (CoT) framework in Go, supporting multiple LLM providers (OpenAI, Anthropic, Google Gemini) and multimodal inputs (Text + Images).

## Features

- **Hexagonal Architecture**: Separation of concerns with domain logic, ports, and adapters.
- **Multi-Provider Support**: 
  - **OpenAI**: GPT-4o, GPT-4 Turbo, etc.
  - **Anthropic**: Claude 3.5 Sonnet, Opus, Haiku.
  - **Google**: Gemini 1.5 Pro/Flash, Gemini 2.0.
- **Multimodal**: Support for reasoning with images (Vision CoT).
- **Extensible Strategies**: Plug-and-play prompting strategies (currently implements Zero-Shot CoT).
- **Streaming**: Support for real-time response streaming.

## Project Structure

```
chain_of_thoughts/
├── cmd/
│   └── main.go           # Entry point
├── internal/
│   ├── domain/           # Core business logic & interfaces
│   ├── core/             # Application business rules (Engine)
│   └── adapters/
│       └── providers/    # LLM Provider implementations
├── go.mod                # Go module definition
└── env.example           # Example environment variables
```

## Setup

1. **Clone the repository** (if not already local).
2. **Install dependencies**:
   ```bash
   go mod tidy
   ```
3. **Configure Environment**:
   Copy `env.example` to `.env` (or set env vars directly):
   ```bash
   cp env.example .env
   ```
   Edit `.env` and add your API key:
   ```env
   LLM_API_KEY=sk-...
   LLM_PROVIDER=openai # or anthropic, google
   ```

## Usage

Run the example:

```bash
go run cmd/main.go
```

The example will:
1. Run a text-only logic puzzle ("Jug problem") showing step-by-step reasoning.
2. If `example_chart.png` exists in the root, it will run a multimodal analysis on it.

## Adding a New Provider

1. Implement the `domain.LLMProvider` interface.
2. Register it in `internal/adapters/providers/factory.go`.
