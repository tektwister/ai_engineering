# Multimodal Chain of Thought (CoT) Framework in Go

This project implements a flexible Chain of Thought (CoT) framework in Go. It supports multiple LLM providers (OpenAI, Google/Gemini, Groq, HuggingFace) and multimodal inputs (text + images), with a hexagonal architecture separating domain logic, ports, and provider adapters.

## About `chain_of_thoughts`

`chain_of_thoughts` is a reference implementation of the CoT reasoning engine. It demonstrates the hexagonal architecture in practice:

- **Domain Layer** (`internal/domain/`): Core business logic, ports (interfaces), and domain models such as `CoTRequest`, `CoTResponse`, `ChainOfThought`, and `Message`.
- **Application Layer** (`internal/core/`): The `Engine` orchestrates reasoning flows and strategies; `ZeroShotCoTStrategy` implements prompt construction and response parsing for step-by-step reasoning.
- **Adapter Layer** (`pkg/llm/providers/`): Provider implementations (OpenAI, Google, Groq, HuggingFace) and the provider factory pattern for pluggable LLM integrations.

The example runner in `cmd/main.go` demonstrates:
1. **Text-only reasoning** on logic puzzles (e.g., the water jug problem).
2. **Multimodal reasoning** with images (e.g., chart analysis), if the provider supports it.

## Features

- **Hexagonal Architecture**: Clear separation between `internal/domain` (ports/models), `internal/core` (engine/business rules), and provider implementations under `pkg/llm/providers`.
- **Multi-Provider Support**: OpenAI, Google/Gemini, Groq, HuggingFace (providers live under `pkg/llm/providers`).
- **Multimodal**: Vision CoT support for image inputs—send images as base64-encoded content alongside text.
- **Extensible Prompt Strategies**: Plug-and-play prompting strategies (e.g., `ZeroShotCoTStrategy`); easily add custom strategies by implementing the `domain.PromptStrategy` interface.
- **Streaming**: Real-time response streaming support via `ReasonStream()`.
- **Structured Output**: Responses are parsed into structured `ChainOfThought` objects with reasoning steps, final answers, and token usage metrics.

## Project Structure

```
chain_of_thoughts/
├── cmd/
│   └── main.go               # Example runner: text-only and multimodal demos
├── internal/
│   ├── core/
│   │   ├── engine.go         # Orchestrator: runs Reason() and ReasonStream()
│   │   └── strategies.go     # ZeroShotCoTStrategy: builds prompts and parses responses
│   └── domain/
│       ├── models.go         # CoTRequest, CoTResponse, ChainOfThought, Message, etc.
│       └── ports.go          # Interfaces: ChainOfThoughtEngine, PromptStrategy, LLMProvider
├── go.mod                    # Go module definition
├── go.sum                    # Dependency lock file
└── env.example               # Example environment variables
```

## Architecture Overview

### Domain Layer (`internal/domain/`)

**Models:**
- `CoTRequest`: Input to the engine; contains messages (text/multimodal) and max tokens.
- `CoTResponse`: Engine output; contains the structured reasoning chain.
- `ChainOfThought`: Parsed reasoning breakdown with `Steps`, `FinalAnswer`, provider/model info, and token usage.
- `Message`: Text or multimodal content (text + images in base64).

**Ports (Interfaces):**
- `ChainOfThoughtEngine`: Reason() for non-streaming, ReasonStream() for streaming, provider/strategy setters.
- `PromptStrategy`: Builds prompts from messages, parses raw responses into `ChainOfThought`.
- `LLMProvider`: Interface for LLM provider implementations.

### Application Layer (`internal/core/`)

- **Engine**: Receives a provider and strategy; orchestrates the reasoning flow.
  - `Reason()`: Sends a prompt to the provider and parses the response into a `ChainOfThought`.
  - `ReasonStream()`: Streams response chunks as they arrive.
- **ZeroShotCoTStrategy**: 
  - `BuildPrompt()`: Injects a CoT instruction ("Think step by step...") into the messages.
  - `ParseResponse()`: Extracts reasoning steps and final answer from the LLM response (looks for patterns like "Step 1:", "Final Answer:", etc.).

## Setup

1. Clone the repository (if not already local).
2. Install dependencies:
```bash
go mod tidy
```
3. Configure environment variables: copy `env.example` to `.env` and add your API key.

Cross-platform copy examples:

```bash
# Unix / macOS
cp env.example .env

# Windows (PowerShell)
Copy-Item env.example .env

# Windows (cmd.exe)
copy env.example .env
```

Edit `.env` and set values such as:

```env
LLM_API_KEY=sk-...
LLM_PROVIDER=openai # or google, groq, huggingface
```

## Usage

### Running the Example

```bash
go run cmd/main.go
```

**What the example does:**
1. **Text-Only Reasoning**: Solves the water jug problem ("How do I measure exactly 4 gallons with a 3-gallon and 5-gallon jug?").
2. **Multimodal Reasoning** (optional): If `example_chart.png` is present and the provider supports multimodal input, analyzes the chart and predicts trends.

**Output includes:**
- Reasoning steps extracted from the LLM response.
- Final answer.
- Model name, provider, duration, and token usage.

### Using the Engine Programmatically

```go
import (
	"github.com/tektwister/ai_engineering/chain-of-thoughts/internal/core"
	"github.com/tektwister/ai_engineering/chain-of-thoughts/internal/domain"
	"github.com/tektwister/ai_engineering/pkg/llm/providers"
)

// Create provider
providerConfig := &domain.ProviderConfig{APIKey: "sk-..."}
provider, _ := providers.Create("openai", providerConfig)

// Create engine with strategy
strategy := &core.ZeroShotCoTStrategy{}
engine := core.NewEngine(provider, strategy)

// Reason
req := &domain.CoTRequest{
	Messages: []domain.Message{
		domain.NewTextMessage(domain.RoleUser, "Your question here"),
	},
	MaxTokens: 1000,
}
resp, _ := engine.Reason(context.Background(), req)

// Use response
for _, step := range resp.Chain.Steps {
	fmt.Printf("Step %d: %s\n", step.StepNumber, step.Reasoning)
}
fmt.Printf("Final Answer: %s\n", resp.Chain.FinalAnswer)
```

### Streaming Responses

```go
respChan, _ := engine.ReasonStream(context.Background(), req)
for chunk := range respChan {
	if chunk.Error != nil {
		log.Printf("Error: %v", chunk.Error)
		break
	}
	fmt.Print(chunk.Content)
}
```

## Adding a New Provider

To use a new provider, simply create it and pass it to the engine:
```go
provider, _ := providers.Create("yourprovider", providerConfig)
engine.SetProvider(provider)
```

Any provider implementing the `LLMProvider` interface can be used with the engine.

## Adding a New Strategy

1. **Implement `domain.PromptStrategy`**:
   ```go
   type MyCustomStrategy struct {}
   
   func (s *MyCustomStrategy) Name() string {
       return "my-custom"
   }
   
   func (s *MyCustomStrategy) BuildPrompt(messages []Message) []Message {
       // Add strategy-specific instructions
   }
   
   func (s *MyCustomStrategy) ParseResponse(rawResponse string) (*ChainOfThought, error) {
       // Extract reasoning and final answer
   }
   ```

2. **Use in the engine**:
   ```go
   strategy := &MyCustomStrategy{}
   engine.SetPromptStrategy(strategy)
   ```

## Environment Variables

- `LLM_API_KEY`: API key for your LLM provider.
- `LLM_PROVIDER`: Provider name (openai, google, groq, huggingface).
- `LLM_BASE_URL`: (Optional) Custom base URL for the provider.
- `LLM_ORG_ID`: (Optional) Organization ID (e.g., for OpenAI).

## License

See the root repository LICENSE file.
