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

### Design Pattern: Hexagonal Architecture

The `chain_of_thoughts` module follows the **Hexagonal Architecture** pattern (also called Ports & Adapters), which isolates domain logic from external dependencies:

```
┌─────────────────────────────────────────────────────────┐
│                    External World                        │
│  (OpenAI, Google, Groq, HuggingFace APIs)              │
└─────────────────────────────────────────────────────────┘
                           ↕
                    ┌──────────────┐
                    │   Adapters   │
                    │  (Providers) │
                    └──────────────┘
                           ↕
                    ┌──────────────┐
                    │    Ports     │
                    │(Interfaces)  │
                    └──────────────┘
                           ↕
            ┌──────────────────────────────┐
            │      Domain Layer            │
            │   (Business Logic, Models)   │
            └──────────────────────────────┘
                           ↕
            ┌──────────────────────────────┐
            │    Application Layer         │
            │  (Engine, Orchestration)     │
            └──────────────────────────────┘
```

**Data Flow:**
1. **User Input** → `cmd/main.go` creates a `CoTRequest` (domain model)
2. **Engine** (`internal/core/Engine`) receives request + provider + strategy
3. **Strategy** (`ZeroShotCoTStrategy`) builds a CoT prompt from the request
4. **Adapter** (Provider) executes the prompt via the LLM API
5. **Strategy** parses the raw response into structured `ChainOfThought` (reasoning steps + final answer)
6. **Engine** returns `CoTResponse` to the caller

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

## Sample Output

Below is an example output from running the text-only logic puzzle example with the Groq provider:

```
=== Running Text-Only Logic Puzzle ===
Question: If I have a 3 gallon jug and a 5 gallon jug, how can I measure exactly 4 gallons of water?

Thinking...
------------------------------------------------
Model:  (groq)
Duration: 3.1042517s
Tokens: 5672 (Prompt: 4875, Response: 797)
------------------------------------------------
### Reasoning Steps:
1. To solve this problem, let's break it down into steps. The first step is to determine the available operations.

We have two jugs: a 3-gallon jug and a 5-gallon jug. The available operations are:

1. Filling a jug from an external water source
2. Emptying a jug
3. Transferring water from one jug to another

Let's consider the initial state of the jugs: empty.

2. Next logical step: Fill the 5-gallon jug from the external water source.

We now have a 5-gallon jug filled with water and an empty 3-gallon jug.

3. Next logical step: Pour the water from the 5-gallon jug into the 3-gallon jug until the 3-gallon jug is full, leaving 2 gallons remaining in the 5-gallon jug (5 - 3 = 2).

We now have a 3-gallon jug filled with water, a 5-gallon jug with 2 gallons of water, and an empty 5-gallon jug.

4. Next logical step: Empty the 3-gallon jug, which is currently full. This will leave the 5-gallon jug with 2 gallons of water and the 3-gallon jug empty.

We now have a 3-gallon jug empty, a 5-gallon jug with 2 gallons of water, and an empty 3-gallon jug.

5. Next logical step: Fill the 3-gallon jug from the external water source.

We now have a 3-gallon jug filled with water, a 5-gallon jug with 2 gallons of water, and an empty 5-gallon jug.

Note: We've made progress, but we still need to find a way to measure exactly 4 gallons. Let's continue.

6. Next logical step: Pour the 2 gallons from the 5-gallon jug into the 3-gallon jug, filling it completely. This will leave 0 gallons in the 5-gallon jug and 2 gallons in the 3-gallon jug.

We now have a 3-gallon jug with 2 gallons of water, a 5-gallon jug empty, and an empty 5-gallon jug.

7. Next logical step: Pour the 2 gallons from the 3-gallon jug into the 5-gallon jug. This will leave 0 gallons in the 3-gallon jug and 2 gallons in the 5-gallon jug.

We now have a 3-gallon jug empty, a 5-gallon jug with 2 gallons of water, and an empty 3-gallon jug.

8. Next logical step: Fill the 5-gallon jug from the external water source. This will add 3 gallons of water to the 5-gallon jug, bringing the total to 5 gallons.

We now have a 5-gallon jug with 5 gallons of water, a 3-gallon jug empty, and the 3-gallon jug is also empty.

Note: We're getting closer to measuring exactly 4 gallons.

9. Next logical step: Empty the 5-gallon jug, which is currently full. This will leave the 5-gallon jug empty and bring the total amount of water in the 5-gallon jug back to 0 gallons.

We now have a 5-gallon jug empty, a 3-gallon jug empty, and we still need to reach 4 gallons.

10. Next logical step: Transfer the 2 gallons from the 3-gallon jug into the 5-gallon jug, then fill the 3-gallon jug from the external water source.

We now have a 3-gallon jug being filled and a 5-gallon jug with 2 gallons of water.

### Final Answer:
Fill the 5-gallon jug, pour into the 3-gallon jug (leaving 2 gallons in the 5-gallon), empty the 3-gallon, transfer the 2 gallons to the 3-gallon, fill the 5-gallon again, pour from the 5-gallon into the 3-gallon until full (adding 1 gallon). Now the 5-gallon jug has exactly 4 gallons.
```

**Key Observations:**
- The engine parsed **10 step-by-step reasoning blocks** from the LLM's response
- Token usage: 4,875 prompt tokens + 797 response tokens = 5,672 total
- Inference latency: **3.1 seconds** (provider-dependent; would vary with OpenAI, Google, etc.)
- The LLM's chain-of-thought output is automatically extracted and structured into individual reasoning steps by the `ZeroShotCoTStrategy.ParseResponse()` method

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