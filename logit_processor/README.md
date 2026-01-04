# Logit Processor in Go

A high-performance logit processing library in Go that implements various token sampling strategies for language models. This module handles the conversion of raw model logits to token selections using techniques like temperature scaling, top-k filtering, nucleus (top-p) sampling, and greedy selection.

## About `logit_processor`

The `logit_processor` module provides a clean, extensible framework for processing model outputs (logits) into token selections. It implements the hexagonal architecture pattern, separating domain logic from implementation details, making it easy to extend with new sampling strategies.

Key features:
- **Multiple Sampling Methods**: Greedy, multinomial, top-k, and top-p sampling
- **Temperature Scaling**: Configurable temperature for controlling randomness
- **Batch Processing**: Efficient processing of multiple logit arrays
- **Extensible Design**: Easy to add new sampling strategies
- **Performance Optimized**: Efficient algorithms for large vocabularies
- **Hexagonal Architecture**: Clean separation of concerns

## Project Structure

```
logit_processor/
├── cmd/
│   └── main.go              # Example runner demonstrating all sampling methods
├── internal/
│   ├── core/
│   │   ├── processor.go     # Main processor implementation
│   │   ├── samplers.go      # Individual sampling strategy implementations
│   │   └── factory.go       # Factory for creating samplers
│   └── domain/
│       ├── models.go        # Domain models (Logits, Probabilities, SamplingConfig, etc.)
│       └── ports.go         # Interface definitions
├── go.mod                   # Go module definition
└── README.md               # This file
```

## Architecture Overview

### Design Pattern: Hexagonal Architecture

The `logit_processor` module follows the **Hexagonal Architecture** pattern:

```
┌─────────────────────────────────────────────────────────┐
│                    External World                        │
│  (Language Models, Applications)                       │
└─────────────────────────────────────────────────────────┘
                           ↕
                    ┌──────────────┐
                    │   Adapters   │
                    │  (Samplers)  │
                    └──────────────┘
                           ↕
                    ┌──────────────┐
                    │    Ports     │
                    │(Interfaces)  │
                    └──────────────┘
                           ↕
            ┌─────────────────────────────────────┐
            │         Domain Layer               │
            │   (Models, Business Logic)         │
            └─────────────────────────────────────┘
                           ↕
            ┌─────────────────────────────────────┐
            │      Application Layer             │
            │  (Processor, Orchestration)        │
            └─────────────────────────────────────┘
```

### Domain Layer (`internal/domain/`)

**Models:**
- `Logits`: Raw model outputs (float32 slice)
- `Probabilities`: Normalized probability distribution after softmax
- `TokenID`: Token identifier type
- `TokenScore`: Token with its score and probability
- `SamplingConfig`: Configuration for sampling parameters
- `LogitRequest`: Input request containing logits and config
- `LogitResponse`: Processing result with selected token and metadata

**Ports (Interfaces):**
- `LogitProcessor`: Main interface for logit processing
- `Sampler`: Interface for different sampling strategies
- `LogitFilter`: Interface for logit filtering operations
- `ProbabilityProcessor`: Interface for probability computations

### Application Layer (`internal/core/`)

- **Processor**: Main orchestrator implementing `LogitProcessor`
- **Samplers**: Individual sampling implementations:
  - `GreedySampler`: Always selects highest probability token
  - `MultinomialSampler`: Temperature-based random sampling
  - `TopKSampler`: Keeps only top-k highest probability tokens
  - `TopPSampler`: Nucleus sampling keeping tokens until cumulative probability reaches p
- **Factory**: Creates appropriate samplers based on configuration

## Sampling Methods

### Greedy Sampling
Selects the token with the highest probability. Deterministic and fast.

```go
config := domain.NewSamplingConfig(domain.SamplingGreedy)
```

### Multinomial Sampling
Samples from the probability distribution with optional temperature scaling.

```go
config := domain.SamplingConfig{
    Method:      domain.SamplingMultinomial,
    Temperature: 0.8,  // Lower = more deterministic, higher = more random
}
```

### Top-K Sampling
Keeps only the top K tokens by probability, then samples from those.

```go
config := domain.SamplingConfig{
    Method:      domain.SamplingTopK,
    TopK:        10,   // Keep top 10 tokens
    Temperature: 0.9,
}
```

### Top-P (Nucleus) Sampling
Keeps the smallest set of tokens whose cumulative probability reaches P, then samples from those.

```go
config := domain.SamplingConfig{
    Method:      domain.SamplingTopP,
    TopP:        0.9,  // Keep tokens until 90% cumulative probability
    Temperature: 0.8,
}
```

## Usage

### Basic Usage

```go
import (
    "context"
    "github.com/tektwister/ai_engineering/logit_processor/internal/core"
    "github.com/tektwister/ai_engineering/logit_processor/internal/domain"
)

// Create logits (normally from your model)
logits := domain.Logits{1.0, 2.0, 0.5, -1.0}

// Create sampling configuration
config := domain.SamplingConfig{
    Method:      domain.SamplingTopK,
    Temperature: 0.8,
    TopK:        2,
}

// Create request
req := domain.NewLogitRequest(logits, config)

// Create processor
processor := core.CreateDefaultProcessor()

// Process logits
ctx := context.Background()
response, err := processor.Process(ctx, req)
if err != nil {
    log.Fatal(err)
}

// Get selected token
selectedToken := response.SelectedToken.Token
probability := response.SelectedToken.Prob

fmt.Printf("Selected token: %d (probability: %.4f)\n", selectedToken, probability)
```

### Using Different Sampling Methods

```go
// Greedy sampling (deterministic)
processor, _ := core.CreateProcessorWithMethod(domain.SamplingGreedy)

// Top-p sampling with temperature
config := domain.SamplingConfig{
    Method:      domain.SamplingTopP,
    Temperature: 0.7,
    TopP:        0.9,
}
req := domain.NewLogitRequest(logits, config)

// Change sampler dynamically
processor.SetSampler(&core.TopPSampler{})
```

### Batch Processing

```go
// Create multiple requests
requests := []*domain.LogitRequest{req1, req2, req3}

// Process all at once
responses, err := processor.ProcessBatch(ctx, requests)
if err != nil {
    log.Fatal(err)
}

for i, response := range responses {
    fmt.Printf("Batch %d: Token %d (prob: %.4f)\n",
        i, response.SelectedToken.Token, response.SelectedToken.Prob)
}
```

### Custom Sampler Implementation

```go
// Implement the Sampler interface
type CustomSampler struct{}

func (s *CustomSampler) Name() string {
    return "custom"
}

func (s *CustomSampler) Sample(logits domain.Logits, config domain.SamplingConfig) (*domain.ProcessingResult, error) {
    // Your custom sampling logic here
    // Apply filters, temperature, etc.
    return &domain.ProcessingResult{
        FilteredLogits: filtered,
        Probabilities:  probs,
        ValidTokens:    valid,
    }, nil
}

func (s *CustomSampler) SampleMultiple(logits []domain.Logits, config domain.SamplingConfig) ([]*domain.ProcessingResult, error) {
    // Batch processing implementation
}

// Use custom sampler
processor.SetSampler(&CustomSampler{})
```

## Running the Example

```bash
cd logit_processor
go run cmd/main.go
```

This will demonstrate all sampling methods with sample logits and show:
- Selected tokens for each method
- Processing times
- Top token probabilities
- Distribution entropy
- Batch processing example

## Sample Output

```
=== Logit Processor Demo ===

--- greedy Sampling ---
Selected Token: 1 (Prob: 0.4226)
Processing Time: 45.2µs
Top 5 Tokens:
  1: 0.4226
  0: 0.2897
  2: 0.1448
  3: 0.0755
  4: 0.0674
Distribution Entropy: 6.2472
Config: Method=greedy, Temp=0.00, TopK=50, TopP=0.90

--- multinomial Sampling ---
Selected Token: 0 (Prob: 0.2897)
Processing Time: 52.8µs
...

--- top_k Sampling ---
Selected Token: 2 (Prob: 0.3333)
Processing Time: 48.1µs
...

--- top_p Sampling ---
Selected Token: 1 (Prob: 0.4226)
Processing Time: 49.7µs
...
```

## Performance Characteristics

- **Greedy**: Fastest, deterministic, O(n) where n is vocabulary size
- **Multinomial**: Fast, random sampling, O(n) with softmax computation
- **Top-K**: Moderate speed, requires sorting, O(n log k)
- **Top-P**: Variable speed, requires sorting and cumulative sum, O(n log n) worst case

## Extending the Library

### Adding a New Sampling Method

1. **Implement the Sampler interface**:
```go
type NewSampler struct{}

func (s *NewSampler) Name() string { return "new_method" }

func (s *NewSampler) Sample(logits domain.Logits, config domain.SamplingConfig) (*domain.ProcessingResult, error) {
    // Implementation
}
```

2. **Add to factory**:
```go
func (f *SamplerFactory) CreateSampler(method domain.SamplingMethod) (domain.Sampler, error) {
    switch method {
    case domain.SamplingNewMethod:
        return &NewSampler{}, nil
    // ... existing cases
    }
}
```

3. **Add method constant**:
```go
const SamplingNewMethod SamplingMethod = "new_method"
```

### Adding Configuration Options

Extend `SamplingConfig` with new fields and update validation:

```go
type SamplingConfig struct {
    // ... existing fields
    NewParameter float64 `json:"new_parameter"`
}

func (c SamplingConfig) Validate() error {
    // Add validation for new parameter
    if c.NewParameter < 0 {
        return NewValidationError("new_parameter must be non-negative")
    }
    return nil
}
```

## Integration with Language Models

This logit processor is designed to work with any language model that outputs logits. Here's how to integrate it:

```go
// After getting logits from your model
modelLogits := model.GenerateLogits(input)

// Create logit processor request
config := domain.SamplingConfig{
    Method:      domain.SamplingTopP,
    Temperature: 0.8,
    TopP:        0.9,
}

req := domain.NewLogitRequest(modelLogits, config)

// Process and get next token
processor := core.CreateDefaultProcessor()
response, _ := processor.Process(ctx, req)

nextToken := response.SelectedToken.Token
// Use nextToken for generation continuation
```

## Dependencies

- Go 1.20+
- Standard library only (no external dependencies)

## License

This module follows the same license as the parent ai_engineering project.
