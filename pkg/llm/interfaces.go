package llm

import (
	"context"
)

// Provider defines the interface that all LLM providers must implement.
type Provider interface {
	// Name returns the provider's name (e.g., "openai", "anthropic", "google").
	Name() string

	// Complete sends a completion request and returns the response.
	Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// CompleteStream sends a completion request and streams the response.
	CompleteStream(ctx context.Context, req *CompletionRequest) (<-chan GenerationChunk, error)

	// ListModels returns the available models for this provider.
	ListModels(ctx context.Context) ([]ModelInfo, error)

	// GetModelInfo returns information about a specific model.
	GetModelInfo(ctx context.Context, modelID string) (*ModelInfo, error)

	// SupportsMultimodal returns true if the provider supports multimodal inputs.
	SupportsMultimodal() bool

	// Close releases any resources held by the provider.
	Close() error
}

// ProviderFactory creates LLM providers.
type ProviderFactory interface {
	// Create creates a new provider instance.
	Create(name string, config *ProviderConfig) (Provider, error)

	// ListAvailable returns the names of available providers.
	ListAvailable() []string

	// Register registers a new provider constructor.
	Register(name string, constructor ProviderConstructor)
}

// ProviderConstructor is a function that creates an LLM provider.
type ProviderConstructor func(config *ProviderConfig) (Provider, error)
