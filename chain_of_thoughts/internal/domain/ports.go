// Package domain contains the port interfaces for the Chain of Thought framework.
package domain

import (
	"context"
	"io"
)

// LLMProvider defines the interface that all LLM providers must implement.
type LLMProvider interface {
	// Name returns the provider's name (e.g., "openai", "anthropic", "google").
	Name() string

	// Complete sends a completion request and returns the response.
	Complete(ctx context.Context, req *CoTRequest) (*CoTResponse, error)

	// CompleteStream sends a completion request and streams the response.
	CompleteStream(ctx context.Context, req *CoTRequest) (<-chan StreamChunk, error)

	// ListModels returns the available models for this provider.
	ListModels(ctx context.Context) ([]ModelInfo, error)

	// GetModelInfo returns information about a specific model.
	GetModelInfo(ctx context.Context, modelID string) (*ModelInfo, error)

	// SupportsMultimodal returns true if the provider supports multimodal inputs.
	SupportsMultimodal() bool

	// Close releases any resources held by the provider.
	Close() error
}

// StreamChunk represents a chunk of streamed response.
type StreamChunk struct {
	Content      string `json:"content"`
	IsThinking   bool   `json:"is_thinking"`   // True if this is reasoning content
	IsFinal      bool   `json:"is_final"`      // True if this is the final answer
	FinishReason string `json:"finish_reason"` // Why generation stopped
	Error        error  `json:"error,omitempty"`
}

// ChainOfThoughtEngine defines the interface for the CoT reasoning engine.
type ChainOfThoughtEngine interface {
	// Reason performs chain of thought reasoning on the given input.
	Reason(ctx context.Context, req *CoTRequest) (*CoTResponse, error)

	// ReasonStream performs chain of thought reasoning with streaming output.
	ReasonStream(ctx context.Context, req *CoTRequest) (<-chan StreamChunk, error)

	// SetProvider sets the LLM provider to use.
	SetProvider(provider LLMProvider)

	// GetProvider returns the current LLM provider.
	GetProvider() LLMProvider

	// SetPromptStrategy sets the prompting strategy for CoT.
	SetPromptStrategy(strategy PromptStrategy)
}

// PromptStrategy defines how to construct CoT prompts.
type PromptStrategy interface {
	// Name returns the strategy name.
	Name() string

	// BuildPrompt constructs the CoT prompt from the input messages.
	BuildPrompt(messages []Message) []Message

	// ParseResponse extracts the chain of thought from the raw response.
	ParseResponse(rawResponse string) (*ChainOfThought, error)
}

// ImageProcessor handles image processing for multimodal inputs.
type ImageProcessor interface {
	// LoadFromURL loads an image from a URL.
	LoadFromURL(ctx context.Context, url string) ([]byte, string, error)

	// LoadFromFile loads an image from a file path.
	LoadFromFile(path string) ([]byte, string, error)

	// LoadFromReader loads an image from an io.Reader.
	LoadFromReader(reader io.Reader) ([]byte, string, error)

	// ToBase64 converts image bytes to base64.
	ToBase64(data []byte) string

	// Resize resizes an image to fit within the given dimensions.
	Resize(data []byte, maxWidth, maxHeight int) ([]byte, error)

	// GetMimeType detects the MIME type of image data.
	GetMimeType(data []byte) string
}

// Logger defines the logging interface.
type Logger interface {
	Debug(msg string, args ...any)
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
}

// ProviderFactory creates LLM providers.
type ProviderFactory interface {
	// Create creates a new provider instance.
	Create(name string, config *ProviderConfig) (LLMProvider, error)

	// ListAvailable returns the names of available providers.
	ListAvailable() []string

	// Register registers a new provider constructor.
	Register(name string, constructor ProviderConstructor)
}

// ProviderConstructor is a function that creates an LLM provider.
type ProviderConstructor func(config *ProviderConfig) (LLMProvider, error)
