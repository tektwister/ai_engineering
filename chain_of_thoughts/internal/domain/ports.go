// Package domain contains the port interfaces for the Chain of Thought framework.
package domain

import (
	"context"
	"io"

	"github.com/tektwister/ai_engineering/pkg/llm"
)

// LLMProvider defines the interface that all LLM providers must implement.
// We alias it to the shared generic provider interface.
type LLMProvider = llm.Provider

// StreamChunk represents a chunk of response (aliased from generic).
// But wait, StreamChunk in domain had specific CoT fields like IsThinking.
// llm.GenerationChunk is generic.
// Should we extend it or keep StreamChunk separate?
// Original StreamChunk: Content, IsThinking, IsFinal, FinishReason, Error.
// llm.GenerationChunk: Content, IsFinal, FinishReason, Error.
// Missing IsThinking.
// If CoT engine needs IsThinking, it parses it from content or provider specific field?
// Providers like default don't have thinking.
// Let's redefine StreamChunk in domain as it might be specific to CoT usage (e.g. if we parse <thought> tags).

// StreamChunk represents a chunk of streamed response in CoT context.
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

	// BuildInitialPrompt constructs the initial prompt to start the reasoning process.
	BuildInitialPrompt(messages []Message) []Message

	// BuildNextStepPrompt constructs the prompt to ask for the next reasoning step.
	// It takes the conversation history including previous steps.
	BuildNextStepPrompt(history []Message, curentChain ChainOfThought) []Message

	// ParseStep parses a single response from the model into a thought step or final answer.
	// Returns the content, whether it is a final answer, and any error.
	ParseStep(response string) (content string, isFinal bool, err error)
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

// ProviderFactory creates LLM providers. (Aliased)
type ProviderFactory = llm.ProviderFactory

// ProviderConstructor is a function that creates an LLM provider. (Aliased)
type ProviderConstructor = llm.ProviderConstructor
