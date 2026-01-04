// Package providers contains the provider factory for creating LLM providers.
package providers

import (
	"fmt"
	"sync"

	"github.com/tektwister/ai_engineering/pkg/llm"
	"github.com/tektwister/ai_engineering/pkg/llm/providers/google"
	"github.com/tektwister/ai_engineering/pkg/llm/providers/groq"
	"github.com/tektwister/ai_engineering/pkg/llm/providers/huggingface"
	"github.com/tektwister/ai_engineering/pkg/llm/providers/openai"
)

// Factory implements the ProviderFactory interface.
type Factory struct {
	mu           sync.RWMutex
	constructors map[string]llm.ProviderConstructor
}

// NewFactory creates a new provider factory with default providers registered.
func NewFactory() *Factory {
	f := &Factory{
		constructors: make(map[string]llm.ProviderConstructor),
	}

	// Register default providers
	f.Register("openai", func(config *llm.ProviderConfig) (llm.Provider, error) {
		return openai.New(config)
	})

	f.Register("huggingface", func(config *llm.ProviderConfig) (llm.Provider, error) {
		return huggingface.New(config)
	})

	f.Register("google", func(config *llm.ProviderConfig) (llm.Provider, error) {
		return google.New(config)
	})
	f.Register("gemini", func(config *llm.ProviderConfig) (llm.Provider, error) {
		return google.New(config) // Alias for google
	})

	f.Register("groq", func(config *llm.ProviderConfig) (llm.Provider, error) {
		return groq.New(config)
	})

	return f
}

// Create creates a new provider instance.
func (f *Factory) Create(name string, config *llm.ProviderConfig) (llm.Provider, error) {
	f.mu.RLock()
	constructor, ok := f.constructors[name]
	f.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", name)
	}

	return constructor(config)
}

// ListAvailable returns the names of available providers.
func (f *Factory) ListAvailable() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()

	names := make([]string, 0, len(f.constructors))
	for name := range f.constructors {
		names = append(names, name)
	}
	return names
}

// Register registers a new provider constructor.
func (f *Factory) Register(name string, constructor llm.ProviderConstructor) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.constructors[name] = constructor
}

// DefaultFactory is the global default provider factory.
var DefaultFactory = NewFactory()

// Create creates a provider using the default factory.
func Create(name string, config *llm.ProviderConfig) (llm.Provider, error) {
	return DefaultFactory.Create(name, config)
}

// ListAvailable lists available providers using the default factory.
func ListAvailable() []string {
	return DefaultFactory.ListAvailable()
}

// Register registers a provider with the default factory.
func Register(name string, constructor llm.ProviderConstructor) {
	DefaultFactory.Register(name, constructor)
}
