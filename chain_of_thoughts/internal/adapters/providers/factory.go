// Package providers contains the provider factory for creating LLM providers.
package providers

import (
	"fmt"
	"sync"

	"github.com/chain-of-thoughts/internal/adapters/providers/anthropic"
	"github.com/chain-of-thoughts/internal/adapters/providers/google"
	"github.com/chain-of-thoughts/internal/adapters/providers/openai"
	"github.com/chain-of-thoughts/internal/domain"
)

// Factory implements the ProviderFactory interface.
type Factory struct {
	mu           sync.RWMutex
	constructors map[string]domain.ProviderConstructor
}

// NewFactory creates a new provider factory with default providers registered.
func NewFactory() *Factory {
	f := &Factory{
		constructors: make(map[string]domain.ProviderConstructor),
	}

	// Register default providers
	f.Register("openai", func(config *domain.ProviderConfig) (domain.LLMProvider, error) {
		return openai.New(config)
	})
	f.Register("anthropic", func(config *domain.ProviderConfig) (domain.LLMProvider, error) {
		return anthropic.New(config)
	})
	f.Register("google", func(config *domain.ProviderConfig) (domain.LLMProvider, error) {
		return google.New(config)
	})
	f.Register("gemini", func(config *domain.ProviderConfig) (domain.LLMProvider, error) {
		return google.New(config) // Alias for google
	})

	return f
}

// Create creates a new provider instance.
func (f *Factory) Create(name string, config *domain.ProviderConfig) (domain.LLMProvider, error) {
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
func (f *Factory) Register(name string, constructor domain.ProviderConstructor) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.constructors[name] = constructor
}

// DefaultFactory is the global default provider factory.
var DefaultFactory = NewFactory()

// Create creates a provider using the default factory.
func Create(name string, config *domain.ProviderConfig) (domain.LLMProvider, error) {
	return DefaultFactory.Create(name, config)
}

// ListAvailable lists available providers using the default factory.
func ListAvailable() []string {
	return DefaultFactory.ListAvailable()
}

// Register registers a provider with the default factory.
func Register(name string, constructor domain.ProviderConstructor) {
	DefaultFactory.Register(name, constructor)
}
