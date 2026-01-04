package core

import (
	"fmt"

	"github.com/tektwister/ai_engineering/logit_processor/internal/domain"
)

// SamplerFactory implements the domain.SamplerFactory interface.
type SamplerFactory struct{}

// NewSamplerFactory creates a new sampler factory.
func NewSamplerFactory() *SamplerFactory {
	return &SamplerFactory{}
}

// CreateSampler creates a sampler for the given method.
func (f *SamplerFactory) CreateSampler(method domain.SamplingMethod) (domain.Sampler, error) {
	switch method {
	case domain.SamplingGreedy:
		return &GreedySampler{}, nil
	case domain.SamplingMultinomial:
		return &MultinomialSampler{}, nil
	case domain.SamplingTopK:
		return &TopKSampler{}, nil
	case domain.SamplingTopP:
		return &TopPSampler{}, nil
	default:
		return nil, fmt.Errorf("unsupported sampling method: %s", method)
	}
}

// GetAvailableSamplers returns the list of available sampling methods.
func (f *SamplerFactory) GetAvailableSamplers() []domain.SamplingMethod {
	return []domain.SamplingMethod{
		domain.SamplingGreedy,
		domain.SamplingMultinomial,
		domain.SamplingTopK,
		domain.SamplingTopP,
	}
}

// CreateDefaultProcessor creates a processor with the default greedy sampler.
func CreateDefaultProcessor() *Processor {
	sampler := &GreedySampler{}
	return NewProcessorWithSampler(sampler)
}

// CreateProcessorWithMethod creates a processor with the specified sampling method.
func CreateProcessorWithMethod(method domain.SamplingMethod) (*Processor, error) {
	factory := NewSamplerFactory()
	sampler, err := factory.CreateSampler(method)
	if err != nil {
		return nil, err
	}
	return NewProcessorWithSampler(sampler), nil
}
