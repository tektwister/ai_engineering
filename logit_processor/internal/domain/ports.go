// Package domain contains the port interfaces for the Logit Processor.
package domain

import (
	"context"
)

// LogitProcessor defines the main interface for logit processing.
type LogitProcessor interface {
	// Process processes raw logits according to the sampling configuration.
	Process(ctx context.Context, req *LogitRequest) (*LogitResponse, error)

	// ProcessBatch processes multiple logit requests in batch.
	ProcessBatch(ctx context.Context, reqs []*LogitRequest) ([]*LogitResponse, error)

	// SetSampler sets the sampling strategy to use.
	SetSampler(sampler Sampler)

	// GetSampler returns the current sampling strategy.
	GetSampler() Sampler
}

// Sampler defines the interface for token sampling strategies.
type Sampler interface {
	// Name returns the name of the sampling strategy.
	Name() string

	// Sample selects a token from the given logits according to the sampling method.
	Sample(logits Logits, config SamplingConfig) (*ProcessingResult, error)

	// SampleMultiple samples multiple tokens for batch processing.
	SampleMultiple(logits []Logits, config SamplingConfig) ([]*ProcessingResult, error)
}

// LogitFilter defines the interface for logit filtering operations.
type LogitFilter interface {
	// Name returns the name of the filter.
	Name() string

	// Filter applies filtering to logits (e.g., top-k, top-p).
	Filter(logits Logits, config SamplingConfig) (Logits, []TokenID, error)

	// FilterBatch applies filtering to multiple logit arrays.
	FilterBatch(logits []Logits, config SamplingConfig) ([]Logits, [][]TokenID, error)
}

// ProbabilityProcessor defines the interface for probability computations.
type ProbabilityProcessor interface {
	// ComputeProbabilities converts logits to probabilities using softmax.
	ComputeProbabilities(logits Logits) Probabilities

	// ComputeProbabilitiesBatch converts multiple logit arrays to probabilities.
	ComputeProbabilitiesBatch(logits []Logits) []Probabilities
}

// TemperatureScaler defines the interface for temperature scaling.
type TemperatureScaler interface {
	// Scale applies temperature scaling to logits.
	Scale(logits Logits, temperature float64) Logits

	// ScaleBatch applies temperature scaling to multiple logit arrays.
	ScaleBatch(logits []Logits, temperature float64) []Logits
}

// RepetitionPenalty defines the interface for repetition penalty application.
type RepetitionPenalty interface {
	// Apply applies repetition penalty to logits based on previous tokens.
	Apply(logits Logits, previousTokens []TokenID, penalty float64) Logits

	// ApplyBatch applies repetition penalty to multiple logit arrays.
	ApplyBatch(logits []Logits, previousTokens [][]TokenID, penalty float64) []Logits
}

// BeamSearcher defines the interface for beam search operations.
type BeamSearcher interface {
	// Search performs beam search on the given logits.
	Search(logits []Logits, config SamplingConfig) ([]BeamCandidate, error)
}

// Logger defines the logging interface.
type Logger interface {
	Debug(msg string, args ...any)
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
}

// MetricsRecorder defines the interface for recording processing metrics.
type MetricsRecorder interface {
	// RecordProcessingTime records the time taken for logit processing.
	RecordProcessingTime(duration float64, method string)

	// RecordSamplingStats records statistics about sampling operations.
	RecordSamplingStats(method string, tokensSampled int, topKUsed int, topPUsed float64)

	// RecordError records processing errors.
	RecordError(method string, err error)
}

// SamplerFactory creates samplers based on sampling method.
type SamplerFactory interface {
	// CreateSampler creates a sampler for the given method.
	CreateSampler(method SamplingMethod) (Sampler, error)

	// GetAvailableSamplers returns the list of available sampling methods.
	GetAvailableSamplers() []SamplingMethod
}

// FilterFactory creates logit filters based on filtering method.
type FilterFactory interface {
	// CreateFilter creates a filter for the given method.
	CreateFilter(method SamplingMethod) (LogitFilter, error)

	// GetAvailableFilters returns the list of available filtering methods.
	GetAvailableFilters() []SamplingMethod
}
