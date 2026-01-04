// Package domain contains the core business models for the Logit Processor.
package domain

import (
	"math"
	"time"
)

// Logits represent raw model outputs before softmax.
// Typically a slice of float32 values representing token scores.
type Logits []float32

// Probabilities represent normalized probability distributions after softmax.
type Probabilities []float64

// TokenID represents a token identifier in the vocabulary.
type TokenID int

// TokenScore represents a token with its score/probability.
type TokenScore struct {
	Token TokenID   `json:"token"`
	Score float64   `json:"score"`
	Prob  float64   `json:"prob,omitempty"`
}

// SamplingMethod defines different token sampling strategies.
type SamplingMethod string

const (
	SamplingGreedy     SamplingMethod = "greedy"
	SamplingMultinomial SamplingMethod = "multinomial"
	SamplingTopK       SamplingMethod = "top_k"
	SamplingTopP       SamplingMethod = "top_p"
	SamplingBeamSearch SamplingMethod = "beam_search"
)

// SamplingConfig defines parameters for token sampling.
type SamplingConfig struct {
	Method      SamplingMethod `json:"method"`
	Temperature float64        `json:"temperature,omitempty"` // For multinomial sampling
	TopK        int            `json:"top_k,omitempty"`        // For top-k sampling
	TopP        float64        `json:"top_p,omitempty"`        // For nucleus (top-p) sampling
	BeamSize    int            `json:"beam_size,omitempty"`    // For beam search
	Length      int            `json:"length,omitempty"`       // Maximum sequence length
	MinLength   int            `json:"min_length,omitempty"`   // Minimum sequence length
	RepetitionPenalty float64  `json:"repetition_penalty,omitempty"` // Penalty for repeating tokens
}

// LogitRequest represents a request to process logits.
type LogitRequest struct {
	Logits   Logits         `json:"logits"`
	Config   SamplingConfig `json:"config"`
	Vocab    []string       `json:"vocab,omitempty"` // Optional vocabulary for debugging
	Context  []TokenID      `json:"context,omitempty"` // Previous tokens for repetition penalty
}

// LogitResponse represents the result of logit processing.
type LogitResponse struct {
	SelectedToken  TokenScore     `json:"selected_token"`
	TopTokens      []TokenScore   `json:"top_tokens"`
	Probabilities  Probabilities  `json:"probabilities"`
	RawLogits      Logits         `json:"raw_logits,omitempty"`
	Config         SamplingConfig `json:"config"`
	ProcessingTime time.Duration  `json:"processing_time"`
	Error          error          `json:"error,omitempty"`
}

// ProcessingResult represents the intermediate results of logit processing.
type ProcessingResult struct {
	FilteredLogits Logits        `json:"filtered_logits"`
	Probabilities  Probabilities `json:"probabilities"`
	ValidTokens    []TokenID     `json:"valid_tokens"`
}

// BeamCandidate represents a candidate sequence in beam search.
type BeamCandidate struct {
	Tokens       []TokenID `json:"tokens"`
	Score        float64   `json:"score"`
	Probability  float64   `json:"probability"`
}

// NewLogitRequest creates a new logit processing request.
func NewLogitRequest(logits Logits, config SamplingConfig) *LogitRequest {
	return &LogitRequest{
		Logits: logits,
		Config: config,
	}
}

// NewSamplingConfig creates a default sampling configuration.
func NewSamplingConfig(method SamplingMethod) SamplingConfig {
	config := SamplingConfig{
		Method:     method,
		Temperature: 1.0,
		TopK:       50,
		TopP:       0.9,
		BeamSize:   5,
		Length:     100,
		MinLength:  1,
		RepetitionPenalty: 1.0,
	}

	// Set defaults based on method
	switch method {
	case SamplingGreedy:
		config.Temperature = 0.0
	case SamplingTopK:
		config.TopK = 10
	case SamplingTopP:
		config.TopP = 0.9
	}

	return config
}

// Validate checks if the sampling configuration is valid.
func (c SamplingConfig) Validate() error {
	if c.Temperature < 0 {
		return NewValidationError("temperature must be non-negative")
	}
	if c.TopK < 0 {
		return NewValidationError("top_k must be non-negative")
	}
	if c.TopP < 0 || c.TopP > 1 {
		return NewValidationError("top_p must be between 0 and 1")
	}
	if c.BeamSize < 1 {
		return NewValidationError("beam_size must be at least 1")
	}
	if c.Length < 0 {
		return NewValidationError("length must be non-negative")
	}
	if c.MinLength < 0 {
		return NewValidationError("min_length must be non-negative")
	}
	if c.RepetitionPenalty <= 0 {
		return NewValidationError("repetition_penalty must be positive")
	}
	return nil
}

// NewValidationError creates a validation error.
func NewValidationError(message string) error {
	return &ValidationError{Message: message}
}

// ValidationError represents a configuration validation error.
type ValidationError struct {
	Message string
}

func (e *ValidationError) Error() string {
	return e.Message
}

// Softmax computes the softmax of the given logits.
func Softmax(logits Logits) Probabilities {
	if len(logits) == 0 {
		return Probabilities{}
	}

	// Find max logit for numerical stability
	maxLogit := float64(logits[0])
	for _, logit := range logits {
		if float64(logit) > maxLogit {
			maxLogit = float64(logit)
		}
	}

	// Compute exp(logit - max) and sum
	sum := 0.0
	expValues := make([]float64, len(logits))
	for i, logit := range logits {
		expValues[i] = math.Exp(float64(logit) - maxLogit)
		sum += expValues[i]
	}

	// Normalize to get probabilities
	probs := make(Probabilities, len(logits))
	for i, expVal := range expValues {
		probs[i] = expVal / sum
	}

	return probs
}

// ApplyTemperature applies temperature scaling to logits.
func ApplyTemperature(logits Logits, temperature float64) Logits {
	if temperature == 0.0 {
		// Greedy sampling - return original logits
		result := make(Logits, len(logits))
		copy(result, logits)
		return result
	}

	result := make(Logits, len(logits))
	for i, logit := range logits {
		result[i] = float32(float64(logit) / temperature)
	}
	return result
}
