package core

import (
	"fmt"
	"math"
	"sort"

	"github.com/tektwister/ai_engineering/logit_processor/internal/domain"
)

// GreedySampler implements greedy (argmax) sampling.
type GreedySampler struct{}

func (s *GreedySampler) Name() string {
	return "greedy"
}

func (s *GreedySampler) Sample(logits domain.Logits, config domain.SamplingConfig) (*domain.ProcessingResult, error) {
	if len(logits) == 0 {
		return nil, fmt.Errorf("empty logits")
	}

	// Apply temperature (though for greedy it's typically 0)
	scaledLogits := domain.ApplyTemperature(logits, config.Temperature)

	// For greedy, all tokens are valid
	validTokens := make([]domain.TokenID, len(logits))
	for i := range validTokens {
		validTokens[i] = domain.TokenID(i)
	}

	return &domain.ProcessingResult{
		FilteredLogits: scaledLogits,
		Probabilities:  domain.Softmax(scaledLogits),
		ValidTokens:    validTokens,
	}, nil
}

func (s *GreedySampler) SampleMultiple(logits []domain.Logits, config domain.SamplingConfig) ([]*domain.ProcessingResult, error) {
	results := make([]*domain.ProcessingResult, len(logits))
	for i, logitSlice := range logits {
		result, err := s.Sample(logitSlice, config)
		if err != nil {
			return nil, fmt.Errorf("failed to sample batch %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// MultinomialSampler implements temperature-based multinomial sampling.
type MultinomialSampler struct{}

func (s *MultinomialSampler) Name() string {
	return "multinomial"
}

func (s *MultinomialSampler) Sample(logits domain.Logits, config domain.SamplingConfig) (*domain.ProcessingResult, error) {
	if len(logits) == 0 {
		return nil, fmt.Errorf("empty logits")
	}

	// Apply temperature scaling
	scaledLogits := domain.ApplyTemperature(logits, config.Temperature)

	// All tokens are valid for multinomial sampling
	validTokens := make([]domain.TokenID, len(logits))
	for i := range validTokens {
		validTokens[i] = domain.TokenID(i)
	}

	return &domain.ProcessingResult{
		FilteredLogits: scaledLogits,
		Probabilities:  domain.Softmax(scaledLogits),
		ValidTokens:    validTokens,
	}, nil
}

func (s *MultinomialSampler) SampleMultiple(logits []domain.Logits, config domain.SamplingConfig) ([]*domain.ProcessingResult, error) {
	results := make([]*domain.ProcessingResult, len(logits))
	for i, logitSlice := range logits {
		result, err := s.Sample(logitSlice, config)
		if err != nil {
			return nil, fmt.Errorf("failed to sample batch %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// TopKSampler implements top-k sampling.
type TopKSampler struct{}

func (s *TopKSampler) Name() string {
	return "top_k"
}

func (s *TopKSampler) Sample(logits domain.Logits, config domain.SamplingConfig) (*domain.ProcessingResult, error) {
	if len(logits) == 0 {
		return nil, fmt.Errorf("empty logits")
	}

	// Apply temperature scaling
	scaledLogits := domain.ApplyTemperature(logits, config.Temperature)

	// Apply top-k filtering
	filteredLogits, validTokens, err := s.filterTopK(scaledLogits, config.TopK)
	if err != nil {
		return nil, err
	}

	return &domain.ProcessingResult{
		FilteredLogits: filteredLogits,
		Probabilities:  domain.Softmax(filteredLogits),
		ValidTokens:    validTokens,
	}, nil
}

func (s *TopKSampler) SampleMultiple(logits []domain.Logits, config domain.SamplingConfig) ([]*domain.ProcessingResult, error) {
	results := make([]*domain.ProcessingResult, len(logits))
	for i, logitSlice := range logits {
		result, err := s.Sample(logitSlice, config)
		if err != nil {
			return nil, fmt.Errorf("failed to sample batch %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

func (s *TopKSampler) filterTopK(logits domain.Logits, k int) (domain.Logits, []domain.TokenID, error) {
	if k <= 0 {
		k = len(logits)
	}
	if k > len(logits) {
		k = len(logits)
	}

	// Create slice of token-score pairs
	type tokenScore struct {
		token domain.TokenID
		score float32
	}

	tokenScores := make([]tokenScore, len(logits))
	for i, score := range logits {
		tokenScores[i] = tokenScore{
			token: domain.TokenID(i),
			score: score,
		}
	}

	// Sort by score (descending)
	sort.Slice(tokenScores, func(i, j int) bool {
		return tokenScores[i].score > tokenScores[j].score
	})

	// Take top k
	filteredLogits := make(domain.Logits, len(logits))
	validTokens := make([]domain.TokenID, k)

	// Set top k tokens to their original scores
	for i := 0; i < k; i++ {
		token := tokenScores[i].token
		validTokens[i] = token
		filteredLogits[token] = logits[token]
	}

	// Set remaining tokens to -inf (effectively zero probability)
	for i := k; i < len(tokenScores); i++ {
		token := tokenScores[i].token
		filteredLogits[token] = float32(math.Inf(-1))
	}

	return filteredLogits, validTokens, nil
}

// TopPSampler implements nucleus (top-p) sampling.
type TopPSampler struct{}

func (s *TopPSampler) Name() string {
	return "top_p"
}

func (s *TopPSampler) Sample(logits domain.Logits, config domain.SamplingConfig) (*domain.ProcessingResult, error) {
	if len(logits) == 0 {
		return nil, fmt.Errorf("empty logits")
	}

	// Apply temperature scaling
	scaledLogits := domain.ApplyTemperature(logits, config.Temperature)

	// Apply top-p filtering
	filteredLogits, validTokens, err := s.filterTopP(scaledLogits, config.TopP)
	if err != nil {
		return nil, err
	}

	return &domain.ProcessingResult{
		FilteredLogits: filteredLogits,
		Probabilities:  domain.Softmax(filteredLogits),
		ValidTokens:    validTokens,
	}, nil
}

func (s *TopPSampler) SampleMultiple(logits []domain.Logits, config domain.SamplingConfig) ([]*domain.ProcessingResult, error) {
	results := make([]*domain.ProcessingResult, len(logits))
	for i, logitSlice := range logits {
		result, err := s.Sample(logitSlice, config)
		if err != nil {
			return nil, fmt.Errorf("failed to sample batch %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

func (s *TopPSampler) filterTopP(logits domain.Logits, p float64) (domain.Logits, []domain.TokenID, error) {
	probabilities := domain.Softmax(logits)

	// Create slice of token-probability pairs
	type tokenProb struct {
		token domain.TokenID
		prob  float64
	}

	tokenProbs := make([]tokenProb, len(probabilities))
	for i, prob := range probabilities {
		tokenProbs[i] = tokenProb{
			token: domain.TokenID(i),
			prob:  prob,
		}
	}

	// Sort by probability (descending)
	sort.Slice(tokenProbs, func(i, j int) bool {
		return tokenProbs[i].prob > tokenProbs[j].prob
	})

	// Find tokens that make up the top p probability mass
	filteredLogits := make(domain.Logits, len(logits))
	var validTokens []domain.TokenID
	cumProb := 0.0

	for _, tp := range tokenProbs {
		validTokens = append(validTokens, tp.token)
		filteredLogits[tp.token] = logits[tp.token]
		cumProb += tp.prob

		if cumProb >= p {
			break
		}
	}

	// Set remaining tokens to -inf
	for i := range logits {
		found := false
		for _, token := range validTokens {
			if int(token) == i {
				found = true
				break
			}
		}
		if !found {
			filteredLogits[i] = float32(math.Inf(-1))
		}
	}

	return filteredLogits, validTokens, nil
}
