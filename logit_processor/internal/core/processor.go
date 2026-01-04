package core

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/tektwister/ai_engineering/logit_processor/internal/domain"
)

// Processor implements the LogitProcessor interface.
type Processor struct {
	sampler domain.Sampler
	logger  domain.Logger
	metrics domain.MetricsRecorder
}

// NewProcessor creates a new logit processor with default sampler.
func NewProcessor() *Processor {
	return &Processor{
		sampler: &GreedySampler{},
	}
}

// NewProcessorWithSampler creates a new processor with the specified sampler.
func NewProcessorWithSampler(sampler domain.Sampler) *Processor {
	return &Processor{
		sampler: sampler,
	}
}

// SetLogger sets the logger for the processor.
func (p *Processor) SetLogger(logger domain.Logger) {
	p.logger = logger
}

// SetMetricsRecorder sets the metrics recorder for the processor.
func (p *Processor) SetMetricsRecorder(metrics domain.MetricsRecorder) {
	p.metrics = metrics
}

// Process processes raw logits according to the sampling configuration.
func (p *Processor) Process(ctx context.Context, req *domain.LogitRequest) (*domain.LogitResponse, error) {
	startTime := time.Now()

	// Validate configuration
	if err := req.Config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	// Process logits using the sampler
	result, err := p.sampler.Sample(req.Logits, req.Config)
	if err != nil {
		if p.metrics != nil {
			p.metrics.RecordError(string(req.Config.Method), err)
		}
		return nil, fmt.Errorf("sampling failed: %w", err)
	}

	// Compute probabilities from filtered logits
	probabilities := domain.Softmax(result.FilteredLogits)

	// Select token based on method
	selectedToken, err := p.selectToken(result, probabilities, req.Config)
	if err != nil {
		return nil, fmt.Errorf("token selection failed: %w", err)
	}

	// Create top tokens list (top 10 by probability)
	topTokens := p.getTopTokens(probabilities, 10)

	processingTime := time.Since(startTime)

	response := &domain.LogitResponse{
		SelectedToken:  selectedToken,
		TopTokens:      topTokens,
		Probabilities:  probabilities,
		RawLogits:      req.Logits,
		Config:         req.Config,
		ProcessingTime: processingTime,
	}

	// Record metrics
	if p.metrics != nil {
		p.metrics.RecordProcessingTime(processingTime.Seconds(), string(req.Config.Method))
		p.metrics.RecordSamplingStats(
			string(req.Config.Method),
			1,
			req.Config.TopK,
			req.Config.TopP,
		)
	}

	if p.logger != nil {
		p.logger.Debug("Processed logits",
			"method", req.Config.Method,
			"selected_token", selectedToken.Token,
			"processing_time", processingTime)
	}

	return response, nil
}

// ProcessBatch processes multiple logit requests in batch.
func (p *Processor) ProcessBatch(ctx context.Context, reqs []*domain.LogitRequest) ([]*domain.LogitResponse, error) {
	responses := make([]*domain.LogitResponse, len(reqs))

	for i, req := range reqs {
		response, err := p.Process(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("batch processing failed at request %d: %w", i, err)
		}
		responses[i] = response
	}

	return responses, nil
}

// SetSampler sets the sampling strategy to use.
func (p *Processor) SetSampler(sampler domain.Sampler) {
	p.sampler = sampler
}

// GetSampler returns the current sampling strategy.
func (p *Processor) GetSampler() domain.Sampler {
	return p.sampler
}

// selectToken selects a token based on the sampling method and results.
func (p *Processor) selectToken(result *domain.ProcessingResult, probabilities domain.Probabilities, config domain.SamplingConfig) (domain.TokenScore, error) {
	switch config.Method {
	case domain.SamplingGreedy:
		return p.selectGreedy(result, probabilities)
	case domain.SamplingMultinomial:
		return p.selectMultinomial(result, probabilities)
	case domain.SamplingTopK, domain.SamplingTopP:
		return p.selectFromFiltered(result, probabilities)
	default:
		return domain.TokenScore{}, fmt.Errorf("unsupported sampling method: %s", config.Method)
	}
}

// selectGreedy selects the token with highest probability.
func (p *Processor) selectGreedy(result *domain.ProcessingResult, probabilities domain.Probabilities) (domain.TokenScore, error) {
	if len(result.ValidTokens) == 0 {
		return domain.TokenScore{}, fmt.Errorf("no valid tokens available")
	}

	maxProb := 0.0
	selectedToken := result.ValidTokens[0]

	for _, token := range result.ValidTokens {
		if int(token) >= len(probabilities) {
			continue
		}
		if probabilities[token] > maxProb {
			maxProb = probabilities[token]
			selectedToken = token
		}
	}

	return domain.TokenScore{
		Token: selectedToken,
		Score: probabilities[selectedToken],
		Prob:  probabilities[selectedToken],
	}, nil
}

// selectMultinomial performs multinomial sampling.
func (p *Processor) selectMultinomial(result *domain.ProcessingResult, probabilities domain.Probabilities) (domain.TokenScore, error) {
	if len(result.ValidTokens) == 0 {
		return domain.TokenScore{}, fmt.Errorf("no valid tokens available")
	}

	// Create cumulative distribution
	cumProbs := make([]float64, len(result.ValidTokens))
	totalProb := 0.0

	for i, token := range result.ValidTokens {
		if int(token) < len(probabilities) {
			totalProb += probabilities[token]
			cumProbs[i] = totalProb
		}
	}

	// Sample from cumulative distribution
	r := rand.Float64() * totalProb
	for i, cumProb := range cumProbs {
		if r <= cumProb {
			token := result.ValidTokens[i]
			return domain.TokenScore{
				Token: token,
				Score: probabilities[token],
				Prob:  probabilities[token],
			}, nil
		}
	}

	// Fallback to first token
	token := result.ValidTokens[0]
	return domain.TokenScore{
		Token: token,
		Score: probabilities[token],
		Prob:  probabilities[token],
	}, nil
}

// selectFromFiltered selects from filtered tokens (for top-k/top-p).
func (p *Processor) selectFromFiltered(result *domain.ProcessingResult, probabilities domain.Probabilities) (domain.TokenScore, error) {
	return p.selectMultinomial(result, probabilities)
}

// getTopTokens returns the top N tokens by probability.
func (p *Processor) getTopTokens(probabilities domain.Probabilities, n int) []domain.TokenScore {
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

	// Take top N
	count := int(math.Min(float64(n), float64(len(tokenProbs))))
	topTokens := make([]domain.TokenScore, count)

	for i := 0; i < count; i++ {
		tp := tokenProbs[i]
		topTokens[i] = domain.TokenScore{
			Token: tp.token,
			Score: tp.prob,
			Prob:  tp.prob,
		}
	}

	return topTokens
}
