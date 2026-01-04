package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/chain-of-thoughts/internal/domain"
)

// Engine implements the ChainOfThoughtEngine interface.
type Engine struct {
	provider domain.LLMProvider
	strategy domain.PromptStrategy
}

// NewEngine creates a new CoT engine.
func NewEngine(provider domain.LLMProvider, strategy domain.PromptStrategy) *Engine {
	return &Engine{
		provider: provider,
		strategy: strategy,
	}
}

// SetProvider sets the LLM provider to use.
func (e *Engine) SetProvider(provider domain.LLMProvider) {
	e.provider = provider
}

// GetProvider returns the current LLM provider.
func (e *Engine) GetProvider() domain.LLMProvider {
	return e.provider
}

// SetPromptStrategy sets the prompting strategy.
func (e *Engine) SetPromptStrategy(strategy domain.PromptStrategy) {
	e.strategy = strategy
}

// Reason performs chain of thought reasoning on the given input.
func (e *Engine) Reason(ctx context.Context, req *domain.CoTRequest) (*domain.CoTResponse, error) {
	if e.provider == nil {
		return nil, fmt.Errorf("no provider configured")
	}
	if e.strategy == nil {
		return nil, fmt.Errorf("no prompt strategy configured")
	}

	// 1. Build the prompt using the strategy
	messages := e.strategy.BuildPrompt(req.Messages)

	// 2. Create the request for the provider
	providerReq := &domain.CoTRequest{
		Messages:    messages,
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Options:     req.Options,
	}

	// 3. Call the provider
	resp, err := e.provider.Complete(ctx, providerReq)
	if err != nil {
		return nil, err
	}

	// 4. Parse the response using the strategy
	cot, err := e.strategy.ParseResponse(resp.RawContent)
	if err != nil {
		// If parsing fails, we still return the raw content but with an error note
		// This allows the user to see what the model output was even if it didn't match the format
		resp.Chain = domain.ChainOfThought{
			Model:       resp.Chain.Model,
			Provider:    resp.Chain.Provider,
			FinalAnswer: resp.RawContent,
		}
		return resp, fmt.Errorf("failed to parse CoT response: %w", err)
	}

	// Merge token usage stats
	cot.TotalTokens = resp.Chain.TotalTokens
	cot.PromptTokens = resp.Chain.PromptTokens
	cot.ResponseTokens = resp.Chain.ResponseTokens
	cot.Model = resp.Chain.Model
	cot.Provider = resp.Chain.Provider

	// Copy the question from the first user message if available
	for _, msg := range req.Messages {
		if msg.Role == domain.RoleUser {
			for _, c := range msg.Contents {
				if c.Type == domain.ContentTypeText {
					cot.Question = c.Text
					break
				}
			}
			break
		}
	}

	resp.Chain = *cot
	return resp, nil
}

// ReasonStream performs chain of thought reasoning with streaming output.
func (e *Engine) ReasonStream(ctx context.Context, req *domain.CoTRequest) (<-chan domain.StreamChunk, error) {
	if e.provider == nil {
		return nil, fmt.Errorf("no provider configured")
	}
	if e.strategy == nil {
		return nil, fmt.Errorf("no prompt strategy configured")
	}

	messages := e.strategy.BuildPrompt(req.Messages)

	providerReq := &domain.CoTRequest{
		Messages:    messages,
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Options:     req.Options,
	}

	// For now, we simply pass through the stream from the provider.
	// In a more advanced implementation, we could try to parse the stream in real-time
	// to identify reasoning steps vs final answer.
	return e.provider.CompleteStream(ctx, providerReq)
}

// --- Default Strategy ---

// ZeroShotCoTStrategy implements the "Let's think step by step" strategy.
type ZeroShotCoTStrategy struct{}

func (s *ZeroShotCoTStrategy) Name() string {
	return "zero_shot_cot"
}

func (s *ZeroShotCoTStrategy) BuildPrompt(messages []domain.Message) []domain.Message {
	// Deep copy messages to avoid side effects
	newMessages := make([]domain.Message, len(messages))
	copy(newMessages, messages)

	// In a real Zero-Shot CoT, we might append "Let's think step by step" to the last user message
	// or add it as a system prompt. Here we'll add a system prompt if one doesn't exist,
	// or prepend to the existing one.
	
	systemMsg := "You are a helpful AI assistant that solves problems by thinking step by step. " +
		"When you answer, format your response as follows:\n\n" +
		"### Reasoning\n" +
		"1. [First step...]\n" +
		"2. [Second step...]\n\n" +
		"### Answer\n" +
		"[Your final answer here]"

	// Prepend system message
	msgs := []domain.Message{
		domain.NewTextMessage(domain.RoleSystem, systemMsg),
	}
	msgs = append(msgs, newMessages...)
	
	return msgs
}

func (s *ZeroShotCoTStrategy) ParseResponse(rawResponse string) (*domain.ChainOfThought, error) {
	cot := &domain.ChainOfThought{
		Steps: []domain.ThoughtStep{},
	}

	// Simple fuzzy parsing based on the requested format
	parts := strings.Split(rawResponse, "### Answer")
	if len(parts) < 2 {
		// Fallback: entire response is the answer if we can't find the separator
		cot.FinalAnswer = rawResponse
		return cot, nil
	}

	reasoningPart := parts[0]
	answerPart := parts[1]

	// Extract steps from reasoning part
	reasoningPart = strings.TrimPrefix(reasoningPart, "### Reasoning")
	lines := strings.Split(reasoningPart, "\n")
	
	stepNum := 1
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		// Remove numbering like "1. " or "2."
		cleanLine := line
		if idx := strings.Index(line, "."); idx > 0 && idx < 5 { // simple check for numbering
			cleanLine = strings.TrimSpace(line[idx+1:])
		}

		cot.AddStep(fmt.Sprintf("Step %d", stepNum), cleanLine)
		stepNum++
	}

	cot.FinalAnswer = strings.TrimSpace(answerPart)

	return cot, nil
}
