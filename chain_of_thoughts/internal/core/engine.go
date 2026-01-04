package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/tektwister/ai_engineering/chain-of-thoughts/internal/domain"
	"github.com/tektwister/ai_engineering/pkg/llm"
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

// Reason performs chain of thought reasoning on the given input using an explicit loop.
func (e *Engine) Reason(ctx context.Context, req *domain.CoTRequest) (*domain.CoTResponse, error) {
	if e.provider == nil {
		return nil, fmt.Errorf("no provider configured")
	}
	if e.strategy == nil {
		return nil, fmt.Errorf("no prompt strategy configured")
	}

	// 1. Initialize Chain
	chain := domain.ChainOfThought{
		Model:    req.Model,
		Provider: e.provider.Name(),
		Steps:    []domain.ThoughtStep{},
	}
	
	// Copy initial history
	history := make([]domain.Message, len(req.Messages))
	copy(history, req.Messages)
	
	// Extract Question for metadata
	for _, msg := range req.Messages {
		if msg.Role == domain.RoleUser {
			for _, c := range msg.Contents {
				if c.Type == domain.ContentTypeText {
					chain.Question = c.Text
					break
				}
			}
			break
		}
	}

	// 2. Initial Setup
	messages := e.strategy.BuildInitialPrompt(history)
	
	const MaxSteps = 10
	var finalAnswer string
	
	// 3. Reasoning Loop
	for stepCount := 1; stepCount <= MaxSteps; stepCount++ {
		// Construct prompt for NEXT step
		stepPrompt := e.strategy.BuildNextStepPrompt(messages, chain)
		
		// Create request
		providerReq := &llm.CompletionRequest{
			Messages:    stepPrompt,
			Model:       req.Model,
			MaxTokens:   req.MaxTokens,
			Temperature: req.Temperature,
			TopP:        req.TopP,
			Options:     req.Options,
		}

		// Call Provider
		resp, err := e.provider.Complete(ctx, providerReq)
		if err != nil {
			return nil, fmt.Errorf("provider error at step %d: %w", stepCount, err)
		}
		
		// Update Token Usage
		chain.TotalTokens += resp.Usage.TotalTokens
		chain.PromptTokens += resp.Usage.PromptTokens
		chain.ResponseTokens += resp.Usage.CompletionTokens

		// Parse Response
		content, isFinal, err := e.strategy.ParseStep(resp.Content)
		if err != nil {
			return nil, fmt.Errorf("strategy parse error at step %d: %w", stepCount, err)
		}
		
		// Append to history so model remembers what it just thought
		messages = append(messages, domain.NewTextMessage(domain.RoleAssistant, resp.Content))
		
		if isFinal {
			finalAnswer = content
			chain.FinalAnswer = finalAnswer
			break
		} else {
			// Add valid thought step to chain
			chain.AddStep(fmt.Sprintf("Step %d", stepCount), content)
		}
	}

	return &domain.CoTResponse{
		Chain: chain,
		RawContent: finalAnswer,
	}, nil
}

// ReasonStream performs chain of thought reasoning with streaming output.
func (e *Engine) ReasonStream(ctx context.Context, req *domain.CoTRequest) (<-chan domain.StreamChunk, error) {
	// Simple implementation for now: Wait for full response then stream it? 
	// Or implement complex streaming loop?
	// For "Mimic Implementation", let's keep it simple and just run sync for now or error.
	return nil, fmt.Errorf("streaming not yet implemented for explicit reasoning loop")
}

// --- Explicit Step Strategy ---

// ExplicitStepStrategy implements a programmatic loop strategy.
type ExplicitStepStrategy struct{}

func (s *ExplicitStepStrategy) Name() string {
	return "explicit_step_cot"
}

func (s *ExplicitStepStrategy) BuildInitialPrompt(messages []domain.Message) []domain.Message {
	// Start with system prompt defining the persona
	systemMsg := "You are a precise reasoning agent. You solve problems by generating one logical step at a time. Do not jump to the conclusion."
	
	msgs := []domain.Message{
		domain.NewTextMessage(domain.RoleSystem, systemMsg),
	}
	msgs = append(msgs, messages...)
	return msgs
}

func (s *ExplicitStepStrategy) BuildNextStepPrompt(history []domain.Message, chain domain.ChainOfThought) []domain.Message {
	// Prompt the model to generate the next step specifically
	var prompt string
	if len(chain.Steps) == 0 {
		prompt = "Please generate the first step of reasoning. Think about the problem structure."
	} else {
		prompt = fmt.Sprintf("You have completed %d steps. Please generate the next logical step. If you have enough information to answer, start your response with 'FINAL ANSWER:'.", len(chain.Steps))
	}
	
	msgs := make([]domain.Message, len(history))
	copy(msgs, history)
	msgs = append(msgs, domain.NewTextMessage(domain.RoleUser, prompt))
	
	return msgs
}

func (s *ExplicitStepStrategy) ParseStep(response string) (content string, isFinal bool, err error) {
	// improved parsing logic
	clean := strings.TrimSpace(response)
	
	if strings.HasPrefix(strings.ToUpper(clean), "FINAL ANSWER:") {
		// Found the answer
		answer := strings.TrimPrefix(clean, "FINAL ANSWER:")
		answer = strings.TrimPrefix(answer, "Final Answer:")
		return strings.TrimSpace(answer), true, nil
	}
	
	return clean, false, nil
}
