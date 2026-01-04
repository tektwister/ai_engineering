// Package anthropic provides an Anthropic (Claude) LLM provider implementation.
package anthropic

import (
	"context"
	"fmt"
	"strings"

	"github.com/chain-of-thoughts/internal/domain"
	"github.com/liushuangls/go-anthropic/v2"
)

// Provider implements the LLMProvider interface for Anthropic.
type Provider struct {
	client *anthropic.Client
	config *domain.ProviderConfig
}

// New creates a new Anthropic provider.
func New(config *domain.ProviderConfig) (*Provider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("Anthropic API key is required")
	}

	opts := []anthropic.ClientOption{}
	if config.BaseURL != "" {
		opts = append(opts, anthropic.WithBaseURL(config.BaseURL))
	}

	client := anthropic.NewClient(config.APIKey, opts...)

	return &Provider{
		client: client,
		config: config,
	}, nil
}

// Name returns the provider's name.
func (p *Provider) Name() string {
	return "anthropic"
}

// Complete sends a completion request and returns the response.
func (p *Provider) Complete(ctx context.Context, req *domain.CoTRequest) (*domain.CoTResponse, error) {
	messages, systemPrompt := p.convertMessages(req.Messages)

	model := req.Model
	if model == "" {
		model = anthropic.ModelClaude3Dot5SonnetLatest
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	temperature := float32(req.Temperature)
	if temperature == 0 {
		temperature = 0.7
	}

	msgReq := anthropic.MessageRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: &temperature,
	}

	if systemPrompt != "" {
		msgReq.System = systemPrompt
	}

	if req.TopP > 0 {
		topP := float32(req.TopP)
		msgReq.TopP = &topP
	}

	resp, err := p.client.CreateMessages(ctx, msgReq)
	if err != nil {
		return nil, fmt.Errorf("Anthropic completion failed: %w", err)
	}

	// Extract text content from response
	var content strings.Builder
	for _, block := range resp.Content {
		if block.Type == anthropic.ContentTypeText {
			content.WriteString(block.GetText())
		}
	}

	return &domain.CoTResponse{
		RawContent: content.String(),
		Chain: domain.ChainOfThought{
			Model:          model,
			Provider:       p.Name(),
			PromptTokens:   resp.Usage.InputTokens,
			ResponseTokens: resp.Usage.OutputTokens,
			TotalTokens:    resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}, nil
}

// CompleteStream sends a completion request and streams the response.
func (p *Provider) CompleteStream(ctx context.Context, req *domain.CoTRequest) (<-chan domain.StreamChunk, error) {
	messages, systemPrompt := p.convertMessages(req.Messages)

	model := req.Model
	if model == "" {
		model = anthropic.ModelClaude3Dot5SonnetLatest
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	temperature := float32(req.Temperature)
	if temperature == 0 {
		temperature = 0.7
	}

	msgReq := anthropic.MessageRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: &temperature,
	}

	if systemPrompt != "" {
		msgReq.System = systemPrompt
	}

	chunks := make(chan domain.StreamChunk)

	go func() {
		defer close(chunks)

		_, err := p.client.CreateMessagesStream(ctx, msgReq, func(event anthropic.StreamEvent) error {
			switch event.Type {
			case anthropic.StreamEventTypeContentBlockDelta:
				if event.Delta != nil && event.Delta.Type == anthropic.ContentTypeTextDelta {
					chunks <- domain.StreamChunk{
						Content: event.Delta.Text,
					}
				}
			case anthropic.StreamEventTypeMessageStop:
				chunks <- domain.StreamChunk{
					IsFinal:      true,
					FinishReason: "stop",
				}
			}
			return nil
		})

		if err != nil {
			chunks <- domain.StreamChunk{Error: err}
		}
	}()

	return chunks, nil
}

// ListModels returns the available models for this provider.
func (p *Provider) ListModels(ctx context.Context) ([]domain.ModelInfo, error) {
	// Anthropic doesn't have a list models API, return predefined list
	return []domain.ModelInfo{
		{
			ID:       anthropic.ModelClaude3Dot5SonnetLatest,
			Name:     "Claude 3.5 Sonnet",
			Provider: p.Name(),
			Capabilities: domain.ModelCapabilities{
				SupportsVision:    true,
				SupportsStreaming: true,
				MaxContextTokens:  200000,
				MaxOutputTokens:   8192,
			},
		},
		{
			ID:       anthropic.ModelClaude3Dot5HaikuLatest,
			Name:     "Claude 3.5 Haiku",
			Provider: p.Name(),
			Capabilities: domain.ModelCapabilities{
				SupportsVision:    true,
				SupportsStreaming: true,
				MaxContextTokens:  200000,
				MaxOutputTokens:   8192,
			},
		},
		{
			ID:       anthropic.ModelClaude3OpusLatest,
			Name:     "Claude 3 Opus",
			Provider: p.Name(),
			Capabilities: domain.ModelCapabilities{
				SupportsVision:    true,
				SupportsStreaming: true,
				MaxContextTokens:  200000,
				MaxOutputTokens:   4096,
			},
		},
	}, nil
}

// GetModelInfo returns information about a specific model.
func (p *Provider) GetModelInfo(ctx context.Context, modelID string) (*domain.ModelInfo, error) {
	models, _ := p.ListModels(ctx)
	for _, m := range models {
		if m.ID == modelID {
			return &m, nil
		}
	}
	return nil, fmt.Errorf("unknown model: %s", modelID)
}

// SupportsMultimodal returns true if the provider supports multimodal inputs.
func (p *Provider) SupportsMultimodal() bool {
	return true
}

// Close releases any resources held by the provider.
func (p *Provider) Close() error {
	return nil
}

// convertMessages converts domain messages to Anthropic messages.
// Returns the messages and any system prompt extracted.
func (p *Provider) convertMessages(messages []domain.Message) ([]anthropic.Message, string) {
	var systemPrompt string
	result := make([]anthropic.Message, 0, len(messages))

	for _, msg := range messages {
		// Extract system message
		if msg.Role == domain.RoleSystem {
			for _, content := range msg.Contents {
				if content.Type == domain.ContentTypeText {
					systemPrompt = content.Text
					break
				}
			}
			continue
		}

		var content []anthropic.MessageContent

		for _, c := range msg.Contents {
			switch c.Type {
			case domain.ContentTypeText:
				content = append(content, anthropic.NewTextMessageContent(c.Text))
			case domain.ContentTypeImage:
				var imageContent anthropic.MessageContent
				if c.ImageBase64 != "" {
					mimeType := c.MimeType
					if mimeType == "" {
						mimeType = "image/png"
					}
					imageContent = anthropic.NewImageMessageContent(
						anthropic.NewMessageContentSource(
							anthropic.ContentSourceTypeBase64,
							mimeType,
							c.ImageBase64,
						),
					)
				} else if c.ImageURL != "" {
					imageContent = anthropic.NewImageMessageContent(
						anthropic.NewMessageContentSource(
							anthropic.ContentSourceTypeURL,
							"",
							c.ImageURL,
						),
					)
				}
				if imageContent.Type != "" {
					content = append(content, imageContent)
				}
			}
		}

		role := anthropic.RoleUser
		if msg.Role == domain.RoleAssistant {
			role = anthropic.RoleAssistant
		}

		result = append(result, anthropic.Message{
			Role:    role,
			Content: content,
		})
	}

	return result, systemPrompt
}
