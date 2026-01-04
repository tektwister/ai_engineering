// Package groq provides a Groq LLM provider implementation.
package groq

import (
	"context"
	"fmt"
	"strings"

	openai "github.com/sashabaranov/go-openai"
	"github.com/tektwister/ai_engineering/pkg/llm"
)

const DefaultBaseURL = "https://api.groq.com/openai/v1"

// Provider implements the Provider interface for Groq.
type Provider struct {
	client *openai.Client
	config *llm.ProviderConfig
}

// New creates a new Groq provider.
func New(config *llm.ProviderConfig) (*Provider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("Groq API key is required")
	}

	clientConfig := openai.DefaultConfig(config.APIKey)

	// Set Groq Base URL
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
	} else {
		clientConfig.BaseURL = DefaultBaseURL
	}

	if config.OrgID != "" {
		clientConfig.OrgID = config.OrgID
	}

	client := openai.NewClientWithConfig(clientConfig)

	return &Provider{
		client: client,
		config: config,
	}, nil
}

// Name returns the provider's name.
func (p *Provider) Name() string {
	return "groq"
}

// Complete sends a completion request and returns the response.
func (p *Provider) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	messages := p.convertMessages(req.Messages)

	model := req.Model
	if model == "" {
		model = "llama-3.1-8b-instant" // Default to Llama 3.1 8B Instant
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 2048 // Groq supports large context but varies by model
	}

	temperature := float32(req.Temperature)
	if temperature == 0 {
		temperature = 0.7
	}

	chatReq := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	}

	if req.TopP > 0 {
		chatReq.TopP = float32(req.TopP)
	}

	resp, err := p.client.CreateChatCompletion(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("Groq completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices returned")
	}

	content := resp.Choices[0].Message.Content

	return &llm.CompletionResponse{
		Content: content,
		Usage: llm.Usage{
			TotalTokens:      resp.Usage.TotalTokens,
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
		},
	}, nil
}

// CompleteStream sends a completion request and streams the response.
func (p *Provider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (<-chan llm.GenerationChunk, error) {
	messages := p.convertMessages(req.Messages)

	model := req.Model
	if model == "" {
		model = "llama-3.1-8b-instant"
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	temperature := float32(req.Temperature)
	if temperature == 0 {
		temperature = 0.7
	}

	chatReq := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Stream:      true,
	}

	stream, err := p.client.CreateChatCompletionStream(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("Groq stream failed: %w", err)
	}

	chunks := make(chan llm.GenerationChunk)

	go func() {
		defer close(chunks)
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err != nil {
				if err.Error() != "EOF" {
					chunks <- llm.GenerationChunk{Error: err}
				}
				return
			}

			if len(response.Choices) > 0 {
				choice := response.Choices[0]
				chunks <- llm.GenerationChunk{
					Content:      choice.Delta.Content,
					FinishReason: string(choice.FinishReason),
					IsFinal:      choice.FinishReason != "",
				}
			}
		}
	}()

	return chunks, nil
}

// ListModels returns the available models for this provider.
func (p *Provider) ListModels(ctx context.Context) ([]llm.ModelInfo, error) {
	resp, err := p.client.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	models := make([]llm.ModelInfo, 0, len(resp.Models))
	for _, m := range resp.Models {
		models = append(models, llm.ModelInfo{
			ID:       m.ID,
			Name:     m.ID,
			Provider: p.Name(),
			Capabilities: llm.ModelCapabilities{
				SupportsVision:    strings.Contains(m.ID, "vision"), // Groq has vision models coming/supported
				SupportsStreaming: true,
				SupportsFunctions: true, // Groq supports tool calling
			},
		})
	}

	return models, nil
}

// GetModelInfo returns information about a specific model.
func (p *Provider) GetModelInfo(ctx context.Context, modelID string) (*llm.ModelInfo, error) {
	// Simple lookup or generic return
	// Groq models often: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it

	return &llm.ModelInfo{
		ID:       modelID,
		Name:     modelID,
		Provider: p.Name(),
		Capabilities: llm.ModelCapabilities{
			SupportsVision:    false, // Default conservative
			SupportsStreaming: true,
			SupportsFunctions: true,
		},
	}, nil
}

// SupportsMultimodal returns true if the provider supports multimodal inputs.
func (p *Provider) SupportsMultimodal() bool {
	return false // Currently primarily text, though vision models exist in preview
}

// Close releases any resources held by the provider.
func (p *Provider) Close() error {
	return nil
}

// convertMessages converts domain messages to OpenAI messages (compatible with Groq).
func (p *Provider) convertMessages(messages []llm.Message) []openai.ChatCompletionMessage {
	result := make([]openai.ChatCompletionMessage, 0, len(messages))

	for _, msg := range messages {
		// Groq handles standard OpenAI generic content
		content := ""
		for _, c := range msg.Contents {
			if c.Type == llm.ContentTypeText {
				content += c.Text
			}
			// Image content is not standard in Groq yet for all models, skipping unless specifically vision model
		}

		result = append(result, openai.ChatCompletionMessage{
			Role:    string(msg.Role),
			Content: content,
		})
	}

	return result
}
