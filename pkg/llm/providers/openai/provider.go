// Package openai provides an OpenAI LLM provider implementation.
package openai

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	openai "github.com/sashabaranov/go-openai"
	"github.com/tektwister/ai_engineering/pkg/llm"
)

// Provider implements the Provider interface for OpenAI.
type Provider struct {
	client *openai.Client
	config *llm.ProviderConfig
}

// New creates a new OpenAI provider.
func New(config *llm.ProviderConfig) (*Provider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

	clientConfig := openai.DefaultConfig(config.APIKey)
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
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
	return "openai"
}

// Complete sends a completion request and returns the response.
func (p *Provider) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	messages := p.convertMessages(req.Messages)

	model := req.Model
	if model == "" {
		model = openai.GPT4o // Default to GPT-4o for multimodal support
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
	}

	if req.TopP > 0 {
		chatReq.TopP = float32(req.TopP)
	}

	resp, err := p.client.CreateChatCompletion(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("OpenAI completion failed: %w", err)
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
		model = openai.GPT4o
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
		return nil, fmt.Errorf("OpenAI stream failed: %w", err)
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
		// Filter to only include chat models
		if strings.Contains(m.ID, "gpt") || strings.Contains(m.ID, "o1") {
			models = append(models, llm.ModelInfo{
				ID:       m.ID,
				Name:     m.ID,
				Provider: p.Name(),
				Capabilities: llm.ModelCapabilities{
					SupportsVision:    strings.Contains(m.ID, "gpt-4") || strings.Contains(m.ID, "o1"),
					SupportsStreaming: true,
					SupportsFunctions: true,
				},
			})
		}
	}

	return models, nil
}

// GetModelInfo returns information about a specific model.
func (p *Provider) GetModelInfo(ctx context.Context, modelID string) (*llm.ModelInfo, error) {
	// OpenAI doesn't have a direct model info endpoint, so we use predefined info
	info := getModelInfo(modelID)
	if info == nil {
		return nil, fmt.Errorf("unknown model: %s", modelID)
	}
	return info, nil
}

// SupportsMultimodal returns true if the provider supports multimodal inputs.
func (p *Provider) SupportsMultimodal() bool {
	return true
}

// Close releases any resources held by the provider.
func (p *Provider) Close() error {
	return nil
}

// convertMessages converts domain messages to OpenAI messages.
func (p *Provider) convertMessages(messages []llm.Message) []openai.ChatCompletionMessage {
	result := make([]openai.ChatCompletionMessage, 0, len(messages))

	for _, msg := range messages {
		var parts []openai.ChatMessagePart

		for _, content := range msg.Contents {
			switch content.Type {
			case llm.ContentTypeText:
				parts = append(parts, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeText,
					Text: content.Text,
				})
			case llm.ContentTypeImage:
				imageURL := content.ImageURL
				if content.ImageBase64 != "" {
					mimeType := content.MimeType
					if mimeType == "" {
						mimeType = "image/png"
					}
					imageURL = fmt.Sprintf("data:%s;base64,%s", mimeType, content.ImageBase64)
				}
				parts = append(parts, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeImageURL,
					ImageURL: &openai.ChatMessageImageURL{
						URL:    imageURL,
						Detail: openai.ImageURLDetail(content.ImageDetail),
					},
				})
			}
		}

		// If there's only text content, use the simple format
		if len(parts) == 1 && parts[0].Type == openai.ChatMessagePartTypeText {
			result = append(result, openai.ChatCompletionMessage{
				Role:    string(msg.Role),
				Content: parts[0].Text,
			})
		} else {
			result = append(result, openai.ChatCompletionMessage{
				Role:         string(msg.Role),
				MultiContent: parts,
			})
		}
	}

	return result
}

// getModelInfo returns predefined model information.
func getModelInfo(modelID string) *llm.ModelInfo {
	models := map[string]*llm.ModelInfo{
		"gpt-4o": {
			ID:       "gpt-4o",
			Name:     "GPT-4o",
			Provider: "openai",
			Capabilities: llm.ModelCapabilities{
				SupportsVision:    true,
				SupportsAudio:     true,
				SupportsStreaming: true,
				SupportsFunctions: true,
				MaxContextTokens:  128000,
				MaxOutputTokens:   16384,
			},
		},
		"gpt-4o-mini": {
			ID:       "gpt-4o-mini",
			Name:     "GPT-4o Mini",
			Provider: "openai",
			Capabilities: llm.ModelCapabilities{
				SupportsVision:    true,
				SupportsStreaming: true,
				SupportsFunctions: true,
				MaxContextTokens:  128000,
				MaxOutputTokens:   16384,
			},
		},
		"gpt-4-turbo": {
			ID:       "gpt-4-turbo",
			Name:     "GPT-4 Turbo",
			Provider: "openai",
			Capabilities: llm.ModelCapabilities{
				SupportsVision:    true,
				SupportsStreaming: true,
				SupportsFunctions: true,
				MaxContextTokens:  128000,
				MaxOutputTokens:   4096,
			},
		},
		"o1-preview": {
			ID:       "o1-preview",
			Name:     "O1 Preview",
			Provider: "openai",
			Capabilities: llm.ModelCapabilities{
				SupportsVision:    true,
				SupportsStreaming: false,
				SupportsFunctions: false,
				MaxContextTokens:  128000,
				MaxOutputTokens:   32768,
			},
		},
		"o1-mini": {
			ID:       "o1-mini",
			Name:     "O1 Mini",
			Provider: "openai",
			Capabilities: llm.ModelCapabilities{
				SupportsVision:    true,
				SupportsStreaming: false,
				SupportsFunctions: false,
				MaxContextTokens:  128000,
				MaxOutputTokens:   65536,
			},
		},
	}

	if info, ok := models[modelID]; ok {
		return info
	}

	// Return a generic info for unknown models to support Ollama/LocalAI/others
	return &llm.ModelInfo{
		ID:       modelID,
		Name:     modelID,
		Provider: "openai",
		Capabilities: llm.ModelCapabilities{
			SupportsVision:    true,  // Assume support
			SupportsStreaming: true,  // Most modern models support this
			SupportsFunctions: false, // Conservative default
			MaxContextTokens:  32768, // Reasonable default for modern models
			MaxOutputTokens:   4096,
		},
	}
}

// Helper function for base64 encoding (exposed for testing)
func encodeBase64(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}
