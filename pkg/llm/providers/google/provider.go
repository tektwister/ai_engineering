// Package google provides a Google Gemini LLM provider implementation.
package google

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"github.com/tektwister/ai_engineering/pkg/llm"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// Provider implements the Provider interface for Google Gemini.
type Provider struct {
	client *genai.Client
	config *llm.ProviderConfig
}

// New creates a new Google Gemini provider.
func New(config *llm.ProviderConfig) (*Provider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("Google API key is required")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(config.APIKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	return &Provider{
		client: client,
		config: config,
	}, nil
}

// Name returns the provider's name.
func (p *Provider) Name() string {
	return "google"
}

// Complete sends a completion request and returns the response.
func (p *Provider) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	model := req.Model
	if model == "" {
		model = "gemini-2.0-flash"
	}

	genModel := p.client.GenerativeModel(model)

	// Configure the model
	if req.MaxTokens > 0 {
		genModel.SetMaxOutputTokens(int32(req.MaxTokens))
	}
	if req.Temperature > 0 {
		genModel.SetTemperature(float32(req.Temperature))
	} else {
		genModel.SetTemperature(0.7)
	}
	if req.TopP > 0 {
		genModel.SetTopP(float32(req.TopP))
	}

	// Convert messages
	parts, systemInstruction := p.convertMessages(req.Messages)

	if systemInstruction != "" {
		genModel.SystemInstruction = &genai.Content{
			Parts: []genai.Part{genai.Text(systemInstruction)},
		}
	}

	resp, err := genModel.GenerateContent(ctx, parts...)
	if err != nil {
		return nil, fmt.Errorf("Gemini completion failed: %w", err)
	}

	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("no response candidates returned")
	}

	// Extract text content
	var content strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			content.WriteString(string(text))
		}
	}

	// Extract token counts
	var promptTokens, responseTokens int
	if resp.UsageMetadata != nil {
		promptTokens = int(resp.UsageMetadata.PromptTokenCount)
		responseTokens = int(resp.UsageMetadata.CandidatesTokenCount)
	}

	return &llm.CompletionResponse{
		Content: content.String(),
		Usage: llm.Usage{
			TotalTokens:      promptTokens + responseTokens,
			PromptTokens:     promptTokens,
			CompletionTokens: responseTokens,
		},
	}, nil
}

// CompleteStream sends a completion request and streams the response.
func (p *Provider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (<-chan llm.GenerationChunk, error) {
	model := req.Model
	if model == "" {
		model = "gemini-2.0-flash"
	}

	genModel := p.client.GenerativeModel(model)

	// Configure the model
	if req.MaxTokens > 0 {
		genModel.SetMaxOutputTokens(int32(req.MaxTokens))
	}
	if req.Temperature > 0 {
		genModel.SetTemperature(float32(req.Temperature))
	} else {
		genModel.SetTemperature(0.7)
	}
	if req.TopP > 0 {
		genModel.SetTopP(float32(req.TopP))
	}

	// Convert messages
	parts, systemInstruction := p.convertMessages(req.Messages)

	if systemInstruction != "" {
		genModel.SystemInstruction = &genai.Content{
			Parts: []genai.Part{genai.Text(systemInstruction)},
		}
	}

	iter := genModel.GenerateContentStream(ctx, parts...)

	chunks := make(chan llm.GenerationChunk)

	go func() {
		defer close(chunks)

		for {
			resp, err := iter.Next()
			if err == iterator.Done {
				chunks <- llm.GenerationChunk{
					IsFinal:      true,
					FinishReason: "stop",
				}
				return
			}
			if err != nil {
				chunks <- llm.GenerationChunk{Error: err}
				return
			}

			if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
				for _, part := range resp.Candidates[0].Content.Parts {
					if text, ok := part.(genai.Text); ok {
						chunks <- llm.GenerationChunk{
							Content: string(text),
						}
					}
				}
			}
		}
	}()

	return chunks, nil
}

// ListModels returns the available models for this provider.
func (p *Provider) ListModels(ctx context.Context) ([]llm.ModelInfo, error) {
	iter := p.client.ListModels(ctx)

	var models []llm.ModelInfo
	for {
		m, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to list models: %w", err)
		}

		// Only include Gemini models
		if strings.Contains(m.Name, "gemini") {
			models = append(models, llm.ModelInfo{
				ID:       m.Name,
				Name:     m.DisplayName,
				Provider: p.Name(),
				Capabilities: llm.ModelCapabilities{
					SupportsVision:    true, // All Gemini models support vision
					SupportsStreaming: true,
					MaxOutputTokens:   int(m.OutputTokenLimit),
					MaxContextTokens:  int(m.InputTokenLimit),
				},
			})
		}
	}

	return models, nil
}

// GetModelInfo returns information about a specific model.
func (p *Provider) GetModelInfo(ctx context.Context, modelID string) (*llm.ModelInfo, error) {
	m, err := p.client.GenerativeModel(modelID).Info(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get model info: %w", err)
	}

	return &llm.ModelInfo{
		ID:       m.Name,
		Name:     m.DisplayName,
		Provider: p.Name(),
		Capabilities: llm.ModelCapabilities{
			SupportsVision:    true,
			SupportsStreaming: true,
			MaxOutputTokens:   int(m.OutputTokenLimit),
			MaxContextTokens:  int(m.InputTokenLimit),
		},
	}, nil
}

// SupportsMultimodal returns true if the provider supports multimodal inputs.
func (p *Provider) SupportsMultimodal() bool {
	return true
}

// Close releases any resources held by the provider.
func (p *Provider) Close() error {
	return p.client.Close()
}

// convertMessages converts domain messages to Gemini parts.
// Returns the parts and any system instruction extracted.
func (p *Provider) convertMessages(messages []llm.Message) ([]genai.Part, string) {
	var systemInstruction string
	var parts []genai.Part

	for _, msg := range messages {
		// Extract system message
		if msg.Role == llm.RoleSystem {
			for _, content := range msg.Contents {
				if content.Type == llm.ContentTypeText {
					systemInstruction = content.Text
					break
				}
			}
			continue
		}

		// Add role prefix for conversation context
		var rolePrefix string
		switch msg.Role {
		case llm.RoleUser:
			rolePrefix = "User: "
		case llm.RoleAssistant:
			rolePrefix = "Assistant: "
		}

		for _, content := range msg.Contents {
			switch content.Type {
			case llm.ContentTypeText:
				parts = append(parts, genai.Text(rolePrefix+content.Text))
			case llm.ContentTypeImage:
				if content.ImageBase64 != "" {
					mimeType := content.MimeType
					if mimeType == "" {
						mimeType = "image/png"
					}
					parts = append(parts, genai.Blob{
						MIMEType: mimeType,
						Data:     []byte(content.ImageBase64),
					})
				}
				// Note: Gemini also supports image URLs but requires different handling
			}
		}
	}

	return parts, systemInstruction
}
