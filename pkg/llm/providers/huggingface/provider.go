// Package huggingface provides a Hugging Face Inference API provider implementation.
package huggingface

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/tektwister/ai_engineering/pkg/llm"
)

// Provider implements the Provider interface for Hugging Face.
type Provider struct {
	client *http.Client
	config *llm.ProviderConfig
}

// New creates a new Hugging Face provider.
func New(config *llm.ProviderConfig) (*Provider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("Hugging Face API key is required")
	}

	return &Provider{
		client: &http.Client{},
		config: config,
	}, nil
}

// Name returns the provider's name.
func (p *Provider) Name() string {
	return "huggingface"
}

type hfRequest struct {
	Inputs     string       `json:"inputs"`
	Parameters hfParameters `json:"parameters,omitempty"`
}

type hfParameters struct {
	MaxNewTokens int     `json:"max_new_tokens,omitempty"`
	Temperature  float64 `json:"temperature,omitempty"`
	TopP         float64 `json:"top_p,omitempty"`
	ReturnFull   bool    `json:"return_full_text,omitempty"`
}

type hfResponse struct {
	GeneratedText string `json:"generated_text"`
}

// Complete sends a completion request and returns the response.
func (p *Provider) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	model := req.Model
	if model == "" {
		return nil, fmt.Errorf("model is required for Hugging Face provider")
	}

	// Construct the prompt from messages
	var promptBuilder strings.Builder
	for _, msg := range req.Messages {
		for _, content := range msg.Contents {
			if content.Type == llm.ContentTypeText {
				// Simple chat format: User: ...\nAssistant: ...
				role := "User"
				if msg.Role == llm.RoleAssistant {
					role = "Assistant"
				} else if msg.Role == llm.RoleSystem {
					role = "System"
				}
				fmt.Fprintf(&promptBuilder, "%s: %s\n", role, content.Text)
			}
		}
	}
	// Add Assistant prompt at the end
	promptBuilder.WriteString("Assistant: ")

	hfReq := hfRequest{
		Inputs: promptBuilder.String(),
		Parameters: hfParameters{
			MaxNewTokens: req.MaxTokens,
			Temperature:  req.Temperature,
			TopP:         req.TopP,
			ReturnFull:   false,
		},
	}

	if hfReq.Parameters.MaxNewTokens == 0 {
		hfReq.Parameters.MaxNewTokens = 256 // Default
	}

	body, err := json.Marshal(hfReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("https://api-inference.huggingface.co/models/%s", model)
	reqHTTP, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	reqHTTP.Header.Set("Authorization", "Bearer "+p.config.APIKey)
	reqHTTP.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(reqHTTP)
	if err != nil {
		return nil, fmt.Errorf("Hugging Face API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Hugging Face API returned error: %s - %s", resp.Status, string(bodyBytes))
	}

	var hfResp []hfResponse
	if err := json.NewDecoder(resp.Body).Decode(&hfResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(hfResp) == 0 {
		return nil, fmt.Errorf("empty response from Hugging Face")
	}

	return &llm.CompletionResponse{
		Content: hfResp[0].GeneratedText,
		Usage: llm.Usage{
			// HF Inference API doesn't always return usage
		},
	}, nil
}

// CompleteStream sends a completion request and streams the response.
func (p *Provider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (<-chan llm.GenerationChunk, error) {
	return nil, fmt.Errorf("streaming not implemented for Hugging Face provider")
}

// ListModels returns the available models for this provider.
func (p *Provider) ListModels(ctx context.Context) ([]llm.ModelInfo, error) {
	// We can't easily list all HF models. Return a few popular ones.
	return []llm.ModelInfo{
		{
			ID:       "meta-llama/Meta-Llama-3-8B-Instruct",
			Name:     "Llama 3 8B Instruct",
			Provider: p.Name(),
			Capabilities: llm.ModelCapabilities{
				SupportsVision:    false,
				SupportsStreaming: false,
			},
		},
		{
			ID:       "mistralai/Mistral-7B-Instruct-v0.3",
			Name:     "Mistral 7B Instruct v0.3",
			Provider: p.Name(),
			Capabilities: llm.ModelCapabilities{
				SupportsVision:    false,
				SupportsStreaming: false,
			},
		},
	}, nil
}

// GetModelInfo returns information about a specific model.
func (p *Provider) GetModelInfo(ctx context.Context, modelID string) (*llm.ModelInfo, error) {
	return &llm.ModelInfo{
		ID:       modelID,
		Name:     modelID,
		Provider: p.Name(),
		Capabilities: llm.ModelCapabilities{
			SupportsVision:    false,
			SupportsStreaming: false,
		},
	}, nil
}

// SupportsMultimodal returns true if the provider supports multimodal inputs.
func (p *Provider) SupportsMultimodal() bool {
	return false
}

// Close releases any resources held by the provider.
func (p *Provider) Close() error {
	return nil
}
