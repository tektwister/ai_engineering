// Package domain contains the core business models for the Chain of Thought framework.
package domain

import (
	"time"
)

// ContentType represents the type of content in a message.
type ContentType string

const (
	ContentTypeText  ContentType = "text"
	ContentTypeImage ContentType = "image"
	ContentTypeAudio ContentType = "audio"
	ContentTypeVideo ContentType = "video"
)

// ImageDetail represents the detail level for image analysis.
type ImageDetail string

const (
	ImageDetailLow  ImageDetail = "low"
	ImageDetailHigh ImageDetail = "high"
	ImageDetailAuto ImageDetail = "auto"
)

// Content represents a piece of content that can be text, image, audio, or video.
type Content struct {
	Type ContentType `json:"type"`

	// For text content
	Text string `json:"text,omitempty"`

	// For image content
	ImageURL    string      `json:"image_url,omitempty"`
	ImageBase64 string      `json:"image_base64,omitempty"`
	ImageDetail ImageDetail `json:"image_detail,omitempty"`
	MimeType    string      `json:"mime_type,omitempty"`

	// For audio/video content (future extension)
	MediaURL    string `json:"media_url,omitempty"`
	MediaBase64 string `json:"media_base64,omitempty"`
}

// NewTextContent creates a new text content.
func NewTextContent(text string) Content {
	return Content{
		Type: ContentTypeText,
		Text: text,
	}
}

// NewImageURLContent creates a new image content from a URL.
func NewImageURLContent(url string, detail ImageDetail) Content {
	if detail == "" {
		detail = ImageDetailAuto
	}
	return Content{
		Type:        ContentTypeImage,
		ImageURL:    url,
		ImageDetail: detail,
	}
}

// NewImageBase64Content creates a new image content from base64 data.
func NewImageBase64Content(base64Data, mimeType string, detail ImageDetail) Content {
	if detail == "" {
		detail = ImageDetailAuto
	}
	return Content{
		Type:        ContentTypeImage,
		ImageBase64: base64Data,
		MimeType:    mimeType,
		ImageDetail: detail,
	}
}

// Role represents the role of a message sender.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message represents a single message in a conversation.
type Message struct {
	Role     Role      `json:"role"`
	Contents []Content `json:"contents"`
}

// NewTextMessage creates a message with text content only.
func NewTextMessage(role Role, text string) Message {
	return Message{
		Role:     role,
		Contents: []Content{NewTextContent(text)},
	}
}

// NewMultimodalMessage creates a message with multiple content pieces.
func NewMultimodalMessage(role Role, contents ...Content) Message {
	return Message{
		Role:     role,
		Contents: contents,
	}
}

// ThoughtStep represents a single step in the chain of thought reasoning.
type ThoughtStep struct {
	StepNumber  int       `json:"step_number"`
	Title       string    `json:"title"`
	Reasoning   string    `json:"reasoning"`
	Observation string    `json:"observation,omitempty"`
	Confidence  float64   `json:"confidence,omitempty"` // 0.0 to 1.0
	Timestamp   time.Time `json:"timestamp"`
}

// ChainOfThought represents the complete reasoning chain.
type ChainOfThought struct {
	Question       string        `json:"question"`
	Steps          []ThoughtStep `json:"steps"`
	FinalAnswer    string        `json:"final_answer"`
	TotalTokens    int           `json:"total_tokens,omitempty"`
	PromptTokens   int           `json:"prompt_tokens,omitempty"`
	ResponseTokens int           `json:"response_tokens,omitempty"`
	Duration       time.Duration `json:"duration"`
	Model          string        `json:"model"`
	Provider       string        `json:"provider"`
}

// AddStep adds a new thought step to the chain.
func (c *ChainOfThought) AddStep(title, reasoning string) {
	step := ThoughtStep{
		StepNumber: len(c.Steps) + 1,
		Title:      title,
		Reasoning:  reasoning,
		Timestamp:  time.Now(),
	}
	c.Steps = append(c.Steps, step)
}

// CoTRequest represents a request for chain of thought reasoning.
type CoTRequest struct {
	Messages    []Message         `json:"messages"`
	Model       string            `json:"model,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Temperature float64           `json:"temperature,omitempty"`
	TopP        float64           `json:"top_p,omitempty"`
	Options     map[string]any    `json:"options,omitempty"`
}

// CoTResponse represents the response from chain of thought reasoning.
type CoTResponse struct {
	Chain      ChainOfThought `json:"chain"`
	RawContent string         `json:"raw_content"`
	Error      error          `json:"error,omitempty"`
}

// ProviderConfig holds configuration for an LLM provider.
type ProviderConfig struct {
	APIKey      string            `json:"api_key"`
	BaseURL     string            `json:"base_url,omitempty"`
	OrgID       string            `json:"org_id,omitempty"`
	ProjectID   string            `json:"project_id,omitempty"` // For Google Cloud
	Region      string            `json:"region,omitempty"`      // For regional endpoints
	ExtraConfig map[string]string `json:"extra_config,omitempty"`
}

// ModelCapabilities describes what a model can do.
type ModelCapabilities struct {
	SupportsVision     bool `json:"supports_vision"`
	SupportsAudio      bool `json:"supports_audio"`
	SupportsVideo      bool `json:"supports_video"`
	SupportsStreaming  bool `json:"supports_streaming"`
	SupportsFunctions  bool `json:"supports_functions"`
	MaxContextTokens   int  `json:"max_context_tokens"`
	MaxOutputTokens    int  `json:"max_output_tokens"`
}

// ModelInfo provides information about a specific model.
type ModelInfo struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Provider     string            `json:"provider"`
	Capabilities ModelCapabilities `json:"capabilities"`
}
