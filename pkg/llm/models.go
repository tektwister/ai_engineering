package llm

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

// CompletionRequest represents a generic request for LLM completion.
// Replaces CoTRequest in the provider interface.
type CompletionRequest struct {
	Messages    []Message         `json:"messages"`
	Model       string            `json:"model,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Temperature float64           `json:"temperature,omitempty"`
	TopP        float64           `json:"top_p,omitempty"`
	Stop        []string          `json:"stop,omitempty"`
	Options     map[string]any    `json:"options,omitempty"`
}

// CompletionResponse represents a generic response from LLM completion.
// Replaces CoTResponse in the provider interface.
type CompletionResponse struct {
	Content string `json:"content"`
	Usage   Usage  `json:"usage,omitempty"`
}

// Usage represents token usage statistics.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// GenerationChunk represents a chunk of streamed response.
// Renamed from StreamChunk to match generic naming, but keeping similar structure.
type GenerationChunk struct {
	Content      string `json:"content"`
	IsFinal      bool   `json:"is_final"`      // True if this is the final answer
	FinishReason string `json:"finish_reason"` // Why generation stopped
	Error        error  `json:"error,omitempty"`
}

// ThoughtStep represents a single step in reasoning (Chain of Thought).
// Moved here for basic reusability if needed, but primarily used in higher layers.
type ThoughtStep struct {
	StepNumber  int       `json:"step_number"`
	Title       string    `json:"title"`
	Reasoning   string    `json:"reasoning"`
	Observation string    `json:"observation,omitempty"`
	Confidence  float64   `json:"confidence,omitempty"` // 0.0 to 1.0
	Timestamp   time.Time `json:"timestamp"`
}
