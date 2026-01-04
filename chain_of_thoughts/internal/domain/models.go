// Package domain contains the core business models for the Chain of Thought framework.
package domain

import (
	"time"

	"github.com/tektwister/ai_engineering/pkg/llm"
)

// Re-export types from pkg/llm for convenience or usage in domain
type ContentType = llm.ContentType
type ImageDetail = llm.ImageDetail
type Content = llm.Content
type Role = llm.Role
type Message = llm.Message
type ProviderConfig = llm.ProviderConfig
type ModelCapabilities = llm.ModelCapabilities
type ModelInfo = llm.ModelInfo

const (
	ContentTypeText  = llm.ContentTypeText
	ContentTypeImage = llm.ContentTypeImage
	ContentTypeAudio = llm.ContentTypeAudio
	ContentTypeVideo = llm.ContentTypeVideo
)

const (
	ImageDetailLow  = llm.ImageDetailLow
	ImageDetailHigh = llm.ImageDetailHigh
	ImageDetailAuto = llm.ImageDetailAuto
)

const (
	RoleSystem    = llm.RoleSystem
	RoleUser      = llm.RoleUser
	RoleAssistant = llm.RoleAssistant
)

// Functions to create content (wrappers around llm functions)

func NewTextContent(text string) Content {
	return llm.NewTextContent(text)
}

func NewImageURLContent(url string, detail ImageDetail) Content {
	return llm.NewImageURLContent(url, detail)
}

func NewImageBase64Content(base64Data, mimeType string, detail ImageDetail) Content {
	return llm.NewImageBase64Content(base64Data, mimeType, detail)
}

func NewTextMessage(role Role, text string) Message {
	return llm.NewTextMessage(role, text)
}

func NewMultimodalMessage(role Role, contents ...Content) Message {
	return llm.NewMultimodalMessage(role, contents...)
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
		Reasoning  : reasoning,
		Timestamp:  time.Now(),
	}
	c.Steps = append(c.Steps, step)
}

// CoTRequest represents a request for chain of thought reasoning.
type CoTRequest struct {
	Messages    []llm.Message     `json:"messages"`
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
