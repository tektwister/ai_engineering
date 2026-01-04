package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/tektwister/ai_engineering/chain-of-thoughts/internal/core"
	"github.com/tektwister/ai_engineering/chain-of-thoughts/internal/domain"
	"github.com/tektwister/ai_engineering/pkg/config"
	"github.com/tektwister/ai_engineering/pkg/llm/providers"
)

func main() {
	// Load Configuration
	cfg, err := config.Load()
	if err != nil {
		log.Printf("Warning loading config: %v", err)
	}

	if cfg.APIKey == "" {
		fmt.Printf("Please set LLM_API_KEY environment variable\n")
		return
	}

	// Initialize Provider
	providerConfig := &domain.ProviderConfig{
		APIKey:  cfg.APIKey,
		BaseURL: cfg.BaseURL,
		OrgID:   cfg.OrgID,
	}

	fmt.Printf("Initializing %s provider...\n", cfg.ProviderName)
	provider, err := providers.Create(cfg.ProviderName, providerConfig)
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}

	// Initialize Engine
	strategy := &core.ExplicitStepStrategy{}
	engine := core.NewEngine(provider, strategy)

	ctx := context.Background()

	// Example 1: Text-only Logic Puzzle
	runTextExample(ctx, engine)

	// Example 2: Multimodal Reasoning (if supported)
	if provider.SupportsMultimodal() {
		// Check for an example image
		imagePath := "example_chart.png"
		if _, err := os.Stat(imagePath); err == nil {
			runMultimodalExample(ctx, engine, imagePath)
		} else {
			fmt.Println("\nSkipping multimodal example (example_chart.png not found)")
		}
	}
}

func runTextExample(ctx context.Context, engine *core.Engine) {
	fmt.Println("\n=== Running Text-Only Logic Puzzle ===")

	question := "If I have a 3 gallon jug and a 5 gallon jug, how can I measure exactly 4 gallons of water?"

	fmt.Printf("Question: %s\n\n", question)
	fmt.Println("Thinking...")

	req := &domain.CoTRequest{
		Messages: []domain.Message{
			domain.NewTextMessage(domain.RoleUser, question),
		},
		MaxTokens: 1000,
	}

	start := time.Now()
	resp, err := engine.Reason(ctx, req)
	if err != nil {
		log.Printf("Error reasoning: %v", err)
		return
	}
	duration := time.Since(start)

	printChain(resp.Chain, duration)
}

func runMultimodalExample(ctx context.Context, engine *core.Engine, imagePath string) {
	fmt.Println("\n=== Running Multimodal Reasoning ===")

	question := "Analyze this chart. What are the key trends and what might be the prediction for next year?"

	// Read image file
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		log.Printf("Failed to read image: %v", err)
		return
	}

	// Convert to base64
	imageBase64 := base64.StdEncoding.EncodeToString(imageData)

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Image: %s\n\n", imagePath)
	fmt.Println("Thinking...")

	req := &domain.CoTRequest{
		Messages: []domain.Message{
			domain.NewMultimodalMessage(
				domain.RoleUser,
				domain.NewTextContent(question),
				domain.NewImageBase64Content(imageBase64, "image/png", domain.ImageDetailAuto),
			),
		},
		MaxTokens: 1000,
	}

	start := time.Now()
	resp, err := engine.Reason(ctx, req)
	if err != nil {
		log.Printf("Error reasoning: %v", err)
		return
	}
	duration := time.Since(start)

	printChain(resp.Chain, duration)
}

func printChain(chain domain.ChainOfThought, duration time.Duration) {
	fmt.Println("------------------------------------------------")
	fmt.Printf("Model: %s (%s)\n", chain.Model, chain.Provider)
	fmt.Printf("Duration: %v\n", duration)
	fmt.Printf("Tokens: %d (Prompt: %d, Response: %d)\n", chain.TotalTokens, chain.PromptTokens, chain.ResponseTokens)
	fmt.Println("------------------------------------------------")

	fmt.Println("### Reasoning Steps:")
	for _, step := range chain.Steps {
		fmt.Printf("%d. %s\n", step.StepNumber, step.Reasoning)
	}

	fmt.Println("\n### Final Answer:")
	fmt.Println(chain.FinalAnswer)
	fmt.Println("------------------------------------------------")
}
