package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/tektwister/ai_engineering/logit_processor/internal/core"
	"github.com/tektwister/ai_engineering/logit_processor/internal/domain"
)

func main() {
	fmt.Println("=== Logit Processor Demo ===")

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create sample logits (simulating a language model output)
	// Higher values = higher probability before softmax
	logits := createSampleLogits(100) // 100 tokens vocabulary

	// Demo different sampling methods
	methods := []domain.SamplingMethod{
		domain.SamplingGreedy,
		domain.SamplingMultinomial,
		domain.SamplingTopK,
		domain.SamplingTopP,
	}

	ctx := context.Background()

	for _, method := range methods {
		fmt.Printf("\n--- %s Sampling ---\n", method)

		// Create processor for this method
		processor, err := core.CreateProcessorWithMethod(method)
		if err != nil {
			log.Printf("Failed to create processor for %s: %v", method, err)
			continue
		}

		// Create request with appropriate config
		req := createRequestForMethod(logits, method)

		// Process the logits
		response, err := processor.Process(ctx, req)
		if err != nil {
			log.Printf("Failed to process logits with %s: %v", method, err)
			continue
		}

		// Display results
		displayResults(response)
	}

	// Demo batch processing
	fmt.Println("\n--- Batch Processing Demo ---")
	demoBatchProcessing(ctx)

	fmt.Println("\n=== Demo Complete ===")
}

// createSampleLogits creates realistic sample logits for demonstration.
func createSampleLogits(vocabSize int) domain.Logits {
	logits := make(domain.Logits, vocabSize)

	// Create a distribution where a few tokens have high logits
	// and most have lower logits (typical of language models)
	for i := range logits {
		if i < 5 {
			// Top 5 tokens have high logits
			logits[i] = float32(2.0 + rand.Float64()*2.0)
		} else if i < 20 {
			// Next 15 tokens have medium logits
			logits[i] = float32(0.5 + rand.Float64()*1.5)
		} else {
			// Rest have low logits
			logits[i] = float32(-2.0 + rand.Float64()*2.0)
		}
	}

	return logits
}

// createRequestForMethod creates a request with appropriate config for the method.
func createRequestForMethod(logits domain.Logits, method domain.SamplingMethod) *domain.LogitRequest {
	config := domain.NewSamplingConfig(method)

	// Customize config based on method
	switch method {
	case domain.SamplingMultinomial:
		config.Temperature = 0.8
	case domain.SamplingTopK:
		config.TopK = 10
		config.Temperature = 0.9
	case domain.SamplingTopP:
		config.TopP = 0.9
		config.Temperature = 0.8
	}

	return domain.NewLogitRequest(logits, config)
}

// displayResults shows the processing results in a readable format.
func displayResults(response *domain.LogitResponse) {
	fmt.Printf("Selected Token: %d (Prob: %.4f)\n",
		response.SelectedToken.Token, response.SelectedToken.Prob)

	fmt.Printf("Processing Time: %v\n", response.ProcessingTime)

	fmt.Printf("Top 5 Tokens:\n")
	for i, token := range response.TopTokens {
		if i >= 5 {
			break
		}
		fmt.Printf("  %d: %.4f\n", token.Token, token.Prob)
	}

	// Show entropy of distribution
	entropy := calculateEntropy(response.Probabilities)
	fmt.Printf("Distribution Entropy: %.4f\n", entropy)

	fmt.Printf("Config: Method=%s, Temp=%.2f, TopK=%d, TopP=%.2f\n",
		response.Config.Method,
		response.Config.Temperature,
		response.Config.TopK,
		response.Config.TopP)
}

// calculateEntropy computes the entropy of a probability distribution.
func calculateEntropy(probs domain.Probabilities) float64 {
	entropy := 0.0
	for _, p := range probs {
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

// demoBatchProcessing demonstrates processing multiple requests at once.
func demoBatchProcessing(ctx context.Context) {
	// Create multiple logit arrays
	batchSize := 3
	requests := make([]*domain.LogitRequest, batchSize)

	for i := 0; i < batchSize; i++ {
		logits := createSampleLogits(50) // Smaller vocab for demo
		config := domain.NewSamplingConfig(domain.SamplingTopK)
		config.TopK = 5
		config.Temperature = 0.7

		requests[i] = domain.NewLogitRequest(logits, config)
	}

	// Create processor and process batch
	processor, _ := core.CreateProcessorWithMethod(domain.SamplingTopK)
	responses, err := processor.ProcessBatch(ctx, requests)
	if err != nil {
		log.Printf("Batch processing failed: %v", err)
		return
	}

	fmt.Printf("Batch processed %d requests:\n", len(responses))
	for i, response := range responses {
		fmt.Printf("  Request %d: Selected token %d (prob: %.4f)\n",
			i+1, response.SelectedToken.Token, response.SelectedToken.Prob)
	}
}

// Example of how to use the logit processor programmatically
func exampleUsage() {
	// This function shows how to use the logit processor in code
	// (not called in main, just for documentation)

	// Create processor
	processor := core.CreateDefaultProcessor()

	// Create logits (normally from your model)
	logits := domain.Logits{1.0, 2.0, 0.5, -1.0}

	// Create sampling config
	config := domain.SamplingConfig{
		Method:      domain.SamplingTopK,
		Temperature: 0.8,
		TopK:        2,
	}

	// Create request
	req := domain.NewLogitRequest(logits, config)

	// Process
	ctx := context.Background()
	response, err := processor.Process(ctx, req)
	if err != nil {
		log.Fatal(err)
	}

	// Use result
	selectedToken := response.SelectedToken.Token
	fmt.Printf("Selected token: %d\n", selectedToken)
}
