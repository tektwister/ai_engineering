package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/tektwister/ai_engineering/logit_processor/internal/domain"
	"github.com/tektwister/ai_engineering/small_language_model"
)

func main() {
	// Command line flags
	mode := flag.String("mode", "generate", "Mode: train, generate, gradient_check")
	modelSize := flag.String("size", "small", "Model size: small")
	dataType := flag.String("data", "shakespeare", "Training data: shakespeare, tinystories")
	prompt := flag.String("prompt", "To be or not to be", "Text prompt for generation")
	maxTokens := flag.Int("max_tokens", 50, "Maximum tokens to generate")
	temperature := flag.Float64("temperature", 1.0, "Sampling temperature")
	topK := flag.Int("top_k", 50, "Top-k sampling")
	topP := flag.Float64("top_p", 0.9, "Top-p sampling")

	flag.Parse()

	fmt.Printf("Small Language Model Demo\n")
	fmt.Printf("========================\n\n")

	switch *mode {
	case "train":
		runTraining(*modelSize, *dataType)
	case "generate":
		runGeneration(*modelSize, *dataType, *prompt, *maxTokens, *temperature, *topK, *topP)
	case "gradient_check":
		runGradientCheck()
	default:
		fmt.Printf("Unknown mode: %s\n", *mode)
		os.Exit(1)
	}
}

func runTraining(modelSize, dataType string) {
	fmt.Printf("Training mode\n")
	fmt.Printf("Model size: %s\n", modelSize)
	fmt.Printf("Data type: %s\n\n", dataType)

	// Create model configuration
	config := createModelConfig(modelSize)
	model := small_language_model.NewLanguageModel(config)

	// Load training data
	var text string
	if dataType == "shakespeare" {
		text = small_language_model.LoadShakespeareData()
	} else {
		text = small_language_model.LoadTinyStoriesData()
	}

	fmt.Printf("Loaded %d characters of training data\n", len(text))

	// Quick training for demo
	trainConfig := &small_language_model.TrainingConfig{
		LearningRate: 1e-3,
		MaxIters:     10,
		BatchSize:    4,
		SeqLength:    32,
		EvalInterval: 5,
	}

	trainer := small_language_model.NewTrainer(model, trainConfig)
	err := trainer.TrainOnData(text[:min(len(text), 5000)], trainConfig)
	if err != nil {
		log.Printf("Training warning: %v", err)
	}

	fmt.Println("Training completed!")
}

func runGeneration(modelSize, dataType, prompt string, maxTokens int, temperature float64, topK int, topP float64) {
	fmt.Printf("Generation mode\n")
	fmt.Printf("Model size: %s\n", modelSize)
	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Printf("Max tokens: %d\n", maxTokens)
	fmt.Printf("Temperature: %.2f\n", temperature)
	fmt.Printf("Top-K: %d\n", topK)
	fmt.Printf("Top-P: %.2f\n\n", topP)

	// Create model
	config := createModelConfig(modelSize)
	model := small_language_model.NewLanguageModel(config)

	// Quick training setup
	var text string
	if dataType == "shakespeare" {
		text = small_language_model.LoadShakespeareData()
	} else {
		text = small_language_model.LoadTinyStoriesData()
	}

	// Train tokenizer
	err := model.TrainTokenizer(text[:min(len(text), 5000)], 300)
	if err != nil {
		log.Fatalf("Tokenizer training failed: %v", err)
	}

	// Generate text
	fmt.Printf("Generating text...\n\n")
	generated, err := model.GenerateWithSampling(prompt, maxTokens, temperature, topP, topK)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	fmt.Printf("Generated text:\n%s\n", generated)
}

func runGradientCheck() {
	fmt.Printf("Gradient Checking mode\n\n")

	config := small_language_model.SmallGPTConfig()
	modelConfig := &small_language_model.ModelConfig{
		GPTConfig:   config,
		SamplerType: domain.SamplingTopK,
		Temperature: 1.0,
		TopK:        10,
		TopP:        0.9,
	}

	model := small_language_model.NewLanguageModel(modelConfig)

	text := small_language_model.LoadShakespeareData()[:2000]
	err := model.TrainTokenizer(text, 200)
	if err != nil {
		log.Fatalf("Tokenizer training failed: %v", err)
	}

	trainConfig := &small_language_model.TrainingConfig{
		LearningRate: 1e-3,
		MaxIters:     1,
		BatchSize:    2,
		SeqLength:    16,
	}

	err = small_language_model.TrainWithGradientChecking(text, trainConfig)
	if err != nil {
		log.Printf("Gradient checking warning: %v", err)
	}
}

func createModelConfig(size string) *small_language_model.ModelConfig {
	var gptConfig *small_language_model.GPTConfig

	switch size {
	case "small":
		gptConfig = small_language_model.SmallGPTConfig()
	default:
		gptConfig = small_language_model.SmallGPTConfig()
	}

	return &small_language_model.ModelConfig{
		GPTConfig:   gptConfig,
		SamplerType: domain.SamplingTopP,
		Temperature: 1.0,
		TopK:        50,
		TopP:        0.9,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
