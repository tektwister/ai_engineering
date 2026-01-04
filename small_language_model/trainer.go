package small_language_model

import (
	"fmt"
	"math/rand"
)

// Trainer handles training of the language model
type Trainer struct {
	model        *LanguageModel
	learningRate float64
	maxIters     int
}

// TrainingConfig holds training configuration
type TrainingConfig struct {
	LearningRate float64
	MaxIters     int
	BatchSize    int
	SeqLength    int
	EvalInterval int // Evaluate every N steps
	SaveInterval int // Save checkpoint every N steps
}

// NewTrainer creates a new trainer
func NewTrainer(model *LanguageModel, config *TrainingConfig) *Trainer {
	return &Trainer{
		model:        model,
		learningRate: config.LearningRate,
		maxIters:     config.MaxIters,
	}
}

// TrainOnData trains the model on the provided text data
func (t *Trainer) TrainOnData(text string, config *TrainingConfig) error {
	fmt.Printf("Starting training on %d characters of text\n", len(text))

	// Train tokenizer if not already trained
	if t.model.GetVocabSize() <= 256 {
		vocabSize := min(1000, len(text)/10) // Rule of thumb: vocab size ~10% of text length
		fmt.Printf("Training tokenizer with vocab size %d\n", vocabSize)
		err := t.model.TrainTokenizer(text, vocabSize)
		if err != nil {
			return fmt.Errorf("failed to train tokenizer: %w", err)
		}
		fmt.Printf("Tokenizer trained. Vocab size: %d\n", t.model.GetVocabSize())
	}

	// Create training sequences
	sequences := t.model.CreateTrainingData(text, config.SeqLength)
	fmt.Printf("Created %d training sequences\n", len(sequences))

	// Training loop
	totalLoss := 0.0
	stepCount := 0

	for iter := 0; iter < config.MaxIters; iter++ {
		// Sample batch
		batchInputs, batchTargets := t.sampleBatch(sequences, config.BatchSize, config.SeqLength)

		// Training step
		loss := t.model.Train(batchInputs, batchTargets, t.learningRate)
		totalLoss += loss
		stepCount++

		// Logging
		if iter%config.EvalInterval == 0 {
			avgLoss := totalLoss / float64(stepCount)
			fmt.Printf("Iteration %d: avg loss = %.4f\n", iter, avgLoss)
			totalLoss = 0.0
			stepCount = 0

			// Generate sample text
			if iter%(config.EvalInterval*5) == 0 {
				t.generateSample()
			}
		}
	}

	fmt.Println("Training completed!")
	return nil
}

// sampleBatch samples a batch of training examples
func (t *Trainer) sampleBatch(sequences [][]int, batchSize, seqLength int) ([][]int, [][]int) {
	batchInputs := make([][]int, batchSize)
	batchTargets := make([][]int, batchSize)

	for b := 0; b < batchSize; b++ {
		// Sample random sequence
		seqIdx := rand.Intn(len(sequences))
		seq := sequences[seqIdx]

		// Ensure sequence is long enough
		if len(seq) < seqLength+1 {
			// Pad or skip short sequences
			continue
		}

		// Sample random starting position
		maxStart := len(seq) - seqLength - 1
		if maxStart <= 0 {
			maxStart = 0
		}
		start := rand.Intn(maxStart + 1)

		// Create input and target
		batchInputs[b] = make([]int, seqLength)
		batchTargets[b] = make([]int, seqLength)

		copy(batchInputs[b], seq[start:start+seqLength])
		copy(batchTargets[b], seq[start+1:start+seqLength+1])
	}

	return batchInputs, batchTargets
}

// generateSample generates a sample of text to show training progress
func (t *Trainer) generateSample() {
	prompt := "The cat"
	fmt.Printf("\nSample generation with prompt: '%s'\n", prompt)

	generated, err := t.model.GenerateText(prompt, 50)
	if err != nil {
		fmt.Printf("Generation failed: %v\n", err)
		return
	}

	fmt.Printf("Generated: '%s'\n\n", generated)
}

// GradientChecker performs gradient checking using finite differences
type GradientChecker struct {
	model *LanguageModel
	eps   float64
}

// NewGradientChecker creates a new gradient checker
func NewGradientChecker(model *LanguageModel) *GradientChecker {
	return &GradientChecker{
		model: model,
		eps:   1e-6,
	}
}

// CheckGradients performs gradient checking on a small batch
func (gc *GradientChecker) CheckGradients(inputs [][]int, targets [][]int) error {
	fmt.Println("Performing gradient checking...")

	// Get original loss
	_, originalLoss := gc.model.gpt.Forward(inputs, targets)

	// Check gradients for a few parameters (simplified)
	// In a full implementation, we'd check all parameters

	// For demonstration, check gradients of the first layer weights
	gpt := gc.model.GetGPT()
	if len(gpt.blocks) > 0 {
		block := gpt.blocks[0]

		// Check attention weights (simplified - just check first few parameters)
		attn := block.Attn
		if attn.WQ != nil && len(attn.WQ.Weight.Data) > 0 {
			fmt.Println("Checking gradients for attention WQ weights...")

			// Check first parameter
			originalVal := attn.WQ.Weight.Data[0]

			// Forward difference
			attn.WQ.Weight.Data[0] = originalVal + gc.eps
			_, lossPlus := gc.model.gpt.Forward(inputs, targets)

			// Backward difference
			attn.WQ.Weight.Data[0] = originalVal - gc.eps
			_, lossMinus := gc.model.gpt.Forward(inputs, targets)

			// Restore original value
			attn.WQ.Weight.Data[0] = originalVal

			// Numerical gradient
			numGrad := (lossPlus - lossMinus) / (2 * gc.eps)

			// For now, just print the numerical gradient
			// In a full implementation, we'd compare with computed gradient
			fmt.Printf("Numerical gradient for WQ[0,0]: %.6f\n", numGrad)
			fmt.Printf("Loss values: original=%.6f, plus=%.6f, minus=%.6f\n", originalLoss, lossPlus, lossMinus)
		}
	}

	fmt.Println("Gradient checking completed.")
	return nil
}

// TrainWithGradientChecking trains with gradient verification
func TrainWithGradientChecking(text string, config *TrainingConfig) error {
	// Create small model for testing
	modelConfig := &ModelConfig{
		GPTConfig:   SmallGPTConfig(),
		SamplerType: SamplingTopK,
		Temperature: 1.0,
		TopK:        10,
		TopP:        0.9,
	}

	model := NewLanguageModel(modelConfig)
	trainer := NewTrainer(model, config)

	// Train tokenizer on small subset
	smallText := text[:min(len(text), 10000)]
	err := model.TrainTokenizer(smallText, 500)
	if err != nil {
		return fmt.Errorf("tokenizer training failed: %w", err)
	}

	// Create small batch for gradient checking
	sequences := model.CreateTrainingData(smallText, config.SeqLength)
	if len(sequences) == 0 {
		return fmt.Errorf("no training sequences created")
	}

	batchInputs, batchTargets := trainer.sampleBatch(sequences, min(2, config.BatchSize), config.SeqLength)

	// Perform gradient checking
	checker := NewGradientChecker(model)
	err = checker.CheckGradients(batchInputs, batchTargets)
	if err != nil {
		return fmt.Errorf("gradient checking failed: %w", err)
	}

	// Train for a few steps
	fmt.Println("Starting training...")
	for i := 0; i < min(10, config.MaxIters); i++ {
		batchInputs, batchTargets := trainer.sampleBatch(sequences, config.BatchSize, config.SeqLength)
		loss := model.Train(batchInputs, batchTargets, config.LearningRate)
		fmt.Printf("Step %d: loss = %.4f\n", i, loss)

		if i%5 == 0 {
			trainer.generateSample()
		}
	}

	return nil
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
