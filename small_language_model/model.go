package small_language_model

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tektwister/ai_engineering/tokenizer"
	"github.com/tektwister/ai_engineering/transformer"
)

// LanguageModel represents the complete language model with tokenizer and sampling
type LanguageModel struct {
	gpt       *GPT
	tokenizer *tokenizer.BPE
	config    *ModelConfig
}

// SamplingMethod defines different token sampling strategies.
type SamplingMethod string

const (
	SamplingGreedy SamplingMethod = "greedy"
	SamplingTopK   SamplingMethod = "top_k"
	SamplingTopP   SamplingMethod = "top_p"
)

// ModelConfig holds configuration for the language model
type ModelConfig struct {
	GPTConfig     *GPTConfig
	TokenizerPath string // Path to save/load tokenizer
	SamplerType   SamplingMethod
	Temperature   float64
	TopK          int
	TopP          float64
}

// NewLanguageModel creates a new language model
func NewLanguageModel(config *ModelConfig) *LanguageModel {
	// Initialize GPT model
	gpt := NewGPT(config.GPTConfig)

	// Initialize tokenizer
	bpe := tokenizer.NewBPE()

	return &LanguageModel{
		gpt:       gpt,
		tokenizer: bpe,
		config:    config,
	}
}

// TrainTokenizer trains the BPE tokenizer on the given text
func (lm *LanguageModel) TrainTokenizer(text string, vocabSize int) error {
	return lm.tokenizer.Train(text, vocabSize)
}

// EncodeTokenizes encodes text to token IDs
func (lm *LanguageModel) Encode(text string) []int {
	return lm.tokenizer.Encode(text)
}

// Decode decodes token IDs back to text
func (lm *LanguageModel) Decode(tokens []int) string {
	return lm.tokenizer.Decode(tokens)
}

// GetVocabSize returns the tokenizer vocabulary size
func (lm *LanguageModel) GetVocabSize() int {
	return lm.tokenizer.GetVocabSize()
}

// GenerateText generates text autoregressively
// prompt: input text prompt
// maxLength: maximum number of tokens to generate
// Returns: generated text
func (lm *LanguageModel) GenerateText(prompt string, maxLength int) (string, error) {
	// Encode prompt
	promptTokens := lm.Encode(prompt)
	if len(promptTokens) == 0 {
		return "", fmt.Errorf("empty prompt after tokenization")
	}

	// Prepare batch (single sequence)
	batchTokens := [][]int{promptTokens}

	// Generate tokens
	generatedTokens := lm.gpt.Generate(
		batchTokens,
		maxLength,
		lm.config.Temperature,
		lm.config.TopP,
		lm.config.TopK,
	)

	// Decode generated tokens
	resultTokens := generatedTokens[0]
	resultText := lm.Decode(resultTokens)

	return resultText, nil
}

// GenerateWithSampling generates text with custom sampling parameters
func (lm *LanguageModel) GenerateWithSampling(prompt string, maxLength int, temperature, topP float64, topK int) (string, error) {
	// Encode prompt
	promptTokens := lm.Encode(prompt)
	if len(promptTokens) == 0 {
		return "", fmt.Errorf("empty prompt after tokenization")
	}

	// Prepare batch (single sequence)
	batchTokens := [][]int{promptTokens}

	// Generate tokens
	generatedTokens := lm.gpt.Generate(
		batchTokens,
		maxLength,
		temperature,
		topP,
		topK,
	)

	// Decode generated tokens
	resultTokens := generatedTokens[0]
	resultText := lm.Decode(resultTokens)

	return resultText, nil
}

// Forward performs a forward pass and returns logits
// text: input text
// Returns: logits tensor and any error
func (lm *LanguageModel) Forward(text string) (*transformer.Tensor, error) {
	tokens := lm.Encode(text)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty text after tokenization")
	}

	// Prepare batch
	batchTokens := [][]int{tokens}

	// Forward pass
	logits, _ := lm.gpt.Forward(batchTokens, nil)

	return logits, nil
}

// GetLogitsForGeneration gets logits for the last token position (used for sampling)
// text: input text
// Returns: logits for the last token as float32 slice
func (lm *LanguageModel) GetLogitsForGeneration(text string) ([]float32, error) {
	logits, err := lm.Forward(text)
	if err != nil {
		return nil, err
	}

	// Extract logits for the last position
	// logits shape: (batch=1, seq_len, vocab_size)
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	lastLogits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		lastLogits[v] = float32(logits.At(0, seqLen-1, v))
	}

	return lastLogits, nil
}

// SampleNextToken samples the next token using the configured sampler
func (lm *LanguageModel) SampleNextToken(text string) (int, error) {
	logits, err := lm.GetLogitsForGeneration(text)
	if err != nil {
		return 0, err
	}

	return lm.sampleToken(logits, lm.config.Temperature, lm.config.TopP, lm.config.TopK), nil
}

// sampleToken samples a token from logits using temperature, top-k, and top-p
func (lm *LanguageModel) sampleToken(logits []float32, temperature, topP float64, topK int) int {
	// Apply temperature
	if temperature != 1.0 && temperature != 0.0 {
		for i := range logits {
			logits[i] /= float32(temperature)
		}
	}

	// Apply top-k filtering
	if topK > 0 && topK < len(logits) {
		logits = lm.applyTopK(logits, topK)
	}

	// Apply top-p (nucleus) filtering
	if topP < 1.0 {
		logits = lm.applyTopP(logits, topP)
	}

	// Greedy sampling if temperature is 0
	if temperature == 0.0 {
		maxIdx := 0
		maxVal := float32(math.Inf(-1))
		for i, v := range logits {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		return maxIdx
	}

	// Convert to probabilities
	probs := lm.softmax(logits)

	// Sample from distribution
	r := rand.Float64()
	cumProb := 0.0
	for i, prob := range probs {
		cumProb += prob
		if r <= cumProb {
			return i
		}
	}

	// Fallback
	return len(logits) - 1
}

// applyTopK keeps only the top-k highest probability tokens
func (lm *LanguageModel) applyTopK(logits []float32, k int) []float32 {
	if k >= len(logits) {
		return logits
	}

	// Create index-value pairs
	type kv struct {
		idx int
		val float32
	}

	pairs := make([]kv, len(logits))
	for i, score := range logits {
		pairs[i] = kv{i, score}
	}

	// Sort by value (descending)
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].val < pairs[j].val {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Zero out everything except top-k
	result := make([]float32, len(logits))
	for i := 0; i < k; i++ {
		result[pairs[i].idx] = pairs[i].val
	}

	// Set others to -inf
	for i := range result {
		if result[i] == 0 {
			result[i] = float32(math.Inf(-1))
		}
	}

	return result
}

// applyTopP keeps only tokens that make up the top-p probability mass
func (lm *LanguageModel) applyTopP(logits []float32, p float64) []float32 {
	probs := lm.softmax(logits)

	// Create index-probability pairs
	type kv struct {
		idx  int
		prob float64
	}

	pairs := make([]kv, len(probs))
	for i, prob := range probs {
		pairs[i] = kv{i, prob}
	}

	// Sort by probability (descending)
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].prob < pairs[j].prob {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Find cutoff point
	cumProb := 0.0
	cutoff := len(pairs)
	for i, pair := range pairs {
		cumProb += pair.prob
		if cumProb >= p {
			cutoff = i + 1
			break
		}
	}

	// Zero out everything except top-p tokens
	result := make([]float32, len(logits))
	for i := 0; i < cutoff; i++ {
		result[pairs[i].idx] = logits[pairs[i].idx]
	}

	// Set others to -inf
	for i := range result {
		if result[i] == 0 {
			result[i] = float32(math.Inf(-1))
		}
	}

	return result
}

// softmax computes softmax of logits
func (lm *LanguageModel) softmax(logits []float32) []float64 {
	// Find max for numerical stability
	maxVal := float64(logits[0])
	for _, v := range logits {
		if float64(v) > maxVal {
			maxVal = float64(v)
		}
	}

	// Compute exp and sum
	sum := 0.0
	expVals := make([]float64, len(logits))
	for i, v := range logits {
		if v == float32(math.Inf(-1)) {
			expVals[i] = 0.0
		} else {
			expVals[i] = math.Exp(float64(v) - maxVal)
			sum += expVals[i]
		}
	}

	// Normalize
	probs := make([]float64, len(logits))
	for i, v := range expVals {
		probs[i] = v / sum
	}

	return probs
}

// Train performs a single training step
// inputs: batch of input token sequences
// targets: batch of target token sequences
// learningRate: learning rate for gradient descent
// Returns: loss value
func (lm *LanguageModel) Train(inputs [][]int, targets [][]int, learningRate float64) float64 {
	// Forward pass
	_, loss := lm.gpt.Forward(inputs, targets)

	// For now, we don't implement actual backpropagation
	// This is a placeholder for future gradient computation
	// In a full implementation, we'd compute gradients and update parameters

	return loss
}

// GetGPT returns the underlying GPT model
func (lm *LanguageModel) GetGPT() *GPT {
	return lm.gpt
}

// GetTokenizer returns the underlying tokenizer
func (lm *LanguageModel) GetTokenizer() *tokenizer.BPE {
	return lm.tokenizer
}

// SetSampler changes the sampling strategy
func (lm *LanguageModel) SetSampler(samplerType SamplingMethod) {
	lm.config.SamplerType = samplerType
}

// CreateTrainingData creates training sequences from text
// text: raw text
// seqLength: length of each training sequence
// Returns: list of training sequences (each as []int)
func (lm *LanguageModel) CreateTrainingData(text string, seqLength int) [][]int {
	tokens := lm.Encode(text)

	if len(tokens) <= seqLength {
		return [][]int{tokens}
	}

	// Create overlapping sequences
	var sequences [][]int
	for i := 0; i <= len(tokens)-seqLength; i += seqLength / 2 {
		end := i + seqLength
		if end > len(tokens) {
			end = len(tokens)
		}
		sequence := make([]int, end-i)
		copy(sequence, tokens[i:end])
		sequences = append(sequences, sequence)

		if end == len(tokens) {
			break
		}
	}

	return sequences
}

// LoadShakespeareData loads Shakespeare's text for training
// This is a simplified version - in practice you'd load from file
func LoadShakespeareData() string {
	// Placeholder - in real implementation, load from file
	return `To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.`
}

// LoadTinyStoriesData loads a small sample of TinyStories-like data
// This is a simplified version - in practice you'd load from file
func LoadTinyStoriesData() string {
	return `Once upon a time, there was a little cat named Whiskers. Whiskers loved to play with yarn. One day, Whiskers found a big ball of red yarn. The yarn was soft and bouncy. Whiskers played with the yarn all day. He rolled it and tossed it around. At night, Whiskers curled up with the yarn. He was very happy.

Once upon a time, there was a little dog named Spot. Spot had black and white spots. He loved to run in the park. Every morning, Spot went to the park with his owner. They played fetch with a ball. Spot could run very fast. He always caught the ball. At home, Spot liked to sleep on his soft bed. He dreamed about running in the park.

Once upon a time, there was a little bird named Tweety. Tweety had yellow feathers. She lived in a big tree. Every morning, Tweety sang a happy song. The song woke up all the animals. Tweety flew around and found food. She ate seeds and worms. At night, Tweety went back to her nest. She slept under the stars.`
}
