package small_language_model

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"

	"github.com/tektwister/ai_engineering/transformer"
)

// GPT2Config represents the configuration of a GPT-2 model
type GPT2Config struct {
	VocabSize      int    `json:"vocab_size"`
	NPositions     int    `json:"n_positions"`
	NEmbd          int    `json:"n_embd"`
	NCtx           int    `json:"n_ctx"`
	NHead          int    `json:"n_head"`
	NLayer         int    `json:"n_layer"`
	ActivationFunc string `json:"activation_function"`
}

// GPT2Weights represents the weights of a GPT-2 model
type GPT2Weights struct {
	Embeddings struct {
		TokenEmbeddings    [][]float64 `json:"wte"`
		PositionEmbeddings [][]float64 `json:"wpe"`
	} `json:"embeddings"`

	Layers []GPT2LayerWeights `json:"layers"`

	LNFinal struct {
		Gamma []float64 `json:"g"`
		Beta  []float64 `json:"b"`
	} `json:"ln_f"`

	LMHead [][]float64 `json:"lm_head"`
}

// GPT2LayerWeights represents weights for a single transformer layer
type GPT2LayerWeights struct {
	LN1 struct {
		Gamma []float64 `json:"ln_1_g"`
		Beta  []float64 `json:"ln_1_b"`
	} `json:"ln_1"`

	Attention struct {
		CAttn struct {
			Weight [][]float64 `json:"c_attn_w"`
			Bias   []float64   `json:"c_attn_b"`
		} `json:"c_attn"`

		CProj struct {
			Weight [][]float64 `json:"c_proj_w"`
			Bias   []float64   `json:"c_proj_b"`
		} `json:"c_proj"`
	} `json:"attn"`

	LN2 struct {
		Gamma []float64 `json:"ln_2_g"`
		Beta  []float64 `json:"ln_2_b"`
	} `json:"ln_2"`

	MLP struct {
		CFC struct {
			Weight [][]float64 `json:"c_fc_w"`
			Bias   []float64   `json:"c_fc_b"`
		} `json:"c_fc"`

		CProj struct {
			Weight [][]float64 `json:"c_proj_w"`
			Bias   []float64   `json:"c_proj_b"`
		} `json:"c_proj"`
	} `json:"mlp"`
}

// GPT2Loader handles loading GPT-2 weights
type GPT2Loader struct {
	modelDir string
}

// NewGPT2Loader creates a new GPT-2 loader
func NewGPT2Loader(modelDir string) *GPT2Loader {
	return &GPT2Loader{modelDir: modelDir}
}

// LoadConfig loads the model configuration
func (l *GPT2Loader) LoadConfig() (*GPT2Config, error) {
	configPath := filepath.Join(l.modelDir, "config.json")
	file, err := os.Open(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	var config GPT2Config
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&config)
	if err != nil {
		return nil, fmt.Errorf("failed to decode config: %w", err)
	}

	return &config, nil
}

// LoadWeights loads the model weights (simplified - would need actual weight files)
func (l *GPT2Loader) LoadWeights(config *GPT2Config) (*GPT2Weights, error) {
	// This is a simplified version. In practice, GPT-2 weights are stored
	// in binary formats like safetensors or pickle files.
	// Here we create mock weights for demonstration.

	weights := &GPT2Weights{}

	// Initialize embeddings
	weights.Embeddings.TokenEmbeddings = make([][]float64, config.VocabSize)
	for i := range weights.Embeddings.TokenEmbeddings {
		weights.Embeddings.TokenEmbeddings[i] = make([]float64, config.NEmbd)
		// Initialize with small random values (would load from file)
		for j := range weights.Embeddings.TokenEmbeddings[i] {
			weights.Embeddings.TokenEmbeddings[i][j] = (0.02 * (0.5 - math.Sqrt(2*math.Pi))) * math.Sin(float64(i*j))
		}
	}

	weights.Embeddings.PositionEmbeddings = make([][]float64, config.NPositions)
	for i := range weights.Embeddings.PositionEmbeddings {
		weights.Embeddings.PositionEmbeddings[i] = make([]float64, config.NEmbd)
		// Initialize with positional encoding pattern
		for j := range weights.Embeddings.PositionEmbeddings[i] {
			if j%2 == 0 {
				weights.Embeddings.PositionEmbeddings[i][j] = math.Sin(float64(i) / math.Pow(10000, float64(j)/float64(config.NEmbd)))
			} else {
				weights.Embeddings.PositionEmbeddings[i][j] = math.Cos(float64(i) / math.Pow(10000, float64(j-1)/float64(config.NEmbd)))
			}
		}
	}

	// Initialize layers
	weights.Layers = make([]GPT2LayerWeights, config.NLayer)
	for l := 0; l < config.NLayer; l++ {
		layer := &weights.Layers[l]

		// Layer norms
		layer.LN1.Gamma = make([]float64, config.NEmbd)
		layer.LN1.Beta = make([]float64, config.NEmbd)
		for i := range layer.LN1.Gamma {
			layer.LN1.Gamma[i] = 1.0 // Initialize gamma to 1
			// Beta initialized to 0
		}

		// Attention weights (c_attn combines Q, K, V)
		attnSize := 3 * config.NEmbd // Q + K + V
		layer.Attention.CAttn.Weight = make([][]float64, config.NEmbd)
		for i := range layer.Attention.CAttn.Weight {
			layer.Attention.CAttn.Weight[i] = make([]float64, attnSize)
			for j := range layer.Attention.CAttn.Weight[i] {
				layer.Attention.CAttn.Weight[i][j] = 0.02 * (0.5 - math.Sqrt(2*math.Pi))
			}
		}
		layer.Attention.CAttn.Bias = make([]float64, attnSize)

		// Attention projection
		layer.Attention.CProj.Weight = make([][]float64, attnSize)
		for i := range layer.Attention.CProj.Weight {
			layer.Attention.CProj.Weight[i] = make([]float64, config.NEmbd)
			for j := range layer.Attention.CProj.Weight[i] {
				layer.Attention.CProj.Weight[i][j] = 0.02 * (0.5 - math.Sqrt(2*math.Pi))
			}
		}
		layer.Attention.CProj.Bias = make([]float64, config.NEmbd)

		// Second layer norm
		layer.LN2.Gamma = make([]float64, config.NEmbd)
		layer.LN2.Beta = make([]float64, config.NEmbd)
		for i := range layer.LN2.Gamma {
			layer.LN2.Gamma[i] = 1.0
		}

		// MLP weights
		mlpSize := 4 * config.NEmbd // GPT-2 uses 4x expansion
		layer.MLP.CFC.Weight = make([][]float64, config.NEmbd)
		for i := range layer.MLP.CFC.Weight {
			layer.MLP.CFC.Weight[i] = make([]float64, mlpSize)
			for j := range layer.MLP.CFC.Weight[i] {
				layer.MLP.CFC.Weight[i][j] = 0.02 * (0.5 - math.Sqrt(2*math.Pi))
			}
		}
		layer.MLP.CFC.Bias = make([]float64, mlpSize)

		layer.MLP.CProj.Weight = make([][]float64, mlpSize)
		for i := range layer.MLP.CProj.Weight {
			layer.MLP.CProj.Weight[i] = make([]float64, config.NEmbd)
			for j := range layer.MLP.CProj.Weight[i] {
				layer.MLP.CProj.Weight[i][j] = 0.02 * (0.5 - math.Sqrt(2*math.Pi))
			}
		}
		layer.MLP.CProj.Bias = make([]float64, config.NEmbd)
	}

	// Final layer norm
	weights.LNFinal.Gamma = make([]float64, config.NEmbd)
	weights.LNFinal.Beta = make([]float64, config.NEmbd)
	for i := range weights.LNFinal.Gamma {
		weights.LNFinal.Gamma[i] = 1.0
	}

	// Language model head (tied to token embeddings in GPT-2)
	weights.LMHead = make([][]float64, config.VocabSize)
	for i := range weights.LMHead {
		weights.LMHead[i] = make([]float64, config.NEmbd)
		copy(weights.LMHead[i], weights.Embeddings.TokenEmbeddings[i])
	}

	return weights, nil
}

// LoadModel loads a complete GPT-2 model
func (l *GPT2Loader) LoadModel() (*GPT, error) {
	// Load config
	config, err := l.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	fmt.Printf("Loading GPT-2 model with config: %+v\n", config)

	// Create GPT config from GPT-2 config
	gptConfig := &GPTConfig{
		VocabSize: config.VocabSize,
		BlockSize: config.NCtx,
		NLayer:    config.NLayer,
		NHead:     config.NHead,
		NEmbd:     config.NEmbd,
		Dropout:   0.0, // No dropout for inference
		Bias:      true,
	}

	// Create model
	gpt := NewGPT(gptConfig)

	// Load weights
	weights, err := l.LoadWeights(config)
	if err != nil {
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	// Apply weights to model
	err = l.applyWeightsToModel(gpt, weights, config)
	if err != nil {
		return nil, fmt.Errorf("failed to apply weights: %w", err)
	}

	fmt.Println("GPT-2 model loaded successfully!")
	return gpt, nil
}

// applyWeightsToModel applies loaded weights to the GPT model
func (l *GPT2Loader) applyWeightsToModel(gpt *GPT, weights *GPT2Weights, config *GPT2Config) error {
	// Apply token embeddings
	for i := 0; i < config.VocabSize; i++ {
		for j := 0; j < config.NEmbd; j++ {
			gpt.tokenEmb.Weight.Set(weights.Embeddings.TokenEmbeddings[i][j], i, j)
		}
	}

	// Apply position embeddings (crop to block size)
	maxPos := min(config.NPositions, gpt.config.BlockSize)
	for i := 0; i < maxPos; i++ {
		for j := 0; j < config.NEmbd; j++ {
			gpt.posEmb.Weight.Set(weights.Embeddings.PositionEmbeddings[i][j], i, j)
		}
	}

	// Apply layer weights
	for l := 0; l < config.NLayer; l++ {
		layerWeights := weights.Layers[l]
		block := gpt.blocks[l]

		// Layer norms
		for i := 0; i < config.NEmbd; i++ {
			block.LN1.Gamma.Data[i] = layerWeights.LN1.Gamma[i]
			block.LN1.Beta.Data[i] = layerWeights.LN1.Beta[i]

			block.LN2.Gamma.Data[i] = layerWeights.LN2.Gamma[i]
			block.LN2.Beta.Data[i] = layerWeights.LN2.Beta[i]
		}

		// Attention weights
		// GPT-2 uses a combined c_attn projection for Q, K, V
		// We need to split this into separate Q, K, V matrices
		attnSize := 3 * config.NEmbd

		for i := 0; i < config.NEmbd; i++ {
			for j := 0; j < attnSize; j++ {
				val := layerWeights.Attention.CAttn.Weight[i][j]

				// Split into Q, K, V parts
				if j < config.NEmbd {
					// Q part
					gpt.setAttentionWeight(block.Attn.WQ, i, j, val, config)
				} else if j < 2*config.NEmbd {
					// K part
					gpt.setAttentionWeight(block.Attn.WK, i, j-config.NEmbd, val, config)
				} else {
					// V part
					gpt.setAttentionWeight(block.Attn.WV, i, j-2*config.NEmbd, val, config)
				}
			}
		}

		// Attention biases
		for j := 0; j < attnSize; j++ {
			val := layerWeights.Attention.CAttn.Bias[j]
			if j < config.NEmbd {
				gpt.setAttentionBias(block.Attn.WQ, j, val, config)
			} else if j < 2*config.NEmbd {
				gpt.setAttentionBias(block.Attn.WK, j-config.NEmbd, val, config)
			} else {
				gpt.setAttentionBias(block.Attn.WV, j-2*config.NEmbd, val, config)
			}
		}

		// Attention output projection
		for i := 0; i < attnSize; i++ {
			for j := 0; j < config.NEmbd; j++ {
				block.Attn.WO.Weight.Set(layerWeights.Attention.CProj.Weight[i][j], i, j)
			}
		}
		for j := 0; j < config.NEmbd; j++ {
			block.Attn.WO.Bias.Data[j] = layerWeights.Attention.CProj.Bias[j]
		}

		// MLP weights
		mlpSize := 4 * config.NEmbd
		for i := 0; i < config.NEmbd; i++ {
			for j := 0; j < mlpSize; j++ {
				block.FFN.Linear1.Weight.Set(layerWeights.MLP.CFC.Weight[i][j], i, j)
			}
			if block.FFN.Linear1.Bias != nil {
				block.FFN.Linear1.Bias.Data[i] = layerWeights.MLP.CFC.Bias[i]
			}
		}

		for i := 0; i < mlpSize; i++ {
			for j := 0; j < config.NEmbd; j++ {
				block.FFN.Linear2.Weight.Set(layerWeights.MLP.CProj.Weight[i][j], i, j)
			}
		}
		for j := 0; j < config.NEmbd; j++ {
			if block.FFN.Linear2.Bias != nil {
				block.FFN.Linear2.Bias.Data[j] = layerWeights.MLP.CProj.Bias[j]
			}
		}
	}

	// Final layer norm
	for i := 0; i < config.NEmbd; i++ {
		gpt.lnF.Gamma.Data[i] = weights.LNFinal.Gamma[i]
		gpt.lnF.Beta.Data[i] = weights.LNFinal.Beta[i]
	}

	// Language model head
	for i := 0; i < config.VocabSize; i++ {
		for j := 0; j < config.NEmbd; j++ {
			gpt.lmHead.Weight.Set(weights.LMHead[i][j], i, j)
		}
	}

	return nil
}

// Helper methods for setting attention weights/biases
func (gpt *GPT) setAttentionWeight(linear *transformer.Linear, row, col int, val float64, config *GPT2Config) {
	// In GPT-2, attention weights are stored in a specific order
	// We need to map this to our multi-head attention format
	headSize := config.NEmbd / config.NHead
	headIdx := col / headSize
	withinHeadIdx := col % headSize

	targetRow := row
	targetCol := headIdx*headSize + withinHeadIdx

	linear.Weight.Set(val, targetRow, targetCol)
}

func (gpt *GPT) setAttentionBias(linear *transformer.Linear, col int, val float64, config *GPT2Config) {
	if linear.Bias != nil {
		headSize := config.NEmbd / config.NHead
		headIdx := col / headSize
		withinHeadIdx := col % headSize
		targetCol := headIdx*headSize + withinHeadIdx
		linear.Bias.Data[targetCol] = val
	}
}

// CreateMockGPT2Model creates a mock GPT-2 model for testing
func CreateMockGPT2Model() (*GPT, error) {
	loader := NewGPT2Loader("./gpt2_mock")
	return loader.LoadModel()
}

// SaveModel saves the current model weights (for checkpointing)
func (gpt *GPT) SaveModel(filepath string) error {
	// Simplified checkpoint saving
	// In practice, this would save all model weights
	data := map[string]interface{}{
		"config": gpt.config,
		// Would include all tensor data here
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(filepath, jsonData, 0644)
}

// LoadModel loads a checkpointed model
func LoadModel(filepath string) (*GPT, error) {
	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	var modelData map[string]interface{}
	err = json.Unmarshal(data, &modelData)
	if err != nil {
		return nil, err
	}

	// Extract config and recreate model
	configData := modelData["config"].(map[string]interface{})
	config := &GPTConfig{
		VocabSize: int(configData["VocabSize"].(float64)),
		BlockSize: int(configData["BlockSize"].(float64)),
		NLayer:    int(configData["NLayer"].(float64)),
		NHead:     int(configData["NHead"].(float64)),
		NEmbd:     int(configData["NEmbd"].(float64)),
		Dropout:   configData["Dropout"].(float64),
		Bias:      configData["Bias"].(bool),
	}

	return NewGPT(config), nil
}
