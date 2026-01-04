package config

import (
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
)

// LLMConfig holds the configuration for LLM providers.
type LLMConfig struct {
	ProviderName string
	APIKey       string
	BaseURL      string
	OrgID        string
}

// Load loads the LLM configuration from environment variables.
// It attempts to find a .env file in the current or parent directories.
func Load() (*LLMConfig, error) {
	// Try to load .env from current or parent directories
	_ = loadEnvFile()

	providerName := os.Getenv("LLM_PROVIDER")
	if providerName == "" {
		providerName = "openai" // Default
	}

	apiKey := os.Getenv("LLM_API_KEY")
	// Allow empty API key if the provider doesn't strictly need it (e.g. some local setups),
	// but generally warn or error in the application layer if missing.

	return &LLMConfig{
		ProviderName: providerName,
		APIKey:       apiKey,
		BaseURL:      os.Getenv("LLM_BASE_URL"), // Or OPENAI_BASE_URL compatibility
		OrgID:        os.Getenv("LLM_ORG_ID"),
	}, nil
}

// loadEnvFile attempts to look up until it finds a .env file
func loadEnvFile() error {
	dir, err := os.Getwd()
	if err != nil {
		return err
	}

	// Look up to 5 levels
	for i := 0; i < 5; i++ {
		envPath := filepath.Join(dir, ".env")
		if _, err := os.Stat(envPath); err == nil {
			return godotenv.Load(envPath)
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	
	// Also check module root relative to this file if possible (optional debugging hack)
	// But standard lookup is usually sufficient.
	return nil
}
