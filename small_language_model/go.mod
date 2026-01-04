module github.com/tektwister/ai_engineering/small_language_model

go 1.25.5

require (
	github.com/tektwister/ai_engineering/logit_processor v0.0.0
	github.com/tektwister/ai_engineering/tokenizer v0.0.0
	github.com/tektwister/ai_engineering/transformer v0.0.0
)

replace (
	github.com/tektwister/ai_engineering/logit_processor => ../logit_processor
	github.com/tektwister/ai_engineering/tokenizer => ../tokenizer
	github.com/tektwister/ai_engineering/transformer => ../transformer
)
