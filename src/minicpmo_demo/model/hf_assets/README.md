# Hugging Face Assets

This folder keeps upstream Hugging Face tokenizer, processor, and generation
metadata together:

- `tokenizer*.json`, `vocab.json`, `merges.txt`, `added_tokens.json`, and
  `special_tokens_map.json` describe tokenizer behavior.
- `preprocessor_config.json` describes the image/audio processor metadata.
- `generation_config.json` stores default generation settings.

The active backend loads model artifacts from the configured model path. These
files are tracked here as source assets/reference metadata rather than Python
modules.
