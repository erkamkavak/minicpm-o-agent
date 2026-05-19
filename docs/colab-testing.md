# Colab Testing Workflow

This repo now has two test layers:

1. Lightweight CPU tests that should run locally without the real MiniCPM weights.
2. Real model validation that is better suited for a Colab GPU runtime.

## Local CPU Checks

From the repo root:

```bash
PYTHONPATH=src pytest \
  tests/test_tool_calling.py \
  tests/test_duplex_system_prompt_update.py \
  tests/test_schemas.py
```

These tests verify:

- tool/function schemas serialize correctly,
- `<tool_call>...</tool_call>` output is parsed,
- the duplex system prompt cache span can be rebuilt while preserving later unit cache values.

If local `torch` is not installed, `tests/test_duplex_system_prompt_update.py` is skipped.
That is expected on low-storage machines; run it in Colab for the cache-level check.

## Getting Changes Into Colab

Recommended workflow:

1. Commit the local changes.
2. Push the branch to a GitHub repo you can access from Colab.
3. In Colab, clone that branch:

```bash
git clone --branch YOUR_BRANCH https://github.com/YOUR_ORG/YOUR_REPO.git
cd YOUR_REPO
```

If you cannot push to GitHub, create a patch locally:

```bash
git diff > minicpmo-duplex-tools.patch
```

Then upload the patch to Colab and apply it after cloning the base repo:

```bash
git apply /content/minicpmo-duplex-tools.patch
```

## Colab Setup

Use a GPU runtime. Then run:

```bash
cd /content/YOUR_REPO
python -m pip install -U pip
python -m pip install -e ".[dev]" || python -m pip install -e .
python -m pip install pytest soundfile librosa websockets fastapi uvicorn
```

If the project extras are incomplete, install the model stack used by your environment:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers accelerate sentencepiece pillow
```

## Run Lightweight Tests In Colab

```bash
PYTHONPATH=src pytest -q \
  tests/test_tool_calling.py \
  tests/test_duplex_system_prompt_update.py
```

## Run Real Duplex Validation

Set paths for your model and reference audio:

```bash
export MODEL_PATH=/content/path/to/MiniCPM-o-model
export PT_PATH=/content/path/to/model.pt
export REF_AUDIO=/content/path/to/ref.wav
```

Then run the existing duplex tests:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src pytest -q tests/test_duplex.py -s
```

For the new mid-session update behavior, run or adapt a small script:

```python
from minicpmo_demo.core.processors.unified import UnifiedProcessor

processor = UnifiedProcessor.from_pretrained(
    model_path=MODEL_PATH,
    pt_path=PT_PATH,
)
duplex = processor.set_duplex_mode(ref_audio_path=REF_AUDIO)
duplex.prepare(
    system_prompt_text="You are concise.",
    tools=[{
        "type": "function",
        "function": {
            "name": "switch_agent",
            "description": "Switch persona.",
            "parameters": {
                "type": "object",
                "properties": {"persona": {"type": "string"}},
                "required": ["persona"],
            },
        },
    }],
)

assert duplex.update_system_prompt(
    system_prompt_text="You are now a poetic assistant. Keep previous context.",
    tools=[],
)
```

The key checks are:

- update returns `True`,
- a following `prefill()` and `generate()` still works,
- previous conversation context is still usable,
- the new instruction/tools affect future behavior.
