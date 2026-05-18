# Agent Guide

This repository is a backend/model-focused MiniCPM-o 4.5 inference service.
Frontend, mobile, static demo, and generated docs-site assets are intentionally
out of scope.

## Project Shape

- `src/minicpmo_demo/server/`: FastAPI worker, gateway, session recording, and cleanup entry points.
- `src/minicpmo_demo/core/`: request schemas, processor factory, capability routing, and mode processors.
- `src/minicpmo_demo/gateway/`: worker pool, queueing, app registry, and reference-audio registry.
- `src/minicpmo_demo/model/`: model config, tokenizer/processor files, HF model wrappers, and model assets.
- `src/minicpmo_demo/model/components/`: reusable neural-network building blocks.
- `src/minicpmo_demo/model/runtime/`: KV-cache helpers, StreamDecoder, sampling guards, and streaming helpers.
- `src/minicpmo_demo/vad/`: VAD helpers.
- `src/minicpmo_demo/tools/`: backend/model utilities such as benchmark and precompile.
- `configs/config.example.json`: tracked example config. Runtime `config.json` stays untracked.

## Working Rules

- Preserve backend/model behavior unless the task explicitly asks for logic changes.
- Prefer package imports from `minicpmo_demo.*`; do not add root-level compatibility wrappers.
- Keep Hugging Face model entry points stable unless a task specifically covers model loading.
- Keep frontend/mobile/static demo code out of this repo.
- Do not commit generated caches, local configs, model weights, sessions, or runtime output.
- Use `PYTHONPATH=src` for direct Python commands from the repo root.

## Validation

Run a syntax/import smoke check after structural changes:

```bash
PYTHONPATH=src python -m compileall src/minicpmo_demo tests
```

Run tests when dependencies are available:

```bash
PYTHONPATH=src python -m pytest tests/test_schemas.py -q
```

GPU/model behavior checks require a configured `config.json`, model weights, CUDA,
and the project dependencies installed.

## Commit Convention

Use Conventional Commits:

- `chore:` repository maintenance, asset removal, dependency metadata, or non-runtime cleanup.
- `refactor:` code movement or restructuring that should not change behavior.
- `docs:` documentation-only changes.
- `test:` test-only changes.
- `fix:` behavior-preserving bug fixes for broken runtime paths, imports, or scripts.
- `feat:` new user-visible behavior.

Keep commit subjects imperative and under about 72 characters, for example:

```text
refactor: organize backend under src package
docs: add agent workflow guidance
```
