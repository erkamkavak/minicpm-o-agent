# MiniCPM-o 4.5 Backend Service

This repository is a backend/model-focused MiniCPM-o 4.5 inference service. It keeps the PyTorch model code, worker/gateway APIs, model assets, presets, protocol examples, tests, and backend design notes. Browser, mobile, static demo, and generated docs-site assets have been removed.

## What Is Included

- `src/minicpmo_demo/model/`: model configuration, tokenization, processing, and unified MiniCPM-o model code.
- `src/minicpmo_demo/server/worker.py`: model-loading inference worker that exposes chat, streaming, and duplex endpoints.
- `src/minicpmo_demo/server/gateway.py`: backend gateway that routes HTTP and WebSocket requests to workers and exposes FastAPI OpenAPI docs at `/docs`.
- `src/minicpmo_demo/core/`: request schemas, processor factory, mode capabilities, and backend request processors.
- `src/minicpmo_demo/gateway/`: app registry, ref-audio registry, gateway models, queueing, and worker-pool support.
- `src/minicpmo_demo/vad/`: Python VAD helpers used by the duplex/audio backend.
- `src/minicpmo_demo/tools/`: backend/model utilities such as benchmarking and torch.compile pre-compilation.
- `configs/`: example service configuration.
- `assets/`: backend presets, reference audio, samples, and VAD model assets.
- `examples/realtime/`: API probes for realtime audio/video sessions.
- `docs/`: backend protocol notes plus `docs/design/update-system-prompt-mid-session.md`.
- `tests/`: Python backend/API tests.

## Requirements

- Linux
- NVIDIA GPU with CUDA support
- Python 3.10
- FFmpeg
- PyTorch 2.8.0 and project dependencies from `requirements.txt`

## Setup

```bash
python3.10 -m venv .venv/base
source .venv/base/bin/activate
pip install "torch==2.8.0" "torchaudio==2.8.0"
pip install -r requirements.txt
cp configs/config.example.json config.json
```

Set `model.model_path` in `config.json` to a local model directory or a Hugging Face model id such as `openbmb/MiniCPM-o-4_5`.

## Run

Start one worker per visible GPU plus the gateway:

```bash
CUDA_VISIBLE_DEVICES=0 bash start_all.sh
```

The gateway defaults to `https://localhost:8006`. Use `bash start_all.sh --http` for local HTTP mode.

Direct module entry points use the package under `src`:

```bash
PYTHONPATH=src .venv/base/bin/python -m minicpmo_demo.server.worker --worker-index 0
PYTHONPATH=src .venv/base/bin/python -m minicpmo_demo.server.gateway
PYTHONPATH=src .venv/base/bin/python -m minicpmo_demo.tools.precompile
PYTHONPATH=src .venv/base/bin/python -m minicpmo_demo.tools.benchmark
```

Useful endpoints:

- `GET /health`
- `GET /status`
- `GET /workers`
- `POST /api/chat`
- `WS /ws/chat`
- `WS /ws/half_duplex/{session_id}`
- `WS /ws/duplex/{session_id}`
- `WS /v1/realtime`
- `GET /docs`

## Docker

```bash
docker build -t minicpm-o-backend .
docker run --gpus all -p 8006:8006 -v "$PWD/workspace:/workspace" minicpm-o-backend
```

Place persistent `config.json`, model weights, certs, session data, and compile cache under `workspace/` as described in `docker-entrypoint.sh`.

## Plan Notes

The current design plan is in `docs/design/update-system-prompt-mid-session.md`. It targets backend/model changes for replacing duplex system prompts mid-session while preserving KV cache history. The plan matches real components in this codebase, but it needs implementation-level adjustment before coding, especially around ref-audio positioning, pending finalize handling, and validation coverage.
