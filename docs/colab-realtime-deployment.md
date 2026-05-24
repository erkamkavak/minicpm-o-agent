# Colab Realtime Deployment

This guide runs the MiniCPM-o backend on a Colab GPU and exposes the gateway so
your realtime audio client can connect to your own `/v1/realtime` endpoint
instead of the hosted public API.

Notebook version: [`colab-realtime-deployment.ipynb`](colab-realtime-deployment.ipynb).

Use a GPU runtime. A larger GPU such as A100 or L4 is strongly preferred; small
free-tier GPUs may run out of memory when loading MiniCPM-o 4.5.

## 1. Check GPU

```bash
!nvidia-smi
```

## 2. Clone This Repo

This clones `erkamkavak/minicpm-o-agent` into `/content/minicpm-o-agent`:

```bash
%cd /content
!git clone --branch main https://github.com/erkamkavak/minicpm-o-agent.git /content/minicpm-o-agent
%cd /content/minicpm-o-agent
!git status --short
```

## 3. Install Dependencies

The service launcher expects `.venv/base`, so use the repo install script.
Colab's Python image may be missing the matching `venv` package, so install it
before running `install.sh`:

```bash
%cd /content/minicpm-o-agent
!sudo apt-get update -qq
!PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && (sudo apt-get install -y -qq python${PYVER}-venv || sudo apt-get install -y -qq python3-venv)
!PYTHON=python3 SKIP_FLASH_ATTN=1 bash install.sh
```

If installation fails because Colab already has an incompatible package pinned
after the venv package is installed, restart the runtime and run the install
cell again before importing the project.

## 4. Configure The Model

For Hugging Face download at first model load:

```bash
%%bash
cd /content/minicpm-o-agent
cp configs/config.example.json config.json
python - <<'PY'
import json
from pathlib import Path

path = Path("config.json")
cfg = json.loads(path.read_text())
cfg["model"]["model_path"] = "openbmb/MiniCPM-o-4_5"
cfg["model"]["pt_path"] = None
cfg["model"]["attn_implementation"] = "auto"
cfg["service"]["gateway_port"] = 8006
cfg["service"]["worker_base_port"] = 22400
cfg["service"]["compile"] = False
path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False))
PY
cat config.json
```

If the model is gated or you hit rate limits, log in first:

```bash
!/content/minicpm-o-agent/.venv/base/bin/huggingface-cli login
```

## 5. Start Worker And Gateway

Run HTTP mode inside Colab. The public tunnel will provide HTTPS/WSS outside.

```bash
%cd /content/minicpm-o-agent
!CUDA_VISIBLE_DEVICES=0 bash start_all.sh --http
```

The first start can take several minutes because model weights are downloaded
and loaded. If the cell finishes with a failed worker, inspect:

```bash
!tail -200 /content/minicpm-o-agent/tmp/worker_0.log
!tail -200 /content/minicpm-o-agent/tmp/gateway.log
```

Health checks:

```bash
!curl -s http://127.0.0.1:8006/health
!curl -s http://127.0.0.1:8006/workers
```

## 6. Expose The Gateway With Cloudflare Tunnel

Install `cloudflared`:

```bash
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
```

Start a tunnel:

```bash
!cloudflared tunnel --url http://127.0.0.1:8006 --no-autoupdate
```

Copy the printed `https://...trycloudflare.com` URL. Your Realtime WebSocket URL
is the same host with `wss` and the realtime path:

```text
wss://YOUR-TUNNEL.trycloudflare.com/v1/realtime?mode=audio
```

## 7. Use The Colab Endpoint

Paste the Colab tunnel endpoint into your realtime audio client:

```text
wss://YOUR-TUNNEL.trycloudflare.com/v1/realtime?mode=audio
```

The gateway also exposes HTTP endpoints on the tunnel host, such as
`https://YOUR-TUNNEL.trycloudflare.com/health`.

## 8. Probe Without The Browser UI

You can also test the Colab endpoint from the repo probe:

```bash
%cd /content/minicpm-o-agent/examples/realtime
!/content/minicpm-o-agent/.venv/base/bin/python audio_probe.py \
  --url https://YOUR-TUNNEL.trycloudflare.com \
  --input-wav assets/test.wav \
  --region colab \
  --pretty-json
```

## Troubleshooting

- `worker_0.log` says CUDA out of memory: switch to a larger Colab GPU, reduce
  other processes, or restart the runtime before starting the service.
- Client connects but no replies: confirm the endpoint uses `wss://...` and
  not `https://...`, speak clearly, and use headphones if using speakers.
- `session.queue_done` never appears: check `/workers`; the gateway may have no
  healthy worker.
- Cloudflare tunnel closes: restart the tunnel cell and paste the new URL into
  the UI.
- Model download fails: run `huggingface-cli login`, accept any required model
  terms on Hugging Face, then restart the worker.

To stop the Colab service:

```bash
!kill $(cat /content/minicpm-o-agent/tmp/*.pid 2>/dev/null) 2>/dev/null || true
```
