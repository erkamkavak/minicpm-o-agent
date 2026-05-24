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

This installs into the Colab runtime Python directly, without creating a virtual
environment. `transformers==4.51.0` is pinned explicitly:

```bash
%cd /content/minicpm-o-agent
%pip install -q "transformers==4.51.0"
%pip install -q -r requirements.txt
%pip install -q librosa soundfile
```

If Colab does not already have PyTorch, install it too:

```python
import importlib.util, subprocess, sys
if importlib.util.find_spec("torch") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchaudio"], check=True)
```

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
!huggingface-cli login
```

## 5. Start Worker And Gateway

Do not use `start_all.sh` in the no-venv Colab flow because that launcher
expects `.venv/base/bin/python`. Start the worker and gateway directly instead:

```python
import json, os, subprocess, time, urllib.request
from pathlib import Path

REPO_DIR = Path("/content/minicpm-o-agent")
TMP_DIR = REPO_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

def read_json_url(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

env = os.environ.copy()
env["PYTHONPATH"] = str(REPO_DIR / "src")
env["CUDA_VISIBLE_DEVICES"] = "0"

worker_log = open(TMP_DIR / "worker_0.log", "w")
worker_proc = subprocess.Popen(
    ["python", "-m", "minicpmo_demo.server.worker", "--port", "22400", "--gpu-id", "0", "--worker-index", "0"],
    cwd=REPO_DIR, env=env, stdout=worker_log, stderr=subprocess.STDOUT, text=True,
)
(TMP_DIR / "worker_0.pid").write_text(str(worker_proc.pid))

while True:
    try:
        health = read_json_url("http://127.0.0.1:22400/health", timeout=3)
        if health.get("model_loaded"):
            print("Worker ready", health)
            break
    except Exception:
        pass
    print("Waiting for worker to load model...")
    time.sleep(30)

gateway_log = open(TMP_DIR / "gateway.log", "w")
gateway_proc = subprocess.Popen(
    ["python", "-m", "minicpmo_demo.server.gateway", "--port", "8006", "--workers", "localhost:22400", "--http"],
    cwd=REPO_DIR, env=env, stdout=gateway_log, stderr=subprocess.STDOUT, text=True,
)
(TMP_DIR / "gateway.pid").write_text(str(gateway_proc.pid))
time.sleep(3)
print("Gateway health", read_json_url("http://127.0.0.1:8006/health"))
```

If startup fails, inspect:

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
!PYTHONPATH=/content/minicpm-o-agent/src python audio_probe.py \
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
