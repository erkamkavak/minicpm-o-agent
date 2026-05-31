# MiMo Inference Optimization Experiments

Small PyTorch experiments for understanding the engineering ideas in Xiaomi's
MiMo-V2.5 inference optimization post.

These are not production kernels and they do not load MiMo weights. They are
controlled toy comparisons: each experiment builds a baseline and an optimized
variant, then reports timing or simulated work reduction.

## Files

- `mimo_optimization_experiments.py`: command-line runner with all experiments.
- `MiMo_Inference_Optimization_Experiments.ipynb`: Colab wrapper.

## Run

```bash
python mimo_optimization_experiments.py --quick --device auto
python mimo_optimization_experiments.py --experiments hybrid_swa,mtp --device cuda
python mimo_optimization_experiments.py --list
```

The local environment must have PyTorch installed. Colab normally has PyTorch
preinstalled; select a GPU runtime for more meaningful speed measurements.

The notebook is Colab-friendly: when opened in Colab, the setup cell clones
`https://github.com/erkamkavak/minicpm-o-agent.git` into
`/content/minicpm-o-agent`, changes into this experiment directory, and runs the
script from the cloned project. No separate upload of the `.py` file is needed.

## Experiment Map

- `hybrid_swa`: compares full-attention decode/KV against a 5:1 SWA/full layer
  mix with a 128-token SWA cache.
- `length_bucketing`: compares random prefill batches against length buckets.
- `scheduler`: simulates load-only FIFO routing vs prefix-cache affinity routing
  using Xiaomi's rough score shape: cache match minus normalized load.
- `encoder_batching`: compares per-request image encoder calls against
  cross-request batched preprocessing/forward.
- `mtp`: compares vanilla one-token decode against MTP-style speculative
  verification with high- and low-acceptance draft tokens.

## Public Runtime Notes

- vLLM's current MiMo-V2 MTP implementation exposes an inference-only MTP
  draft model and notes that vLLM currently uses the first MTP layer.
- vLLM's MiMo-V2 omni model implements a vision encoder with full-attention
  and window-attention blocks.
- The SGLang MiMo-V2.5 recipe documents the large-model serving knobs that
  inspired these toy experiments, including `--swa-full-tokens-ratio` and
  `--enable-multi-layer-eagle`.
