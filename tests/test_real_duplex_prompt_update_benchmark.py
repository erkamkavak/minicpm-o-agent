"""Real-model duplex system-prompt update benchmark.

This test is intended for Colab/GPU runs, not normal local CI. It compares:

1. Cache surgery:
   old prompt -> replay history -> update_system_prompt(new prompt) -> probe logits
2. Full replay:
   new prompt -> replay same history from raw inputs -> probe logits

The test records speed and divergence metrics to
``tests/results/duplex_prompt_update_benchmark.json``.

Example:

    CUDA_VISIBLE_DEVICES=0 \\
    MINICPMO45_RUN_REAL_PROMPT_UPDATE_BENCH=1 \\
    MINICPMO45_MODEL_PATH=/content/MiniCPM-o-4_5 \\
    MINICPMO45_REF_AUDIO_PATH=/content/minicpm-o-agent/tests/cases/common/ref_audio/BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav \\
    PYTHONPATH=src python -m pytest -q tests/test_real_duplex_prompt_update_benchmark.py -s
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_CASES = PROJECT_ROOT / "tests" / "cases" / "common"
DEFAULT_REF_AUDIO = (
    COMMON_CASES
    / "ref_audio"
    / "BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav"
)
DEFAULT_USER_AUDIO = COMMON_CASES / "user_audio" / "000_user_audio0.wav"
DEFAULT_RESULT_PATH = PROJECT_ROOT / "tests" / "results" / "duplex_prompt_update_benchmark.json"


def _require_real_bench_enabled() -> None:
    if os.environ.get("MINICPMO45_RUN_REAL_PROMPT_UPDATE_BENCH") != "1":
        pytest.skip("Set MINICPMO45_RUN_REAL_PROMPT_UPDATE_BENCH=1 to run the real-model benchmark")


def _existing_path_from_env(name: str, default: Path | None = None, required: bool = True) -> Path | None:
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else default
    if path is None:
        if required:
            pytest.skip(f"{name} is required for the real-model benchmark")
        return None
    if required and not path.exists():
        pytest.skip(f"{name} does not exist: {path}")
    return path


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_ms(fn: Callable[[], Any]) -> tuple[Any, float, int | None]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync_cuda()
    start = time.perf_counter()
    result = fn()
    _sync_cuda()
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_memory_mb = None
    if torch.cuda.is_available():
        peak_memory_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return result, elapsed_ms, peak_memory_mb


def _load_audio(path: Path, seconds: float) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    target_samples = max(1, int(seconds * 16000))
    if len(audio) >= target_samples:
        audio = audio[:target_samples]
    else:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    return audio.astype(np.float32, copy=False)


def _history_audio_paths() -> list[Path]:
    raw = os.environ.get("MINICPMO45_HISTORY_AUDIO_PATHS")
    if raw:
        paths = [Path(item).expanduser() for item in raw.split(",") if item.strip()]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            pytest.skip(f"Missing MINICPMO45_HISTORY_AUDIO_PATHS entries: {missing}")
        return paths

    unit_count = int(os.environ.get("MINICPMO45_HISTORY_UNITS", "3"))
    default_audio = _existing_path_from_env(
        "MINICPMO45_HISTORY_AUDIO_PATH",
        default=DEFAULT_USER_AUDIO,
        required=True,
    )
    assert default_audio is not None
    return [default_audio] * unit_count


def _history_texts() -> list[str]:
    raw = os.environ.get("MINICPMO45_HISTORY_TEXTS")
    if raw:
        raw = raw.strip()
        if raw.startswith("["):
            values = json.loads(raw)
            assert isinstance(values, list)
            return [str(value) for value in values]
        return [item.strip() for item in raw.split("||") if item.strip()]

    unit_count = int(os.environ.get("MINICPMO45_HISTORY_UNITS", "3"))
    return [
        f"User message {idx + 1}: remember project detail {idx + 1} and keep replies concise."
        for idx in range(unit_count)
    ]


def _replay_history(duplex, audio_chunks: list[np.ndarray]) -> None:
    for idx, audio in enumerate(audio_chunks):
        prefill = duplex.prefill(audio_waveform=audio)
        assert prefill.get("success"), f"prefill failed for history unit {idx}: {prefill}"

        # Force-listen keeps the replay deterministic and avoids TTS work while still
        # closing each unit with the normal generate/finalize path.
        duplex.generate(force_listen=True)
        duplex.finalize()


def _feed_text_unit(duplex, text: str) -> None:
    capability = duplex._model.duplex
    decoder = capability.decoder
    tokenizer = capability.tokenizer

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    unit_end_id = tokenizer.convert_tokens_to_ids("</unit>")

    decoder.register_unit_start()
    decoder.feed(decoder.embed_token(capability.unit_token_id))
    if token_ids:
        decoder.feed(decoder.embed_tokens(token_ids))
    decoder.feed(decoder.embed_token(capability.listen_token_id))
    decoder.feed(decoder.embed_token(unit_end_id))
    decoder.register_unit_end(
        input_type="text",
        generated_tokens=[capability.listen_token_id],
        is_listen=True,
        generated_text="",
    )


def _replay_text_history(duplex, history_texts: list[str]) -> None:
    for text in history_texts:
        _feed_text_unit(duplex, text)


def _capture_probe_logits(duplex, probe_audio: np.ndarray) -> torch.Tensor:
    prefill = duplex.prefill(audio_waveform=probe_audio)
    assert prefill.get("success"), f"probe prefill failed: {prefill}"

    pending_logits = getattr(duplex._model.duplex, "pending_logits", None)
    assert pending_logits is not None, "probe did not produce pending logits"
    assert torch.isfinite(pending_logits).all(), "probe logits contain non-finite values"
    return pending_logits.detach().float().cpu()


def _capture_text_probe_logits(duplex, probe_text: str) -> torch.Tensor:
    capability = duplex._model.duplex
    decoder = capability.decoder
    token_ids = capability.tokenizer.encode(probe_text, add_special_tokens=False)

    decoder.register_unit_start()
    decoder.feed(decoder.embed_token(capability.unit_token_id))
    if token_ids:
        logits, _ = decoder.feed(decoder.embed_tokens(token_ids), return_logits=True)
    else:
        logits, _ = decoder.feed(decoder.embed_token(capability.listen_token_id), return_logits=True)

    assert torch.isfinite(logits).all(), "probe logits contain non-finite values"
    return logits.detach().float().cpu()


def _kl_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1e-8) -> float:
    probs_a = torch.softmax(logits_a, dim=-1)
    return float(torch.sum(probs_a * (torch.log(probs_a + eps) - torch.log_softmax(logits_b, dim=-1))))


def _js_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1e-8) -> float:
    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    midpoint = 0.5 * (probs_a + probs_b)
    kl_a = torch.sum(probs_a * (torch.log(probs_a + eps) - torch.log(midpoint + eps)))
    kl_b = torch.sum(probs_b * (torch.log(probs_b + eps) - torch.log(midpoint + eps)))
    return float(0.5 * (kl_a + kl_b))


def _topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> int:
    top_a = set(torch.topk(logits_a.reshape(-1), k).indices.tolist())
    top_b = set(torch.topk(logits_b.reshape(-1), k).indices.tolist())
    return len(top_a & top_b)


def _relative_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    return float((a - b).norm() / (b.norm() + eps))


def test_real_duplex_prompt_update_speed_and_divergence() -> None:
    _require_real_bench_enabled()

    from minicpmo_demo.core.processors import UnifiedProcessor
    from minicpmo_demo.core.schemas import DuplexConfig

    model_path = _existing_path_from_env("MINICPMO45_MODEL_PATH", required=True)
    pt_path = _existing_path_from_env("MINICPMO45_PT_PATH", required=False)
    ref_audio_path = _existing_path_from_env(
        "MINICPMO45_REF_AUDIO_PATH",
        default=DEFAULT_REF_AUDIO,
        required=True,
    )
    probe_audio_path = _existing_path_from_env(
        "MINICPMO45_PROBE_AUDIO_PATH",
        default=DEFAULT_USER_AUDIO,
        required=True,
    )
    assert model_path is not None
    assert ref_audio_path is not None
    assert probe_audio_path is not None

    old_prompt = os.environ.get(
        "MINICPMO45_OLD_PROMPT",
        "Streaming Duplex Conversation! You are a concise helpful assistant.",
    )
    new_prompt = os.environ.get(
        "MINICPMO45_NEW_PROMPT",
        "Streaming Duplex Conversation! You are a concise helpful assistant. "
        "When relevant, mention that tools may be available.",
    )
    input_mode = os.environ.get("MINICPMO45_BENCH_INPUT_MODE", "audio").lower()
    assert input_mode in {"audio", "text"}
    audio_seconds = float(os.environ.get("MINICPMO45_AUDIO_CHUNK_SECONDS", "1.0"))
    result_path = Path(os.environ.get("MINICPMO45_BENCH_RESULT_PATH", str(DEFAULT_RESULT_PATH)))

    if input_mode == "audio":
        history_items = [_load_audio(path, audio_seconds) for path in _history_audio_paths()]
        probe_item = _load_audio(probe_audio_path, audio_seconds)
    else:
        history_items = _history_texts()
        probe_item = os.environ.get(
            "MINICPMO45_PROBE_TEXT",
            "Given the previous messages, what should you remember?",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = UnifiedProcessor(
        model_path=str(model_path),
        pt_path=str(pt_path) if pt_path else None,
        device=device,
        ref_audio_path=str(ref_audio_path),
        duplex_config=DuplexConfig(
            generate_audio=False,
            force_listen_count=0,
            max_new_speak_tokens_per_chunk=4,
        ),
    )
    duplex = processor.set_duplex_mode()

    # Path A: build old state, then update prompt in-place.
    duplex.prepare(system_prompt_text=old_prompt, ref_audio_path=str(ref_audio_path))
    if input_mode == "audio":
        _replay_history(duplex, history_items)
    else:
        _replay_text_history(duplex, history_items)
    cache_len_before_update = int(processor.kv_cache_length)

    update_ok, cache_surgery_ms, cache_surgery_peak_mb = _time_ms(
        lambda: duplex.update_system_prompt(system_prompt_text=new_prompt)
    )
    assert update_ok is True
    cache_len_after_update = int(processor.kv_cache_length)
    surgery_logits = (
        _capture_probe_logits(duplex, probe_item)
        if input_mode == "audio"
        else _capture_text_probe_logits(duplex, probe_item)
    )

    # Path B: reset and replay the same raw history under the new prompt.
    def _full_replay_path() -> None:
        duplex.prepare(system_prompt_text=new_prompt, ref_audio_path=str(ref_audio_path))
        if input_mode == "audio":
            _replay_history(duplex, history_items)
        else:
            _replay_text_history(duplex, history_items)

    _, full_replay_ms, full_replay_peak_mb = _time_ms(_full_replay_path)
    cache_len_after_full_replay = int(processor.kv_cache_length)
    full_replay_logits = (
        _capture_probe_logits(duplex, probe_item)
        if input_mode == "audio"
        else _capture_text_probe_logits(duplex, probe_item)
    )

    metrics = {
        "model_path": str(model_path),
        "pt_path": str(pt_path) if pt_path else None,
        "device": device,
        "input_mode": input_mode,
        "history_units": len(history_items),
        "audio_chunk_seconds": audio_seconds if input_mode == "audio" else None,
        "probe_text_chars": len(probe_item) if input_mode == "text" else None,
        "old_prompt_chars": len(old_prompt),
        "new_prompt_chars": len(new_prompt),
        "cache_len_before_update": cache_len_before_update,
        "cache_len_after_update": cache_len_after_update,
        "cache_len_after_full_replay": cache_len_after_full_replay,
        "cache_surgery_ms": cache_surgery_ms,
        "full_replay_ms": full_replay_ms,
        "speedup": full_replay_ms / cache_surgery_ms if cache_surgery_ms else None,
        "cache_surgery_peak_memory_mb": cache_surgery_peak_mb,
        "full_replay_peak_memory_mb": full_replay_peak_mb,
        "kl_surgery_to_full": _kl_from_logits(surgery_logits, full_replay_logits),
        "kl_full_to_surgery": _kl_from_logits(full_replay_logits, surgery_logits),
        "js_divergence": _js_from_logits(surgery_logits, full_replay_logits),
        "relative_l2_logits": _relative_l2(surgery_logits, full_replay_logits),
        "max_abs_logit_delta": float((surgery_logits - full_replay_logits).abs().max()),
        "top1_same": bool(torch.argmax(surgery_logits) == torch.argmax(full_replay_logits)),
        "top5_overlap": _topk_overlap(surgery_logits, full_replay_logits, 5),
        "top10_overlap": _topk_overlap(surgery_logits, full_replay_logits, 10),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[duplex prompt update benchmark]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    assert metrics["cache_len_before_update"] > 0
    assert metrics["cache_surgery_ms"] > 0
    assert metrics["full_replay_ms"] > 0
