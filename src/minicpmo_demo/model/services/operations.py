"""Compile, warmup, and benchmark helpers for the unified model wrapper."""

import logging
import os
import threading
from typing import List
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class UnifiedOperationsMixin:
    def apply_torch_compile(
        self,
        mode: str = "default",
        dynamic: bool = True,
        skip_modules: Optional[List[str]] = None,
    ) -> "MiniCPMO":
        """Apply torch.compile to compute-intensive sub-modules.
    
        Must be called after init_unified().  DuplexCapability and other
        components access the same model instance by reference, so they
        automatically use the compiled versions after this call.
    
        Compile targets (compute-intensive sub-modules):
          - vpm: SiglipVisionTransformer (vision encoder)
          - llm.model: Qwen3Model backbone (LLM core; outer lm_head + generate
            logic is kept un-compiled)
          - resampler: vision resampler
          - tts.model: LlamaModel backbone (TTS core)
    
        NOT compiled:
          - apm (Whisper audio encoder): streaming-specific behavior + dynamic
            shapes, low compile benefit
          - tts.audio_tokenizer (Token2wav / CosyVoice2): external library, not
            a standard nn.Module
          - MiniCPMO outer wrapper: heavy Python control flow, low compile benefit
    
        Note: torch.compile only wraps the modules; actual Triton compilation is
        triggered on the first forward pass.  Call warmup_compile() afterwards to
        trigger compilation proactively.
    
        Args:
            mode: torch.compile mode.
                - "default": balanced compile time vs runtime speed (recommended)
                - "reduce-overhead": uses CUDA Graphs (static shapes only)
                - "max-autotune": maximum optimization (very long compile time)
            dynamic: Enable dynamic shape support (recommended True to avoid
                recompilation when shapes change).
            skip_modules: Module names to skip (e.g. ["llm.model"] for AWQ
                quantized LLM whose custom kernels are incompatible with compile).
    
        Returns:
            self (for method chaining).
        """
        import time as _time
        skip = set(skip_modules or [])
        logger.info(
            f"[torch.compile] Compiling sub-modules "
            f"(mode={mode}, dynamic={dynamic}, skip={skip or 'none'})"
        )
        t0 = _time.time()
    
        compile_kwargs = dict(mode=mode, dynamic=dynamic)
        compiled_modules: list = []
        skipped_modules: list = []
    
        # if hasattr(self, "vpm") and "vpm" not in skip:
        #     self.vpm = torch.compile(self.vpm, **compile_kwargs)
        #     compiled_modules.append("vpm")
        # elif "vpm" in skip:
        #     skipped_modules.append("vpm")
    
        if hasattr(self, "llm") and "llm.model" not in skip:
            self.llm.model = torch.compile(self.llm.model, **compile_kwargs)
            compiled_modules.append("llm.model")
        elif "llm.model" in skip:
            skipped_modules.append("llm.model")
    
        # if hasattr(self, "resampler") and "resampler" not in skip:
        #     self.resampler = torch.compile(self.resampler, **compile_kwargs)
        #     compiled_modules.append("resampler")
        # elif "resampler" in skip:
        #     skipped_modules.append("resampler")
    
        if hasattr(self, "tts") and hasattr(self.tts, "model") and "tts.model" not in skip:
            self.tts.model = torch.compile(self.tts.model, **compile_kwargs)
            compiled_modules.append("tts.model")
        elif "tts.model" in skip:
            skipped_modules.append("tts.model")
    
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")
    
        elapsed = _time.time() - t0
        self._compiled = True
        self._compile_active = True
        logger.info(
            f"[torch.compile] Wrapping done ({elapsed:.2f}s), "
            f"compiled: {compiled_modules}"
            + (f", skipped: {skipped_modules}" if skipped_modules else "")
            + ". Actual compilation triggers on first forward."
        )
        return self
    
    def set_compile_enabled(self, enabled: bool) -> None:
        """Switch between compiled and eager execution for all compiled sub-modules.
    
        Only effective after apply_torch_compile() has been called.
        Compiled and eager modules share the same weights (zero copy),
        so switching is instant and costs no extra memory.
        """
        if not getattr(self, "_compiled", False):
            return
        if enabled == getattr(self, "_compile_active", True):
            return
    
        swapped: list = []
    
        if hasattr(self, "llm"):
            cur = self.llm.model
            if enabled:
                compiled = getattr(cur, "_compiled_ref", None)
                if compiled is not None:
                    self.llm.model = compiled
                    swapped.append("llm.model")
            else:
                orig = getattr(cur, "_orig_mod", None)
                if orig is not None:
                    orig._compiled_ref = cur
                    self.llm.model = orig
                    swapped.append("llm.model")
    
        if hasattr(self, "tts") and hasattr(self.tts, "model"):
            cur = self.tts.model
            if enabled:
                compiled = getattr(cur, "_compiled_ref", None)
                if compiled is not None:
                    self.tts.model = compiled
                    swapped.append("tts.model")
            else:
                orig = getattr(cur, "_orig_mod", None)
                if orig is not None:
                    orig._compiled_ref = cur
                    self.tts.model = orig
                    swapped.append("tts.model")
    
        self._compile_active = enabled
        logger.info(f"[torch.compile] {'enabled' if enabled else 'disabled'} → swapped {swapped}")
    
    def warmup_compile(
        self,
        warmup_video_path: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        max_warmup_chunks: int = 10,
        total_estimate_seconds: int = 400,
    ) -> None:
        """Trigger Triton kernel compilation via a real omni full-duplex session.
    
        Runs a complete duplex inference loop (prepare → prefill → generate →
        finalize) using an actual MP4 video, exercising the four compiled
        sub-modules (vpm / resampler / llm.model / tts.model) in their real
        execution context.  apm and token2wav are NOT compile targets; they
        simply run as part of the duplex pipeline.
    
        Per-unit timing breakdown is logged for each chunk.
    
        Args:
            warmup_video_path: MP4 video for warmup.  Defaults to
                ``assets/samples/compile.mp4``.
            ref_audio_path: Reference audio for TTS voice cloning.  Defaults to
                ``assets/ref_audio/ref_minicpm_signature.wav``.
            max_warmup_chunks: Maximum number of 1-second chunks to process.
        """
        if not getattr(self, "_compiled", False):
            logger.warning("[warmup] model not compiled, skipping warmup")
            return
    
        if self.duplex is None:
            logger.warning("[warmup] duplex not initialized, skipping warmup")
            return
    
        import sys
        import time as _time
        import threading
    
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if warmup_video_path is None:
            warmup_video_path = os.path.join(
                project_root, "assets", "samples", "compile.mp4"
            )
        if ref_audio_path is None:
            ref_audio_path = os.path.join(
                project_root, "assets", "ref_audio", "ref_minicpm_signature.wav"
            )
    
        if not os.path.isfile(warmup_video_path):
            logger.warning("[warmup] warmup video not found: %s, skipping", warmup_video_path)
            return
        if not os.path.isfile(ref_audio_path):
            logger.warning("[warmup] ref audio not found: %s, skipping", ref_audio_path)
            return
    
        # ── Persistent spinner ──
        _SPINNER = r"\|/-"
        _TOTAL_EST_S = total_estimate_seconds
        _G = "\033[32m"
        _B = "\033[1m"
        _R = "\033[0m"
    
        t_total = _time.time()
        _spin_idx = [0]
        _stage_info = ["(1/6) Initializing"]
        _lock = threading.Lock()
        _stop_evt = threading.Event()
        _out = sys.stderr
    
        def _render_spinner():
            elapsed = _time.time() - t_total
            remaining = max(0, _TOTAL_EST_S - elapsed)
            s = _SPINNER[_spin_idx[0] % 4]
            _spin_idx[0] += 1
            info = _stage_info[0]
            pct = min(100, int(elapsed / _TOTAL_EST_S * 100))
            bar_w = 20
            filled = int(bar_w * pct / 100)
            bar = "█" * filled + "░" * (bar_w - filled)
            return (
                f"\r{_G}{_B}[{s}] {bar} {pct:3d}% | {info} "
                f"| elapsed={elapsed:.0f}s remaining~{remaining:.0f}s{_R}\033[K"
            )
    
        def _spinner_loop():
            while not _stop_evt.wait(0.15):
                with _lock:
                    _out.write(_render_spinner())
                    _out.flush()
    
        def _set_stage(stage_idx: int, total: int, name: str, detail: str = ""):
            tag = f" {detail}" if detail else ""
            _stage_info[0] = f"({stage_idx}/{total}) {name}{tag}"
    
        def _log(msg: str):
            with _lock:
                _out.write(f"\r\033[K")
                _out.flush()
                logger.info(f"{_G}%s{_R}", msg)
                _out.write(_render_spinner())
                _out.flush()
    
        spinner_thread = threading.Thread(target=_spinner_loop, daemon=True)
        spinner_thread.start()
    
        _log(f"[warmup] starting omni duplex warmup (video={warmup_video_path}, "
             f"max_chunks={max_warmup_chunks}, est~{_TOTAL_EST_S}s)")
    
        # ── 1. Extract audio chunks and video frames from MP4 ──
        _set_stage(1, 6, "Extracting MP4")
        audio_chunks, frames = self._extract_mp4_chunks(
            warmup_video_path, max_chunks=max_warmup_chunks
        )
        _log(f"[warmup] (1/6) MP4 extracted: {len(audio_chunks)} chunks, {len(frames)} frames")
        if not audio_chunks:
            _stop_evt.set()
            with _lock:
                _out.write("\r\033[K")
                _out.flush()
            logger.warning("[warmup] no audio extracted from MP4, skipping")
            return
    
        # ── 2. Load reference audio ──
        _set_stage(2, 6, "Loading reference audio")
        import librosa as _librosa
        ref_audio, _ = _librosa.load(ref_audio_path, sr=16000, mono=True)
        _log("[warmup] (2/6) Reference audio loaded")
    
        # ── 3. Start duplex session ──
        _set_stage(3, 6, "Duplex prepare")
        self.duplex.prepare(
            prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
            suffix_system_prompt="<|audio_end|><|im_end|>",
            ref_audio=ref_audio,
            prompt_wav_path=ref_audio_path,
        )
        _log("[warmup] (3/6) Duplex prepare done")
    
        # ── 4. Per-chunk warmup (Triton compilation) ──
        tts_triggered = False
        num_chunks = min(len(audio_chunks), len(frames)) if frames else len(audio_chunks)
    
        for i in range(num_chunks):
            _set_stage(4, 6, "Triton compilation", f"unit {i}/{num_chunks}")
            frame_list = [frames[i]] if frames and i < len(frames) else None
    
            # ── Prefill ──
            t_pf = _time.time()
            try:
                prefill_result = self.duplex.streaming_prefill(
                    audio_waveform=audio_chunks[i],
                    frame_list=frame_list,
                    max_slice_nums=1,
                )
            except Exception as e:
                _log(f"[warmup] unit {i} prefill failed: {e}")
                break
            cost_prefill = _time.time() - t_pf
    
            pf_vp = prefill_result.get("cost_vision_process", 0) * 1000
            pf_ve = prefill_result.get("cost_vision_embed", 0) * 1000
            pf_vf = prefill_result.get("cost_vision_feed", 0) * 1000
            pf_ap = prefill_result.get("cost_audio_process", 0) * 1000
            pf_ae = prefill_result.get("cost_audio_embed", 0) * 1000
            pf_af = prefill_result.get("cost_audio_feed", 0) * 1000
            pf_all = cost_prefill * 1000
    
            # ── Generate ──
            t_gen = _time.time()
            try:
                gen_result = self.duplex.streaming_generate()
            except Exception as e:
                _log(f"[warmup] unit {i} generate failed: {e}")
                break
            cost_generate = _time.time() - t_gen
    
            is_listen = gen_result.get("is_listen", True)
            gen_llm = gen_result.get("cost_llm", 0) * 1000 if gen_result.get("cost_llm") else 0
            gen_tts_p = gen_result.get("cost_tts_prep", 0) * 1000 if gen_result.get("cost_tts_prep") else 0
            gen_tts = gen_result.get("cost_tts", 0) * 1000 if gen_result.get("cost_tts") else 0
            gen_t2w = gen_result.get("cost_token2wav", 0) * 1000 if gen_result.get("cost_token2wav") else 0
            gen_all = cost_generate * 1000
            decision = "LISTEN" if is_listen else "SPEAK"
    
            if not is_listen:
                tts_triggered = True
                text = gen_result.get("text", "")
                if text:
                    decision += f' "{text[:20]}"'
    
            elapsed = _time.time() - t_total
            remaining = max(0, _TOTAL_EST_S - elapsed)
            _log(
                f"[warmup] unit={i}/{num_chunks} | prefill: vis_proc={pf_vp:.0f}ms vis_emb={pf_ve:.0f}ms "
                f"vis_feed={pf_vf:.0f}ms aud_proc={pf_ap:.0f}ms aud_emb={pf_ae:.0f}ms "
                f"aud_feed={pf_af:.0f}ms total={pf_all:.0f}ms | generate: llm={gen_llm:.0f}ms "
                f"tts_prep={gen_tts_p:.0f}ms tts={gen_tts:.0f}ms token2wav={gen_t2w:.0f}ms "
                f"total={gen_all:.0f}ms | decision={decision} | elapsed={elapsed:.0f}s remaining~{remaining:.0f}s"
            )
    
            # ── Finalize ──
            try:
                self.duplex.finalize_unit()
            except Exception as e:
                _log(f"[warmup] unit {i} finalize failed: {e}")
                break
    
            if gen_result.get("end_of_turn", False):
                _log("[warmup] model emitted end_of_turn, stopping early")
                break
    
        # ── 5. TTS fallback if model stayed in LISTEN throughout ──
        if not tts_triggered and hasattr(self, "tts") and hasattr(self.tts, "model"):
            _set_stage(5, 6, "TTS fallback warmup")
            _log("[warmup] (5/6) TTS was not triggered during duplex, running fallback...")
            self._warmup_tts_fallback()
            _log("[warmup] (5/6) TTS fallback done")
        else:
            _log("[warmup] (5/6) TTS fallback skipped (already triggered)")
    
        # ── 6. Clean up duplex session state ──
        _set_stage(6, 6, "Cleanup")
        self.duplex._reset_streaming_state()
        self.duplex.decoder.reset()
        if hasattr(self.tts, "audio_tokenizer"):
            tokenizer = self.tts.audio_tokenizer
            for attr in ("stream_cache", "hift_cache_dict", "cache"):
                if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
                    setattr(tokenizer, attr, None)
        self.reset_session(reset_token2wav_cache=True)
        torch.cuda.empty_cache()
    
        # ── Stop spinner and print final line ──
        _stop_evt.set()
        spinner_thread.join(timeout=1)
        with _lock:
            _out.write("\r\033[K")
            _out.flush()
    
        total = _time.time() - t_total
        logger.info(
            "%s[warmup] ✓ omni duplex warmup complete (total=%.1fs, tts_triggered=%s)%s",
            _G, total, tts_triggered, _R,
        )
    
    def benchmark(
        self,
        video_paths: Optional[List[str]] = None,
        video_dir: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        system_prompt: str = "Streaming Omni Conversation.",
        max_chunks_per_video: int = 0,
    ) -> dict:
        """Run omni duplex benchmark, collecting per-module timing for LISTEN / SPEAK.
    
        Processes one or more MP4 videos through the full duplex pipeline
        (prepare → prefill → generate → finalize) and logs per-unit timing
        breakdown.  Summary statistics are printed at the end, grouped by
        decision type (LISTEN vs SPEAK).
    
        Args:
            video_paths: List of MP4 video file paths.
            video_dir: Directory containing MP4 videos (scanned for ``*.mp4``).
                Can be used together with *video_paths*; all paths are merged.
            ref_audio_path: Reference audio for TTS voice cloning.
            system_prompt: Content of the system prompt (wrapped in the
                standard ``<|im_start|>`` framing automatically).
            max_chunks_per_video: Maximum number of 1-second chunks to process
                per video.  ``0`` means process the entire video.
    
        Returns:
            Dict with ``units`` (per-unit records), counts, and elapsed time.
        """
        import time as _time
    
        if self.duplex is None:
            logger.warning("[bench] duplex not initialized, cannot benchmark")
            return {}
    
        # ── Resolve video paths ──
        resolved_videos: List[str] = []
        if video_dir and os.path.isdir(video_dir):
            for f in sorted(os.listdir(video_dir)):
                if f.lower().endswith(".mp4"):
                    resolved_videos.append(os.path.join(video_dir, f))
        if video_paths:
            for p in video_paths:
                if os.path.isfile(p):
                    resolved_videos.append(p)
                else:
                    logger.warning("[bench] video not found, skipping: %s", p)
        if not resolved_videos:
            logger.warning("[bench] no valid video files found, aborting")
            return {}
    
        # ── Resolve ref audio ──
        if ref_audio_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            ref_audio_path = os.path.join(
                project_root, "assets", "ref_audio", "ref_minicpm_signature.wav"
            )
        if not os.path.isfile(ref_audio_path):
            logger.warning("[bench] ref audio not found: %s", ref_audio_path)
            return {}
    
        import librosa as _librosa
        ref_audio, _ = _librosa.load(ref_audio_path, sr=16000, mono=True)
    
        prefix_prompt = f"<|im_start|>system\n{system_prompt}\n<|audio_start|>"
        suffix_prompt = "<|audio_end|><|im_end|>"
    
        all_units: List[dict] = []
        t_total = _time.time()
    
        logger.info(
            "[bench] Starting benchmark: %d video(s), system_prompt=%r",
            len(resolved_videos), system_prompt,
        )
    
        for vid_idx, video_path in enumerate(resolved_videos):
            video_name = os.path.basename(video_path)
            max_c = max_chunks_per_video if max_chunks_per_video > 0 else 99999
    
            logger.info(
                "[bench] ── Video %d/%d: %s (max_chunks=%s) ──",
                vid_idx + 1, len(resolved_videos), video_name,
                max_chunks_per_video if max_chunks_per_video > 0 else "all",
            )
    
            # ── Extract audio chunks and video frames ──
            audio_chunks, frames = self._extract_mp4_chunks(
                video_path, max_chunks=max_c
            )
            if not audio_chunks:
                logger.warning("[bench] no audio extracted from %s, skipping", video_name)
                continue
    
            num_chunks = (
                min(len(audio_chunks), len(frames)) if frames else len(audio_chunks)
            )
            logger.info("[bench] Extracted %d chunks, %d frames", len(audio_chunks), len(frames))
    
            # ── Prepare duplex session ──
            self.duplex.prepare(
                prefix_system_prompt=prefix_prompt,
                suffix_system_prompt=suffix_prompt,
                ref_audio=ref_audio,
                prompt_wav_path=ref_audio_path,
            )
    
            # ── Per-chunk loop ──
            for i in range(num_chunks):
                frame_list = [frames[i]] if frames and i < len(frames) else None
    
                # Prefill
                t_pf = _time.time()
                try:
                    prefill_result = self.duplex.streaming_prefill(
                        audio_waveform=audio_chunks[i],
                        frame_list=frame_list,
                        max_slice_nums=1,
                    )
                except Exception as e:
                    logger.error("[bench] video=%s unit=%d prefill failed: %s", video_name, i, e)
                    break
                cost_prefill = (_time.time() - t_pf) * 1000
    
                pf = {
                    "vision_process": prefill_result.get("cost_vision_process", 0) * 1000,
                    "vision_embed": prefill_result.get("cost_vision_embed", 0) * 1000,
                    "vision_feed": prefill_result.get("cost_vision_feed", 0) * 1000,
                    "audio_process": prefill_result.get("cost_audio_process", 0) * 1000,
                    "audio_embed": prefill_result.get("cost_audio_embed", 0) * 1000,
                    "audio_feed": prefill_result.get("cost_audio_feed", 0) * 1000,
                    "total": cost_prefill,
                }
    
                # Generate
                t_gen = _time.time()
                try:
                    gen_result = self.duplex.streaming_generate()
                except Exception as e:
                    logger.error("[bench] video=%s unit=%d generate failed: %s", video_name, i, e)
                    break
                cost_generate = (_time.time() - t_gen) * 1000
    
                is_listen = gen_result.get("is_listen", True)
                gn = {
                    "llm": (gen_result.get("cost_llm") or 0) * 1000,
                    "tts_prep": (gen_result.get("cost_tts_prep") or 0) * 1000,
                    "tts": (gen_result.get("cost_tts") or 0) * 1000,
                    "token2wav": (gen_result.get("cost_token2wav") or 0) * 1000,
                    "total": cost_generate,
                }
    
                decision = "LISTEN" if is_listen else "SPEAK"
                unit_total = pf["total"] + gn["total"]
                text_snippet = ""
                if not is_listen:
                    text_snippet = gen_result.get("text", "")[:40]
    
                unit_record = {
                    "video": video_name,
                    "unit_idx": i,
                    "num_chunks": num_chunks,
                    "decision": decision,
                    "prefill": pf,
                    "generate": gn,
                    "unit_total": unit_total,
                }
                if text_snippet:
                    unit_record["text"] = text_snippet
    
                # Per-unit log
                decision_str = decision
                if text_snippet:
                    decision_str += f' "{text_snippet}"'
    
                logger.info(
                    "[bench] video=%s unit=%d/%d | %s | "
                    "prefill: vis_proc=%.0fms vis_emb=%.0fms vis_feed=%.0fms "
                    "aud_proc=%.0fms aud_emb=%.0fms aud_feed=%.0fms total=%.0fms | "
                    "generate: llm=%.0fms tts_prep=%.0fms tts=%.0fms token2wav=%.0fms "
                    "total=%.0fms | unit_total=%.0fms",
                    video_name, i, num_chunks, decision_str,
                    pf["vision_process"], pf["vision_embed"], pf["vision_feed"],
                    pf["audio_process"], pf["audio_embed"], pf["audio_feed"], pf["total"],
                    gn["llm"], gn["tts_prep"], gn["tts"], gn["token2wav"], gn["total"],
                    unit_total,
                )
    
                all_units.append(unit_record)
    
                # Finalize
                try:
                    self.duplex.finalize_unit()
                except Exception as e:
                    logger.error("[bench] video=%s unit=%d finalize failed: %s", video_name, i, e)
                    break
    
                if gen_result.get("end_of_turn", False):
                    logger.info("[bench] end_of_turn at unit %d, stopping video", i)
                    break
    
            # ── Cleanup after each video ──
            self.duplex._reset_streaming_state()
            self.duplex.decoder.reset()
            if hasattr(self.tts, "audio_tokenizer"):
                tokenizer = self.tts.audio_tokenizer
                for attr in ("stream_cache", "hift_cache_dict", "cache"):
                    if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
                        setattr(tokenizer, attr, None)
            self.reset_session(reset_token2wav_cache=True)
            torch.cuda.empty_cache()
    
        total_elapsed = _time.time() - t_total
    
        # ── Summary ──
        listen_units = [u for u in all_units if u["decision"] == "LISTEN"]
        speak_units = [u for u in all_units if u["decision"] == "SPEAK"]
    
        def _agg(records: list, key_path: str) -> dict:
            vals = []
            for r in records:
                v = r
                for k in key_path.split("."):
                    v = v[k]
                vals.append(v)
            if not vals:
                return {"avg": 0.0, "min": 0.0, "max": 0.0}
            return {
                "avg": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
    
        def _print_group(label: str, records: list):
            if not records:
                logger.info("[bench] %s: (no units)", label)
                return
            n = len(records)
            logger.info("[bench] %s (n=%d):", label, n)
    
            pf_t = _agg(records, "prefill.total")
            logger.info(
                "[bench]   prefill:        avg=%.0fms  min=%.0fms  max=%.0fms",
                pf_t["avg"], pf_t["min"], pf_t["max"],
            )
            for key in ("vision_process", "vision_embed", "vision_feed",
                        "audio_process", "audio_embed", "audio_feed"):
                s = _agg(records, f"prefill.{key}")
                logger.info(
                    "[bench]     %-14s avg=%.0fms  min=%.0fms  max=%.0fms",
                    key + ":", s["avg"], s["min"], s["max"],
                )
    
            gn_t = _agg(records, "generate.total")
            logger.info(
                "[bench]   generate:       avg=%.0fms  min=%.0fms  max=%.0fms",
                gn_t["avg"], gn_t["min"], gn_t["max"],
            )
            for key in ("llm", "tts_prep", "tts", "token2wav"):
                s = _agg(records, f"generate.{key}")
                if s["avg"] > 0 or key == "llm":
                    logger.info(
                        "[bench]     %-14s avg=%.0fms  min=%.0fms  max=%.0fms",
                        key + ":", s["avg"], s["min"], s["max"],
                    )
    
            ut = _agg(records, "unit_total")
            logger.info(
                "[bench]   unit_total:     avg=%.0fms  min=%.0fms  max=%.0fms",
                ut["avg"], ut["min"], ut["max"],
            )
    
        logger.info("[bench] " + "=" * 60)
        logger.info("[bench] Benchmark Summary")
        logger.info("[bench] " + "=" * 60)
        logger.info(
            "[bench] Total: %d units (%d LISTEN, %d SPEAK) over %d video(s), elapsed=%.1fs",
            len(all_units), len(listen_units), len(speak_units),
            len(resolved_videos), total_elapsed,
        )
        logger.info("[bench]")
        _print_group("LISTEN", listen_units)
        logger.info("[bench]")
        _print_group("SPEAK", speak_units)
        logger.info("[bench] " + "=" * 60)
    
        def _build_group_stats(records: list) -> dict:
            if not records:
                return {}
            prefill_keys = ("vision_process", "vision_embed", "vision_feed",
                            "audio_process", "audio_embed", "audio_feed", "total")
            generate_keys = ("llm", "tts_prep", "tts", "token2wav", "total")
            return {
                "count": len(records),
                "prefill": {k: _agg(records, f"prefill.{k}") for k in prefill_keys},
                "generate": {k: _agg(records, f"generate.{k}") for k in generate_keys},
                "unit_total": _agg(records, "unit_total"),
            }
    
        return {
            "total_time": total_elapsed,
            "num_videos": len(resolved_videos),
            "num_units": len(all_units),
            "listen_count": len(listen_units),
            "speak_count": len(speak_units),
            "listen_stats": _build_group_stats(listen_units),
            "speak_stats": _build_group_stats(speak_units),
            "units": all_units,
        }
    
    def _extract_mp4_chunks(
        self,
        video_path: str,
        max_chunks: int = 10,
        sample_rate: int = 16000,
    ) -> tuple:
        """Extract 1-second audio chunks and corresponding video frames from MP4.
    
        Uses ffmpeg for both audio extraction and frame extraction (no cv2).
    
        Returns:
            (audio_chunks, frames): audio_chunks is list[np.ndarray] (16kHz mono),
            frames is list[PIL.Image].
        """
        import subprocess
        import tempfile
        from PIL import Image
    
        audio_chunks: list = []
        frames: list = []
        tmp_dir = tempfile.mkdtemp(prefix="warmup_")
    
        try:
            # ── Extract audio ──
            tmp_wav_path = os.path.join(tmp_dir, "audio.wav")
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-ar", str(sample_rate), "-ac", "1",
                    "-t", str(max_chunks),
                    "-f", "wav", "-y", tmp_wav_path,
                ],
                capture_output=True,
                check=True,
            )
            import librosa as _librosa
            audio, _ = _librosa.load(tmp_wav_path, sr=sample_rate, mono=True)
    
            chunk_size = sample_rate
            n_audio = min(max_chunks, len(audio) // chunk_size)
            for i in range(n_audio):
                audio_chunks.append(audio[i * chunk_size : (i + 1) * chunk_size])
    
            # ── Extract video frames (ffmpeg 1fps → JPEG) ──
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-t", str(max_chunks),
                    "-vf", "fps=1",
                    os.path.join(frames_dir, "frame_%04d.jpg"),
                ],
                capture_output=True,
                check=True,
            )
            frame_files = sorted(
                f for f in os.listdir(frames_dir) if f.endswith(".jpg")
            )
            for fname in frame_files[:n_audio]:
                frames.append(
                    Image.open(os.path.join(frames_dir, fname)).convert("RGB")
                )
    
        except Exception as e:
            logger.warning("[warmup] MP4 extraction failed: %s", e)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
        return audio_chunks, frames
    
    def _warmup_tts_fallback(self) -> None:
        """TTS sub-module fallback warmup (called when duplex stayed in LISTEN)."""
        import time as _time
        device = next(self.tts.model.parameters()).device
        tts_hidden = self.tts.config.hidden_size
        tts_dtype = self.tts.model.embed_tokens.weight.dtype
    
        with torch.no_grad():
            t0 = _time.time()
            prefill_len = 32
            dummy = torch.randn(1, prefill_len, tts_hidden, device=device, dtype=tts_dtype)
            pos = torch.arange(prefill_len, dtype=torch.long, device=device).unsqueeze(0)
            out = self.tts.model(inputs_embeds=dummy, position_ids=pos, use_cache=True)
            torch.cuda.synchronize(device)
            logger.info("[warmup]   tts prefill fallback done (%.1fs)", _time.time() - t0)
    
            t1 = _time.time()
            dec = torch.randn(1, 1, tts_hidden, device=device, dtype=tts_dtype)
            dec_pos = torch.tensor([[prefill_len]], dtype=torch.long, device=device)
            _ = self.tts.model(
                inputs_embeds=dec, position_ids=dec_pos,
                past_key_values=out.past_key_values, use_cache=True,
            )
            torch.cuda.synchronize(device)
            del out
            logger.info("[warmup]   tts decode fallback done (%.1fs)", _time.time() - t1)
