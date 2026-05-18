"""Speculative VAD rollback helpers for streaming generation."""

import logging
import time

import torch
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import EncoderDecoderCache

from ..runtime.state import SpeculativeSnapshot

logger = logging.getLogger(__name__)


class SpeculativeSnapshotMixin:
    # ============== 抢跑快照/恢复接口 ==============
    
    def _save_speculative_snapshot(self) -> SpeculativeSnapshot:
        """Internal method: save speculative snapshot.
    
        Called at the start of streaming_generate, saves to self._speculative_snapshot.
    
        Save strategy:
        - LLM KV Cache: only record length (restore by truncation, zero extra VRAM)
        - Audio KV Cache: deep clone (as generate sets it to None)
        - Mel processor: full state snapshot (including buffer)
        """
        # 1. 获取 LLM cache 信息
        llm_cache_length = self._get_kv_cache_length()
        llm_cache_checksum = None
        if self.llm_past_key_values is not None and hasattr(self.llm_past_key_values, "key_cache"):
            if len(self.llm_past_key_values.key_cache) > 0:
                llm_cache_checksum = self.llm_past_key_values.key_cache[0].sum().item()
    
        # 2. 获取 audio cache 长度并克隆 audio_past_key_values
        audio_cache_length = 0
        audio_cache_checksum = None
        audio_past_key_values_clone = None
        if self.audio_past_key_values is not None:
            # 处理 DynamicCache 格式（Whisper encoder 可能返回此格式）
            if isinstance(self.audio_past_key_values, DynamicCache):
                if hasattr(self.audio_past_key_values, "key_cache") and len(self.audio_past_key_values.key_cache) > 0:
                    audio_cache_length = self.audio_past_key_values.key_cache[0].shape[2]
                    audio_cache_checksum = self.audio_past_key_values.key_cache[0].sum().item()
                # 深度克隆 DynamicCache
                cloned_cache = DynamicCache()
                for k, v in zip(self.audio_past_key_values.key_cache, self.audio_past_key_values.value_cache):
                    cloned_cache.update(k.clone(), v.clone(), layer_idx=len(cloned_cache.key_cache))
                audio_past_key_values_clone = cloned_cache
                logger.debug(f"[Speculative] Cloned DynamicCache with length {audio_cache_length}")
            # 处理 EncoderDecoderCache 格式
            elif isinstance(self.audio_past_key_values, EncoderDecoderCache):
                self_attn_cache = self.audio_past_key_values.self_attention_cache
                if hasattr(self_attn_cache, "key_cache") and len(self_attn_cache.key_cache) > 0:
                    audio_cache_length = self_attn_cache.key_cache[0].shape[2]
                    audio_cache_checksum = self_attn_cache.key_cache[0].sum().item()
                # 深度克隆 EncoderDecoderCache
                cloned_self_attn = DynamicCache()
                if hasattr(self_attn_cache, "key_cache"):
                    for k, v in zip(self_attn_cache.key_cache, self_attn_cache.value_cache):
                        cloned_self_attn.update(k.clone(), v.clone(), layer_idx=len(cloned_self_attn.key_cache))
                cross_attn_cache = self.audio_past_key_values.cross_attention_cache
                cloned_cross_attn = DynamicCache()
                if hasattr(cross_attn_cache, "key_cache"):
                    for k, v in zip(cross_attn_cache.key_cache, cross_attn_cache.value_cache):
                        cloned_cross_attn.update(k.clone(), v.clone(), layer_idx=len(cloned_cross_attn.key_cache))
                audio_past_key_values_clone = EncoderDecoderCache(cloned_self_attn, cloned_cross_attn)
                logger.debug(f"[Speculative] Cloned EncoderDecoderCache with length {audio_cache_length}")
            # 处理 tuple 格式（兼容旧格式）
            elif isinstance(self.audio_past_key_values, tuple) and len(self.audio_past_key_values) > 0:
                audio_cache_length = self.audio_past_key_values[0][0].shape[2]
                audio_cache_checksum = self.audio_past_key_values[0][0].sum().item()
                # 深度克隆 audio_past_key_values（tuple of tuples of tensors）
                audio_past_key_values_clone = tuple(
                    tuple(t.clone() for t in layer_cache) for layer_cache in self.audio_past_key_values
                )
    
        # 3. 获取 mel processor 快照
        mel_processor_snapshot = None
        mel_buffer_checksum = None
        if hasattr(self, "processor") and self.processor is not None:
            mel_processor_snapshot = self.processor.get_streaming_snapshot()
            if mel_processor_snapshot:
                buf = mel_processor_snapshot.get("buffer")
                if buf is not None and len(buf) > 0:
                    mel_buffer_checksum = float(buf.sum())
    
        # 4. 保存 RNG 状态（关键：用于恢复后确保 dithering 等随机操作的确定性）
        rng_state_cpu = torch.get_rng_state()
        rng_state_cuda = None
        if torch.cuda.is_available() and self.device.type == "cuda":
            rng_state_cuda = torch.cuda.get_rng_state(self.device)
    
        # 5. 创建快照
        snapshot = SpeculativeSnapshot(
            llm_cache_length=llm_cache_length,
            audio_cache_length=audio_cache_length,
            new_user_msg=self.new_user_msg,
            llm_generated=self.llm_generated,
            llm_generate_completed=self.llm_generate_completed,
            next_round_id=self._next_round_id,
            pending_round_id=self._pending_round_id,
            omni_chunk_history_length=len(self._omni_chunk_history),
            tts_last_turn_tokens=self.tts_last_turn_tokens.clone() if self.tts_last_turn_tokens is not None else None,
            audio_chunk_idx=self.audio_chunk_idx,
            mel_processor_snapshot=mel_processor_snapshot,
            audio_past_key_values=audio_past_key_values_clone,
            timestamp=time.time(),
            # 调试字段
            llm_cache_checksum=llm_cache_checksum,
            audio_cache_checksum=audio_cache_checksum,
            mel_buffer_checksum=mel_buffer_checksum,
            # RNG 状态
            rng_state_cpu=rng_state_cpu,
            rng_state_cuda=rng_state_cuda,
        )
    
        logger.info("[Speculative] Saved snapshot: %s", snapshot.summary())
        logger.debug(
            "[Speculative] Snapshot checksums: llm=%.6f, audio=%.6f, mel_buf=%.6f",
            llm_cache_checksum or 0.0,
            audio_cache_checksum or 0.0,
            mel_buffer_checksum or 0.0,
        )
    
        return snapshot
    
    def restore_speculative_snapshot(self) -> bool:
        """Restore speculative snapshot - called when VAD speculation fails.
    
        Restores model state to before streaming_generate was called,
        allowing continued streaming_prefill for newly arrived audio.
    
        Notes:
        - Snapshot is saved when streaming_generate is called with enable_speculative_snapshot=True
        - This method uses the most recent snapshot for restoration
        - Snapshot is cleared after restore, cannot be called repeatedly
    
        Returns:
            bool: Whether restoration was successful
        """
        snapshot = getattr(self, "_speculative_snapshot", None)
    
        if snapshot is None:
            logger.warning("[Speculative] No snapshot to restore")
            return False
    
        try:
            # 记录恢复前的状态（用于日志对比）
            current_cache_length = self._get_kv_cache_length()
            current_history_length = len(self._omni_chunk_history)
    
            logger.info(
                "[Speculative] Restoring snapshot: target=%s",
                snapshot.summary(),
            )
            logger.info(
                "[Speculative] Current state before restore: llm_cache=%d, history_len=%d, "
                "audio_chunk_idx=%d, new_user_msg=%s, llm_generated=%s",
                current_cache_length,
                current_history_length,
                self.audio_chunk_idx,
                self.new_user_msg,
                self.llm_generated,
            )
    
            # 1. 裁剪 LLM KV Cache
            if current_cache_length > snapshot.llm_cache_length:
                self._truncate_llm_cache(snapshot.llm_cache_length)
                logger.debug(
                    "[Speculative] Truncated LLM cache: %d -> %d",
                    current_cache_length,
                    snapshot.llm_cache_length,
                )
    
            # 2. 恢复 Audio KV Cache（关键：从克隆的副本恢复）
            # 因为 streaming_generate 会将 audio_past_key_values 设为 None
            self.audio_past_key_values = snapshot.audio_past_key_values
            if snapshot.audio_past_key_values is not None:
                logger.debug(
                    "[Speculative] Restored audio cache: length=%d, checksum=%.6f",
                    snapshot.audio_cache_length,
                    snapshot.audio_cache_checksum or 0.0,
                )
            else:
                logger.debug("[Speculative] Audio cache restored to None")
    
            # 3. 恢复会话状态
            self.new_user_msg = snapshot.new_user_msg
            self.llm_generated = snapshot.llm_generated
            self.llm_generate_completed = snapshot.llm_generate_completed
    
            # 4. 恢复 Round 管理
            self._next_round_id = snapshot.next_round_id
            self._pending_round_id = snapshot.pending_round_id
    
            # 5. 截断 chunk 历史
            if current_history_length > snapshot.omni_chunk_history_length:
                self._omni_chunk_history = self._omni_chunk_history[: snapshot.omni_chunk_history_length]
                logger.debug(
                    "[Speculative] Truncated chunk history: %d -> %d",
                    current_history_length,
                    snapshot.omni_chunk_history_length,
                )
    
            # 6. 恢复 TTS 状态
            self.tts_last_turn_tokens = snapshot.tts_last_turn_tokens
    
            # 7. 恢复 streaming 处理器状态
            self.audio_chunk_idx = snapshot.audio_chunk_idx
    
            # 8. 恢复 mel processor 状态（关键！否则后续 prefill 会因帧数不匹配而失败）
            if (
                snapshot.mel_processor_snapshot is not None
                and hasattr(self, "processor")
                and self.processor is not None
            ):
                self.processor.restore_streaming_snapshot(snapshot.mel_processor_snapshot)
                mel_snap = snapshot.mel_processor_snapshot
                logger.info(
                    "[Speculative] Restored mel processor: buffer_len=%d, chunk_count=%d, "
                    "last_emitted_T=%d, total_samples=%d",
                    len(mel_snap.get("buffer", [])),
                    mel_snap.get("chunk_count", 0),
                    mel_snap.get("last_emitted_T", 0),
                    mel_snap.get("total_samples_processed", 0),
                )
    
            # 9. 恢复 RNG 状态（关键：确保 dithering 等随机操作的确定性）
            if snapshot.rng_state_cpu is not None:
                torch.set_rng_state(snapshot.rng_state_cpu)
                logger.debug("[Speculative] Restored CPU RNG state")
            if snapshot.rng_state_cuda is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(snapshot.rng_state_cuda, self.device)
                logger.debug("[Speculative] Restored CUDA RNG state")
    
            # 11. 清理生成过程中产生的临时状态
            if hasattr(self, "_streaming_generated_token_ids"):
                del self._streaming_generated_token_ids
            if hasattr(self, "_last_streaming_text"):
                del self._last_streaming_text
    
            # 12. 验证恢复后的状态
            restored_cache_length = self._get_kv_cache_length()
            if restored_cache_length != snapshot.llm_cache_length:
                logger.warning(
                    "[Speculative] LLM cache length mismatch after restore: expected=%d, actual=%d",
                    snapshot.llm_cache_length,
                    restored_cache_length,
                )
    
            # 验证 LLM cache checksum（如果有）
            if snapshot.llm_cache_checksum is not None and self.llm_past_key_values is not None:
                if hasattr(self.llm_past_key_values, "key_cache") and len(self.llm_past_key_values.key_cache) > 0:
                    current_checksum = self.llm_past_key_values.key_cache[0].sum().item()
                    if abs(current_checksum - snapshot.llm_cache_checksum) > 1e-3:
                        logger.warning(
                            "[Speculative] LLM cache checksum mismatch: expected=%.6f, actual=%.6f",
                            snapshot.llm_cache_checksum,
                            current_checksum,
                        )
                    else:
                        logger.debug(
                            "[Speculative] LLM cache checksum verified: %.6f",
                            current_checksum,
                        )
    
            # 11. 清除快照（只能恢复一次）
            self._speculative_snapshot = None
    
            logger.info(
                "[Speculative] Restore completed: llm_cache %d -> %d, elapsed=%.3fs",
                current_cache_length,
                snapshot.llm_cache_length,
                time.time() - snapshot.timestamp,
            )
    
            return True
    
        except Exception as e:
            import traceback
    
            logger.error("[Speculative] Failed to restore snapshot: %s", e)
            logger.error("[Speculative] Traceback: %s", traceback.format_exc())
            return False
    
    def has_speculative_snapshot(self) -> bool:
        return getattr(self, "_speculative_snapshot", None) is not None
    
    def clear_speculative_snapshot(self) -> None:
        if hasattr(self, "_speculative_snapshot"):
            self._speculative_snapshot = None
    
    def _truncate_llm_cache(self, target_length: int) -> None:
        if self.llm_past_key_values is None:
            return
    
        cache = self._ensure_dynamic_cache()
        if cache is None:
            return
    
        current_length = self._get_kv_cache_length(cache)
        if current_length <= target_length:
            return
    
        # 裁剪每一层的 cache
        for layer_idx in range(len(cache.key_cache)):
            if cache.key_cache[layer_idx].numel() > 0:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :target_length, :].contiguous()
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :target_length, :].contiguous()
    
        # 更新 cache 元数据
        cache.crop(target_length)
        cache._seen_tokens = target_length
    
        logger.debug("[Speculative] Truncated LLM cache: %d -> %d", current_length, target_length)
    
    # ============== 抢跑快照/恢复接口 结束 ==============

