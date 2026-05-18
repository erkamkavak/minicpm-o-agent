"""Streaming session and cache-window helpers for the MiniCPMO wrapper."""

import logging
from typing import List
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from ..runtime.cache import as_dynamic_cache
from ..runtime.cache import drop_tokens_from_cache
from ..runtime.cache import get_kv_cache_length
from ..runtime.cache import realign_rotary_suffix

logger = logging.getLogger(__name__)


class StreamingSessionMixin:
    # for sliding window
    def _ensure_dynamic_cache(self):
        cache = self.llm_past_key_values
        if cache is None:
            return None

        cache = as_dynamic_cache(cache)
        if isinstance(cache, DynamicCache):
            self.llm_past_key_values = cache
            return cache

        return None

    def _get_kv_cache_length(self, cache=None):
        cache = cache if cache is not None else self.llm_past_key_values
        return get_kv_cache_length(cache)

    # todo: not-used del?
    def _rebuild_cache_from_history(self):
        preserved_ids: List[torch.Tensor] = []
        for entry in self._omni_chunk_history:
            ids = entry.get("input_ids")
            if ids is None or not isinstance(ids, torch.Tensor) or ids.numel() == 0:
                continue
            preserved_ids.append(ids.to(self.device))
        if not preserved_ids:
            self.llm_past_key_values = None
            self.streaming_position_offset = 0
            self._rope_inv_freq_cache.clear()
            return

        concat_ids = torch.cat(preserved_ids, dim=1)
        attention_mask = torch.ones((1, concat_ids.shape[1]), dtype=torch.bool, device=self.device)
        outputs = self.llm(
            input_ids=concat_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        self.llm_past_key_values = outputs.past_key_values
        self.streaming_position_offset = 0
        self._rope_inv_freq_cache.clear()

    def _get_rope_theta(self) -> float:
        return float(getattr(self.llm.config, "rope_theta", 10000.0))

    def _realign_rotary_suffix(
        self,
        suffix_keys: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapper for realign_rotary_suffix using instance's rope_theta and cache."""
        return realign_rotary_suffix(
            suffix_keys,
            old_positions,
            new_positions,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )

    def _encode_text(self, tokenizer, text) -> Optional[torch.Tensor]:
        if tokenizer is None or not text:
            return None
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.to(self.device)

    @staticmethod
    def _safe_decode(tokenizer, input_ids):
        if tokenizer is None or input_ids is None:
            return None
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.cpu().tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            ids = input_ids
        try:
            return tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            return None

    def _finalize_round(
        self, round_id: Optional[int], cache_before: int, assistant_input_ids: Optional[torch.Tensor] = None
    ):
        if round_id is None:
            self._pending_round_id = None
            return
        cache_after = self._get_kv_cache_length()
        if assistant_input_ids is not None:
            assistant_len = assistant_input_ids.shape[1]
        else:
            assistant_len = max(cache_after - cache_before, 0)
        if assistant_len > 0:
            self._register_chunk(
                assistant_len,
                "assistant",
                round_id=round_id,
                input_ids=assistant_input_ids,
                tokenizer=self.processor.tokenizer if hasattr(self, "processor") else None,
            )
        logger.info(
            "Finalized round=%s cache len before=%s after=%s assistant_len=%s",
            round_id,
            cache_before,
            cache_after,
            assistant_len,
        )
        self._pending_round_id = None
        self._next_round_id += 1

    def _register_chunk(
        self,
        seq_len: int,
        chunk_type: str,
        *,
        round_id: int,
        input_ids=None,
        tokenizer=None,
    ) -> None:
        if seq_len <= 0:
            return
        entry = {"length": int(seq_len), "type": chunk_type, "round": round_id}
        if input_ids is not None:
            entry["input_ids"] = input_ids.clone().detach()
            entry["decoded"] = self._safe_decode(tokenizer, entry["input_ids"])
        else:
            entry["input_ids"] = None
            entry["decoded"] = None
        self._omni_chunk_history.append(entry)
        logger.info(
            "Registered chunk round=%s type=%s len=%s decoded=%s",
            round_id,
            chunk_type,
            entry["length"],
            entry["decoded"],
        )
        if chunk_type == "system":
            self.streaming_text_preserve = max(self.streaming_text_preserve, entry["length"])

    def _drop_tokens_from_cache(self, length: int, cache: DynamicCache) -> bool:
        """Drop tokens from cache using the utility function."""
        _, new_offset, success = drop_tokens_from_cache(
            cache=cache,
            length=length,
            preserve=self.streaming_text_preserve,
            position_offset=self.streaming_position_offset,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )
        if success:
            self.streaming_position_offset = new_offset
        return success

    def _drop_next_round(self, cache: DynamicCache) -> bool:
        seen_rounds = set()
        for entry in self._omni_chunk_history:
            round_id = entry.get("round")
            if round_id is None or round_id in seen_rounds:
                continue
            seen_rounds.add(round_id)
            round_entries = [e for e in self._omni_chunk_history if e.get("round") == round_id]
            if any(e.get("type") == "system" for e in round_entries):
                continue
            if self._drop_round(round_id, cache):
                return True
        return False

    def _drop_round(self, round_id: int, cache: DynamicCache) -> bool:
        entries = [e for e in self._omni_chunk_history if e.get("round") == round_id]
        if not entries:
            return False
        total_len = sum(e["length"] for e in entries)
        if total_len <= 0:
            for e in entries:
                self._omni_chunk_history.remove(e)
            return False
        if not self._drop_tokens_from_cache(total_len, cache):
            return False
        for e in entries:
            logger.info(
                "Dropped round=%s chunk type=%s len=%s decoded=%s",
                round_id,
                e["type"],
                e["length"],
                e.get("decoded"),
            )
            self._omni_chunk_history.remove(e)
        return True

    def _enforce_text_window(self) -> None:
        if not self.streaming_window_enabled:
            return
        cache = self._ensure_dynamic_cache()
        if cache is None:
            return
        high_limit = max(0, int(self.streaming_window_config.text_window_high_tokens))
        low_limit = max(0, int(self.streaming_window_config.text_window_low_tokens))
        if high_limit <= 0:
            return
        target = max(0, low_limit)
        total_len = self._get_kv_cache_length(cache)
        if total_len <= high_limit:
            return
        dropped_any = False
        while total_len > target:
            if not self._drop_next_round(cache):
                break
            dropped_any = True
            total_len = self._get_kv_cache_length(cache)

    # for sliding window

    # Speculative snapshot helpers are provided by SpeculativeSnapshotMixin.

