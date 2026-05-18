"""Streaming and non-streaming conversation APIs for the MiniCPMO wrapper."""

import json
import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..components import gen_logits
from ..processing_minicpmo import MiniCPMOProcessor
from ..runtime.cache import as_dynamic_cache
from ..runtime.tensor_ops import torch_clone_recursive
from ..runtime.text_generation import ChunkPrefillChunkGenerate
from ..runtime.text_generation import streaming_token_decoder
from ..runtime.tts_streaming import TTSSamplingParams
from ..runtime.tts_streaming import TTSStreamingGenerator

logger = logging.getLogger(__name__)


class StreamingGenerationMixin:
    @torch.inference_mode()
    def non_streaming_prefill(
        self,
        session_id,
        msgs,
        image=None,
        omni_mode=False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template=False,
        enable_thinking=False,
        stream_input=False,
        max_inp_length=8192,
        merge_audio_from_same_content=True,
    ):
        """一次性 prefill 所有消息到 KV cache（非流式，复用 chat 的消息解析逻辑）

        与 streaming_prefill 的区别：
        - streaming_prefill 每次处理 1 条 msg，需要调用多次
        - non_streaming_prefill 一次处理所有 msgs，只调用一次

        两者都不加 generation_prompt，都设置好 KV cache 状态，
        之后统一用 streaming_generate() 或 non_streaming_generate() 做解码。

        Args:
            session_id: 会话 ID
            msgs: 消息列表 [{role, content}, ...]，content 可含 PIL.Image / np.ndarray / str
            image: 兼容 chat() 的 image 参数（一般传 None，图像集成到 msgs 中）
            omni_mode: 是否为 omni 模式（视频输入时为 True）
            max_slice_nums: HD 图像最大切片数
            use_image_id: 是否使用图像 ID
            use_tts_template: 是否使用 TTS 模板
            enable_thinking: 是否启用思考模式
            stream_input: 音频输入模式（False=完整音频）
            max_inp_length: 最大输入长度
            merge_audio_from_same_content: 是否合并同一 content 中的音频

        Returns:
            str: 构建的 prompt 字符串
        """
        assert session_id is not None, "session_id cannot be None"

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        # ── 1. 消息解析（复用 chat() 的逻辑） ──

        if isinstance(msgs, str):
            msgs = json.loads(msgs)

        copy_msgs = deepcopy(msgs)
        assert len(copy_msgs) > 0, "msgs is empty"

        if image is not None and isinstance(copy_msgs[0]["content"], str):
            copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

        images = []
        audios = []
        audio_parts = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["system", "user", "assistant"]
            if i == 0:
                assert role in ["user", "system"], "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("<image>./</image>")
                elif isinstance(c, np.ndarray):
                    audios.append(c)
                    audio_parts.append(i)
                    cur_msgs.append("<audio>./</audio>")
                    use_tts_template = True
                elif isinstance(c, str):
                    cur_msgs.append(c)

            if omni_mode or stream_input:
                msg["content"] = "".join(cur_msgs)
            else:
                msg["content"] = "\n".join(cur_msgs)

        prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=False,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )

        if not merge_audio_from_same_content:
            audio_parts = None

        # ── 2. Tokenize + 预处理 ──

        inputs = self.processor(
            [prompt],
            [images],
            [audios],
            [audio_parts] if audio_parts is not None else None,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            stream_input=stream_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        inputs.pop("image_sizes", None)

        # ── 3. Session 状态初始化（与 streaming_prefill 对齐） ──

        self.reset_session(reset_token2wav_cache=False)
        self.session_id = session_id
        self.init_streaming_processor()

        # ── 4. Embedding 计算 ──

        model_inputs = {
            "input_ids": inputs["input_ids"],
            "audio_features": inputs.get("audio_features"),
            "audio_feature_lens": inputs.get("audio_feature_lens"),
            "image_bound": inputs.get("image_bound"),
            "audio_bounds": inputs.get("audio_bounds"),
            "spk_bounds": inputs.get("spk_bounds"),
        }

        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs["pixel_values"]
            model_inputs["tgt_sizes"] = inputs.get("tgt_sizes")

        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        inputs_embeds = self.get_omni_embedding(
            model_inputs,
            input_embeddings=model_inputs["inputs_embeds"],
            chunk_length=self.config.audio_chunk_length,
        )

        # ── 5. KV Cache Prefill ──

        round_id = self._next_round_id
        self._pending_round_id = round_id
        seq_len = inputs_embeds.shape[1]
        self._enforce_text_window()
        cache_length = self._get_kv_cache_length()

        attention_mask = torch.ones(
            (1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device
        )

        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )

        self.llm_past_key_values = as_dynamic_cache(outputs["past_key_values"])
        self._register_chunk(
            seq_len,
            "user",
            round_id=round_id,
            input_ids=inputs["input_ids"],
            tokenizer=self.processor.tokenizer,
        )
        self._enforce_text_window()
        if self.force_rope_reindex:
            self._force_reindex_all_cache()

        logger.info(
            f"non_streaming_prefill done: session={session_id}, "
            f"prompt_len={seq_len}, kv_cache_len={self._get_kv_cache_length()}"
        )

        return prompt

    @torch.inference_mode()
    def non_streaming_generate(
        self,
        session_id,
        max_new_tokens=256,
        do_sample=True,
        min_new_tokens=0,
        generate_audio=False,
        use_tts_template=True,
        enable_thinking=False,
        tts_ref_audio=None,
        tts_sampling_params=None,
        output_audio_path=None,
        length_penalty=1.1,
        tts_proj_layer=-1,
    ):
        """基于已有 KV cache 做非流式 HF generate + 可选 TTS

        必须在 non_streaming_prefill() 之后调用。
        """
        assert self.llm_past_key_values is not None, \
            "KV cache is empty — call non_streaming_prefill() first"

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
        tokenizer = self.processor.tokenizer

        # 1. 构建 bos string（与 streaming_generate 对齐）
        bos_input = "".join([
            "<|im_end|>\n<|im_start|>assistant\n",
            "" if enable_thinking else self.think_str.replace("\\n", "\n"),
            "<|tts_bos|>" if use_tts_template else "",
        ])

        bos_input_ids = tokenizer.encode(bos_input)
        bos_input_ids = torch.tensor(
            bos_input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        bos_embeds = self.llm.get_input_embeddings()(bos_input_ids)

        # 2. bos prefill（注入 KV cache）
        cache_length = self._get_kv_cache_length()
        attention_mask = torch.ones(
            (1, cache_length + bos_embeds.shape[1]),
            dtype=torch.bool, device=self.device,
        )

        bos_outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=bos_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )
        self.llm_past_key_values = as_dynamic_cache(bos_outputs["past_key_values"])

        bos_seq_len = bos_embeds.shape[1]
        round_id = self._next_round_id
        self._pending_round_id = round_id
        self._register_chunk(
            bos_seq_len, "assistant", round_id=round_id,
            input_ids=bos_input_ids, tokenizer=tokenizer,
        )

        # 3. HF generate（基于 KV cache）
        generation_config = self.prepare_generation_config(
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
        )
        generation_config.pop("max_new_tokens", None)

        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]

        cache_length_for_gen = self._get_kv_cache_length()
        gen_attention_mask = torch.ones(
            (1, cache_length_for_gen + 1),
            dtype=torch.bool, device=self.device,
        )

        last_logits = bos_outputs.logits[:, -1:, :]
        next_token = torch.argmax(last_logits, dim=-1)

        outputs = self.llm.generate(
            input_ids=next_token,
            past_key_values=self.llm_past_key_values,
            attention_mask=gen_attention_mask,
            pad_token_id=0,
            eos_token_id=terminators,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **generation_config,
        )

        # 4. 文本提取
        generated_ids = outputs.sequences[0]
        full_sequence = torch.cat([bos_input_ids[0], generated_ids])
        full_sequences = full_sequence.unsqueeze(0)
        outputs["full_sequences"] = full_sequences

        self._last_chat_token_stats = {
            "input_tokens": cache_length + bos_seq_len,
            "generated_tokens": len(generated_ids),
        }

        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )
        for term_str in self.terminators:
            text = text.replace(term_str, "")
        text = text.rstrip("<|tts_eos|>").strip()

        # 更新 KV cache 状态
        self.llm_past_key_values = as_dynamic_cache(outputs.past_key_values) \
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None \
            else self.llm_past_key_values
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = True

        # 5. TTS（可选）
        generated_waveform = None
        if use_tts_template and generate_audio:
            try:
                tts_bos_token = tokenizer.convert_tokens_to_ids("<|tts_bos|>")
                tts_eos_token = tokenizer.convert_tokens_to_ids("<|tts_eos|>")

                tts_bos_indices = []
                tts_eos_indices = []
                for i, x in enumerate(full_sequences[0]):
                    if x == tts_bos_token:
                        tts_bos_indices.append(i + 1)
                    elif x == tts_eos_token:
                        tts_eos_indices.append(i)

                tts_bos_idx = tts_bos_indices[-1] if tts_bos_indices else -1
                tts_eos_idx = tts_eos_indices[-1] if tts_eos_indices else None
                tts_bound = (tts_bos_idx, tts_eos_idx)

                _tts_audio_prompt = tts_ref_audio
                if _tts_audio_prompt is not None:
                    logger.info(f"[non_streaming_generate TTS] ref_audio: {len(_tts_audio_prompt)} samples")
                else:
                    logger.warning("[non_streaming_generate TTS] No ref audio")

                if tts_sampling_params is None:
                    tts_sampling_params = TTSSamplingParams()

                generated_waveform = self._generate_speech_non_streaming(
                    outputs=outputs,
                    tts_bound=tts_bound,
                    tts_proj_layer=tts_proj_layer,
                    audio_prompt=_tts_audio_prompt,
                    tts_sampling_params=tts_sampling_params,
                )
                if isinstance(generated_waveform, torch.Tensor):
                    generated_waveform = generated_waveform.cpu().numpy()

                if output_audio_path and generated_waveform is not None:
                    import soundfile as sf
                    sf.write(output_audio_path, generated_waveform, samplerate=24000)
            except:
                import traceback
                traceback.print_exc()
                generated_waveform = None

        logger.info(
            f"non_streaming_generate done: session={session_id}, "
            f"generated_tokens={len(generated_ids)}, "
            f"kv_cache_len={self._get_kv_cache_length()}, "
            f"has_audio={generated_waveform is not None}"
        )

        if generated_waveform is not None:
            return text, generated_waveform
        return text

    @torch.inference_mode()
    def streaming_prefill(
        self,
        session_id,
        msgs,
        tokenizer=None,  # deprecated
        omni_mode=True,
        max_slice_nums=None,
        use_tts_template=True,
        enable_thinking=False,
        is_last_chunk=False,  # for audio chunk, if is the last chunk, set to True
        stream_input=None,  # None=auto (is_not_system_prefill), False=完整音频, True=实时流式音频(双工)
        **kwargs,
    ):
        assert session_id is not None, "session_id cannot be None"
        self.is_first = self.session_id is None or session_id != self.session_id

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        images = []
        audios = []

        assert len(msgs) == 1
        copy_msgs = deepcopy(msgs)
        msg = copy_msgs[0]

        assert msg["role"] in ["system", "user", "assistant"]
        is_not_system_prefill = msg["role"] != "system"

        content = msg["content"]
        cur_msgs = []
        for j, c in enumerate(content):
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("<image>./</image>")
            elif isinstance(c, np.ndarray):
                audios.append(c)
                cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur_msgs.append(c)
            else:
                logger.error(f"Invalid content type: {c}, ignore it.")

        cur_contents = "".join(cur_msgs) if omni_mode else "\n".join(cur_msgs)

        if msg["role"] in ["system", "assistant"]:
            self.new_user_msg = True
            self.audio_past_key_values = None

        if self.is_first:
            self.reset_session(reset_token2wav_cache=False)
            self.session_id = session_id

            self.init_streaming_processor()

            if msg["role"] == "user":
                # 没有 system prefill，第一个 user turn 的第一个 segment
                # 不使用 apply_chat_template，手动构建 prompt 以避免自动添加 <|im_end|>
                prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False  # 标记后续 segments 不需要再添加 user 前缀
            else:
                # system 或 assistant prefill，使用 apply_chat_template
                msg["content"] = cur_contents
                prompt = self.processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            add_special_tokens = True  # add bos
        else:
            # 非首次 prefill
            if self.new_user_msg and msg["role"] == "user":
                # 新的 user turn 的第一个 segment
                if self.llm_generated:
                    # todo: when to set llm_generate_completed?
                    if self.llm_generate_completed:
                        prompt = "<|im_end|>\n<|im_start|>user\n" + cur_contents
                    else:
                        prompt = "<|tts_eos|><|im_end|>\n<|im_start|>user\n" + cur_contents
                else:
                    prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False
            else:
                # 同一个 turn 的后续 segments，直接使用内容
                prompt = cur_contents
            add_special_tokens = False

        # when first user audio prefill, ensure audio length satisfies FIRST_CHUNK_MS requirements
        if is_not_system_prefill and len(audios) > 0 and self.audio_chunk_idx == 0:
            assert len(audios) == 1, f"streaming mode only supports single audio, currently {len(audios)}"
            first_chunk_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
            if len(audios[0]) < first_chunk_samples:
                pad_len = first_chunk_samples - len(audios[0])
                audios[0] = np.concatenate([np.zeros(pad_len, dtype=audios[0].dtype), audios[0]])

        # stream_input: None=auto, False=完整音频, True=实时流式（双工）
        _stream_input = stream_input if stream_input is not None else is_not_system_prefill

        # online_streaming: 控制 processor 是否使用流式 mel 处理
        # 完整音频（stream_input=False）时不用流式 mel
        _online_streaming = is_not_system_prefill if _stream_input else False

        model_inputs = self.processor(
            [prompt],
            [images],
            [audios],
            max_slice_nums=1 if max_slice_nums is None else max_slice_nums,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            max_length=None,
            sampling_rate=16000,
            add_special_tokens=add_special_tokens,
            online_streaming=_online_streaming,
            audio_chunk_idx=self.audio_chunk_idx,
            is_last_chunk=is_last_chunk,
        ).to(self.device)

        # DEBUG: 打印 mel 特征的 checksum（用于诊断 rollback 不一致问题）
        if len(audios) > 0 and is_not_system_prefill and hasattr(self, "_debug_prefill") and self._debug_prefill:
            audio_feats = model_inputs.get("audio_features", None)
            if audio_feats is not None and hasattr(audio_feats, "sum"):
                mel_sum = audio_feats.sum().item()
                mel_shape = audio_feats.shape
                print(
                    f"[DEBUG prefill] audio_chunk_idx={self.audio_chunk_idx}, mel_sum={mel_sum:.6f}, mel_shape={mel_shape}"
                )
            else:
                print(f"[DEBUG prefill] audio_chunk_idx={self.audio_chunk_idx}, audio_feats type={type(audio_feats)}")

        if len(audios) > 0 and is_not_system_prefill:
            self.audio_chunk_idx += 1

        # 1. prepare input embeddings
        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        # get audio embedding with audio_past_key_values
        # todo: should pass chunk_length=self.config.audio_chunk_length ?
        inputs_embeds = self.get_omni_embedding(
            model_inputs, input_embeddings=model_inputs["inputs_embeds"], stream_input=_stream_input
        )

        # DEBUG: 打印 inputs_embeds 的 checksum
        if len(audios) > 0 and is_not_system_prefill and hasattr(self, "_debug_prefill") and self._debug_prefill:
            embed_sum = inputs_embeds.sum().item()
            embed_shape = inputs_embeds.shape
            print(f"[DEBUG prefill] inputs_embeds sum={embed_sum:.6f}, shape={embed_shape}")

        if self.is_first:
            self.audio_past_key_values = None  # clean audio_past_key_values after first prefill

        round_id = self._next_round_id
        self._pending_round_id = round_id
        chunk_type = "system" if msg["role"] == "system" else ("user" if msg["role"] == "user" else "assistant")
        seq_len = inputs_embeds.shape[1]
        self._enforce_text_window()
        cache_length = self._get_kv_cache_length()

        attention_mask = torch.ones((1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)

        # 2. do prefill
        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )

        self.llm_past_key_values = as_dynamic_cache(outputs["past_key_values"])
        self._register_chunk(
            seq_len,
            chunk_type,
            round_id=round_id,
            input_ids=model_inputs["input_ids"],
            tokenizer=self.processor.tokenizer,
        )
        self._enforce_text_window()
        if self.force_rope_reindex:
            self._force_reindex_all_cache()

        return prompt

    @torch.inference_mode()
    def streaming_generate(
        self,
        session_id,
        tokenizer=None,  # deprecated
        bos_input=None,
        generate_audio=True,
        audio_token_chunk_size=25,  # 25 token/s
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        max_new_tokens=256,
        fn="chunk_generate",
        enable_thinking=False,
        use_tts_template=True,
        do_sample=True,
        enable_speculative_snapshot=False,
        **kwargs,
    ):
        # 保存抢跑快照（在修改任何状态之前）
        # 用于 VAD 抢跑场景：如果抢跑失败，可调用 restore_speculative_snapshot() 恢复
        # enable_speculative_snapshot=True 时启用，False 时跳过（节省少量开销）
        if enable_speculative_snapshot:
            self._speculative_snapshot = self._save_speculative_snapshot()

        # reset buf
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = False
        self.audio_past_key_values = None

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        # reset current turn generated token IDs
        if hasattr(self, "_streaming_generated_token_ids"):
            del self._streaming_generated_token_ids
        # reset full generated text
        if hasattr(self, "_last_streaming_text"):
            del self._last_streaming_text

        cache = self._ensure_dynamic_cache()
        cache_length = self._get_kv_cache_length(cache)
        host_round_id = self._pending_round_id
        logger.info("streaming_generate kv cache length before= %s", cache_length)

        ## 单工情况每调用一次 streaming_generate 需要重新初始化 streaming_processor, 进入下一个 turn
        self.init_streaming_processor()

        # 1) llm generate token and hidden states per chunk=10, 2) tts generate audio token chunk per chunk=25, 3) yield 1 chunk audio token
        def audio_chunk_generator(
            bos_input,
            tokenizer,
            generate_audio,
            tts_sampling_params,
            max_new_tokens,
            do_sample,
            **kwargs,
        ):
            generate_chunk_size = 10

            if bos_input is None:
                bos_input = "".join(
                    [
                        "<|im_end|>\n<|im_start|>assistant\n",
                        "" if enable_thinking else self.think_str.replace("\\n", "\n"),
                        "<|tts_bos|>" if use_tts_template else "",
                    ]
                )

            bos_input_ids = tokenizer.encode(bos_input)
            bos_input_ids = torch.tensor(bos_input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            # DEBUG: 打印生成开始时的状态
            _cache_len = self._get_kv_cache_length()
            _cache_sum = self.llm_past_key_values.key_cache[0].sum().item() if self.llm_past_key_values else 0
            # 检查 KV Cache 最后几个位置的值
            _k_last = (
                self.llm_past_key_values.key_cache[0][0, 0, -5:, :3].flatten().tolist()
                if self.llm_past_key_values
                else []
            )
            print(f"[DEBUG streaming_generate] cache_len={_cache_len}, cache_sum={_cache_sum:.6f}, k_last={_k_last}")

            bos_input_embeds = self.llm.get_input_embeddings()(bos_input_ids)

            generation_inputs_embeds = bos_input_embeds
            generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)

            num_chunks_decode = (max_new_tokens + generate_chunk_size - 1) // generate_chunk_size

            conditions = []

            # generate chunk by chunk, each chunk has 10 tokens, each chunk takes last hidden states, and pass tokens to tts
            llm_streaming_generator = ChunkPrefillChunkGenerate(
                model=self.llm,
                tokenizer=tokenizer,
                terminators=["<|tts_eos|>", "<|im_end|>", "</s>"],
            )

            if generate_audio:
                logits_warpers, logits_processors = gen_logits(
                    num_code=self.tts.config.num_audio_tokens,
                    repetition_penalty=tts_sampling_params.repetition_penalty,
                    top_p=tts_sampling_params.top_p,
                    top_k=tts_sampling_params.top_k,
                )

                tts_streaming_generator = TTSStreamingGenerator(
                    model=self.tts,
                    temperature=tts_sampling_params.temperature,
                    eos_token=torch.tensor(
                        [self.tts.config.num_audio_tokens - 1],
                        dtype=torch.long,
                        device=self.tts.device,
                    ),
                    chunk_size=audio_token_chunk_size,  # s3tokenizer 1s = 25token
                    tts_last_turn_tokens=self.tts_last_turn_tokens,
                    logits_processors=logits_processors,
                    logits_warpers=logits_warpers,
                )

            # LLM chunk generate outer loop
            for chunk_idx in range(num_chunks_decode):
                is_first_generate_chunk = chunk_idx == 0

                output = llm_streaming_generator.chunk_generate(
                    inputs_embeds=generation_inputs_embeds,
                    past_key_values=self.llm_past_key_values,
                    is_first_generate_chunk=is_first_generate_chunk,
                    return_hidden_states=True,
                    chunk_size=generate_chunk_size + 1 * is_first_generate_chunk,
                    do_sample=do_sample,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.8),
                    top_k=kwargs.get("top_k", 100),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.02),
                    length_penalty=kwargs.get("length_penalty", 1.0),
                    all_input_ids=generated_ids,
                    suppress_forbidden_tokens=generate_audio,
                )

                if output.chunk_token_ids is None:
                    break

                # DEBUG: 打印第一个 chunk 生成的 token
                if chunk_idx == 0:
                    print(f"[DEBUG streaming_generate] first_chunk_tokens={output.chunk_token_ids.tolist()}")

                if is_first_generate_chunk:
                    if generate_audio:
                        spk_emb = torch.empty(
                            (bos_input_embeds.shape[0], 0, bos_input_embeds.shape[2]),
                            dtype=bos_input_embeds.dtype,
                            device=bos_input_embeds.device,
                        )
                        tts_streaming_generator.spk_emb = spk_emb

                    if output.finished:
                        yield_chunk_token_ids = output.chunk_token_ids
                    else:
                        # the first chunk generated chunk_size + 1 tokens, we only take the first chunk_size tokens,
                        # the last token is not prefilled, and last hidden states is not obtained
                        yield_chunk_token_ids = output.chunk_token_ids[:, :-1]

                elif output.finished:
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], output.chunk_token_ids], dim=1)
                else:
                    # in the chunk that is not the first chunk, we need to add the token at the end of the previous chunk,
                    # it is not prefilled into the model to get last hidden states
                    # similarly, the last generated token of subsequent chunks is not prefilled, and last hidden states is not obtained,
                    # so it is not passed out
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], output.chunk_token_ids[:, :-1]], dim=1)

                if not generate_audio:
                    chunk_generated_text = tokenizer.decode(yield_chunk_token_ids[0])
                    yield yield_chunk_token_ids, output.finished
                else:
                    # TTS inner loop
                    # dense connection here is hardcoded to use text-hidden merged as condition
                    llm_embeds = self.tts.emb_text(yield_chunk_token_ids)
                    hidden_embeds = output.last_hidden_states
                    hidden_embeds = self.tts.projector_semantic(hidden_embeds)
                    if self.tts.config.normalize_projected_hidden:  # default should be opened
                        hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

                    tts_embeds = llm_embeds + hidden_embeds
                    conditions.append(tts_embeds)

                    # Store token IDs instead of decoded text to avoid UTF-8 multi-byte character truncation
                    if not hasattr(self, "_streaming_generated_token_ids"):
                        self._streaming_generated_token_ids = []
                    self._streaming_generated_token_ids.extend(yield_chunk_token_ids[0].tolist())

                    # there is buffer generated, each time exactly returns 25 audio tokens,
                    # the last audio chunk returns audio tokens of variable length, length [0, 25]
                    tts_generator = tts_streaming_generator.generate_with_buffer(
                        condition=tts_embeds, text_finished=output.finished
                    )

                    for audio_token_chunk, is_last_audio_chunk in tts_generator:
                        yield audio_token_chunk, is_last_audio_chunk

                generated_ids = torch.cat([generated_ids, output.chunk_token_ids], dim=1)
                generation_inputs_embeds = output.current_inputs_embeds
                self.llm_past_key_values = output.past_key_values

                if output.finished:
                    if generate_audio:
                        self.tts_last_turn_tokens = tts_streaming_generator.tts_last_turn_tokens
                    break

            # IMPORTANT: Flush remaining TTS buffer when LLM generation ends
            # This handles BOTH cases:
            # 1. LLM finished with terminator (output.finished=True) - buffer may still have tokens
            # 2. LLM hit max chunks limit (output.finished=False) - buffer definitely has tokens
            if generate_audio:
                if len(tts_streaming_generator._token_buffer) > 0:
                    batch = torch.cat(tts_streaming_generator._token_buffer, dim=1)
                    yield batch, True
                    tts_streaming_generator._token_buffer = []

            if generate_audio:
                if hasattr(self, "_streaming_generated_token_ids"):
                    try:
                        self._last_streaming_text = tokenizer.decode(self._streaming_generated_token_ids)
                        assistant_input_ids = self._encode_text(tokenizer=tokenizer, text=self._last_streaming_text)
                        self._finalize_round(
                            round_id=host_round_id, cache_before=cache_length, assistant_input_ids=assistant_input_ids
                        )
                    except Exception:
                        self._last_streaming_text = None
                else:
                    self._last_streaming_text = None

                yield None, None
            else:
                return

        # iter for generating text chunk and audio chunk
        audio_chunk_generator_iter = audio_chunk_generator(
            bos_input=bos_input,
            tokenizer=self.processor.tokenizer,
            generate_audio=generate_audio,
            tts_sampling_params=tts_sampling_params,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        if generate_audio:
            if self.tts.config.audio_tokenizer_type == "s3tokenizer_step_audio":
                self.tts.audio_tokenizer.stream_cache = torch_clone_recursive(self.token2wav_cache["flow_cache_base"])
                self.tts.audio_tokenizer.hift_cache_dict = torch_clone_recursive(
                    self.token2wav_cache["hift_cache_base"]
                )

                # pre-insert 3-5 prefix 4218 silence tokens, each token corresponds to 0.04s,
                # adding 5 tokens means introducing 0.2s of silence
                buffer = [4218] * 3
                pre_lookahead = 3
                CHUNK_SIZE = 25
                chunk_idx = 0
                prev_text_len = 0  # track text position for streaming text output
                for audio_token_chunk, is_last_audio_chunk in audio_chunk_generator_iter:
                    if audio_token_chunk is None:
                        break

                    buffer += audio_token_chunk.reshape(-1).tolist()

                    if len(buffer) >= CHUNK_SIZE + pre_lookahead:
                        waveform_chunk = self.tts.audio_tokenizer.stream(
                            buffer[: CHUNK_SIZE + pre_lookahead],
                            prompt_wav=None,
                            last_chunk=is_last_audio_chunk,
                            return_waveform=True,
                        )

                        waveform_chunk = torch.from_numpy(waveform_chunk)

                        # get new text chunk corresponding to this waveform
                        # Decode from accumulated token IDs to avoid UTF-8 multi-byte truncation
                        new_text = ""
                        if hasattr(self, "_streaming_generated_token_ids"):
                            current_text = self.processor.tokenizer.decode(self._streaming_generated_token_ids)
                            # Filter out trailing replacement characters (incomplete UTF-8 sequences)
                            safe_end = len(current_text)
                            while safe_end > 0 and current_text[safe_end - 1] == "\ufffd":
                                safe_end -= 1
                            safe_text = current_text[:safe_end]
                            new_text = safe_text[prev_text_len:]
                            prev_text_len = len(safe_text)

                        yield waveform_chunk, new_text

                        buffer = buffer[CHUNK_SIZE:]
                        chunk_idx += 1

                # flush rest
                if len(buffer) > 0:
                    waveform_chunk = self.tts.audio_tokenizer.stream(
                        buffer,
                        prompt_wav=None,
                        last_chunk=True,
                        return_waveform=True,
                    )

                    waveform_chunk = torch.from_numpy(waveform_chunk)

                    # get remaining new text for the final chunk
                    # Final chunk: decode all remaining text without filtering
                    new_text = ""
                    if hasattr(self, "_streaming_generated_token_ids"):
                        current_text = self.processor.tokenizer.decode(self._streaming_generated_token_ids)
                        new_text = current_text[prev_text_len:]
                        prev_text_len = len(current_text)

                    yield waveform_chunk, new_text

                # maybe the buffer is empty, and text is not empty, should we flush text without wave?
            else:
                raise NotImplementedError(f"not supported audio tokenizer: {self.tts.config.audio_tokenizer_type}")
        else:
            # For text-only generation, decode tokens and handle partial multi-byte characters
            yield from streaming_token_decoder(
                audio_chunk_generator_iter,
                self.processor.tokenizer,
                skip_special_tokens=False,
            )

    # Duplex convenience methods are provided by DuplexProxyMixin.

