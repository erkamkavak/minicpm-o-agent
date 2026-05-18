"""Chat and speech generation helpers for the MiniCPMO wrapper."""

import json
import logging
import math
import os
import tempfile
from copy import deepcopy
from threading import Thread
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TextIteratorStreamer

from ..components import gen_logits
from ..processing_minicpmo import MiniCPMOProcessor
from ..runtime.tensor_ops import torch_clone_recursive
from ..runtime.tts_streaming import TTSSamplingParams

logger = logging.getLogger(__name__)


class ChatGenerationMixin:
    def _decode(self, inputs_embeds, tokenizer, attention_mask, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        return outputs

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_config = {
            "inputs_embeds": inputs_embeds,
            "pad_token_id": 0,
            "eos_token_id": terminators,
            "streamer": streamer,
        }
        generation_config.update(kwargs)
        thread = Thread(target=self.llm.generate, kwargs=generation_config)
        thread.start()
        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result))
        return result_text

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=None,
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        **kwargs,
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                model_inputs,
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=self.config.audio_chunk_length,
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
                outputs = {}  # if stream return TextIteratorStreamer and output is empty
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)
                result = self._decode_text(outputs.sequences, tokenizer)

        return result, outputs

    def _build_streaming_mask(self, tts_tokens_len):
        tts_sequence_full_length = 1 + self.tts.streaming_text_reserved_len + 1
        streaming_attention_mask = torch.zeros(tts_sequence_full_length, dtype=torch.int8)
        streaming_attention_mask[0 : 1 + 1 + tts_tokens_len + 1] = 1
        streaming_attention_mask[-1] = 1
        return streaming_attention_mask

    def _generate_mel_spec(self, inputs, outputs, text, output_chunk_size=25, tts_max_new_tokens=2048):
        spk_embeds = self._get_last_spk_embeds(inputs, outputs)

        text = text.split("<|tts_bos|>")[-1]
        gen_text = text.split("<|tts_eos|>")[0]
        tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
        tts_inputs = self.tts_processor.text_tokenizer.encode(tts_text, add_special_tokens=False)
        tts_input_ids = torch.Tensor(tts_inputs).unsqueeze(0).to(self.device, dtype=torch.long)
        streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=626,
            top_p=self.tts.top_p,
            top_k=self.tts.top_k,
            repetition_penalty=self.tts.repetition_penalty,
        )

        condition_length = 1 + self.tts.streaming_text_reserved_len + 1

        dtype = self.tts.emb_text.weight.dtype
        emb = torch.zeros(1, condition_length, self.tts.num_vq, dtype=dtype, device=self.tts.device)
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]

        audio_input_ids = torch.zeros(
            1,
            condition_length,
            self.tts.num_vq,
            dtype=torch.long,
            device=self.tts.device,
        )

        eos_lab = False
        for chunk_idx in range(math.ceil(emb.shape[1] / self.tts.streaming_text_chunk_size)):
            if chunk_idx == 0:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 0
                end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + 1
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 1
                end = min(
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size + 1,
                    condition_length - 1,
                )

            if end - begin > 0:
                text_input_ids = tts_input_ids[:, begin:end]
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                if begin == 0:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        lm_spk_emb_last_hidden_states=spk_embeds,
                    )
                else:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    )

            outputs = self.tts.generate(
                input_ids=audio_input_ids,
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
                max_new_token=output_chunk_size,
                force_no_stop=self.force_no_stop,
                temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
            )
            audio_input_ids = outputs.audio_input_ids
            past_key_values = outputs.past_key_values

            if outputs.finished:
                logger.debug("Generation finished.")
                eos_lab = True
                break

        if not eos_lab:
            logger.debug("eos_lab False, Generation continue.")
            while True:
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )

                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > tts_max_new_tokens:
                    logger.debug(f"Generation length > {tts_max_new_tokens}, stopped.")
                    break

    @staticmethod
    def prepare_generation_config(do_sample, max_new_tokens=50, min_new_tokens=0, **kwargs):
        num_beams = kwargs.get("num_beams", 3)
        generation_config = {
            "num_beams": num_beams,
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.02,
        }

        if do_sample:
            generation_config.update(
                {
                    "top_p": 0.8,
                    "top_k": 100,
                    "temperature": 0.7,
                    "do_sample": True,
                    "repetition_penalty": 1.02,
                }
            )
        elif num_beams > 1:
            generation_config.update({"num_beams": num_beams, "repetition_penalty": 1.2, "do_sample": False})
        else:
            generation_config.update({"do_sample": False, "repetition_penalty": 1.02})

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config["max_new_tokens"] = max_new_tokens

        return generation_config

    @torch.inference_mode()
    def chat(
        self,
        image=None,
        msgs=None,
        tokenizer=None,  # deprecated
        processor=None,  # deprecated
        vision_hidden_states=None,
        max_new_tokens=4096,
        min_new_tokens=0,
        do_sample=True,
        sampling=None,  # deprecated, please use do_sample
        max_inp_length=8192,
        stream=False,
        stream_input=False,
        max_slice_nums=None,
        use_image_id=None,
        enable_thinking=False,
        use_tts_template=False,
        generate_audio=False,
        output_audio_path=None,
        output_tts_inputs_embeds_path=None,
        # add
        omni_mode=False,
        omni_input=None,  # deprecated, please use omni_mode
        teacher_forcing=False,
        return_prompt=False,
        tts_proj_layer=-1,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        merge_audio_from_same_content=True,
        tts_ref_audio: Optional[np.ndarray] = None,
        **kwargs,
    ):
        # todo: deprecated
        if sampling is not None:
            do_sample = sampling
        if omni_input is not None:
            omni_mode = omni_input

        batched = isinstance(msgs[0], list)
        msgs_list = msgs
        images_list = image

        if not batched:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert do_sample or not stream, "if use stream mode, make sure do_sample=True"

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
                    elif isinstance(c, np.ndarray):  # audio
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

            prompts_lists.append(
                self.processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False if teacher_forcing else True,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        if not merge_audio_from_same_content:
            audio_parts_list = None

        inputs = self.processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            stream_input=stream_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        generation_config = self.prepare_generation_config(
            do_sample=do_sample, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, **kwargs
        )
        generation_config.pop("max_new_tokens", None)

        inputs.pop("image_sizes")

        # teacher_forcing = True => generate audio with given text
        with torch.inference_mode():
            res, outputs = self.generate(
                **inputs,
                tokenizer=self.processor.tokenizer,
                max_new_tokens=1 if teacher_forcing else max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config,
            )

        # spk bound and tts bound
        tts_bos_token = self.processor.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        tts_eos_token = self.processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>")

        # Combine input_ids and generated sequences to get complete sequence
        input_ids = inputs["input_ids"][0]
        generated_ids = outputs.sequences[0]
        # Combine by concatenating input_ids with the new tokens from generated sequence
        full_sequence = torch.cat([input_ids, generated_ids])
        # Update the sequences in outputs
        full_sequences = full_sequence.unsqueeze(0)

        outputs["full_sequences"] = full_sequences

        # 存储 token 统计，供 ChatView 等外部消费者读取
        # input_tokens: tokenizer 级别（含 audio/image 占位符，不含 embedding 展开）
        # generated_tokens: LLM 实际生成的 token 数
        self._last_chat_token_stats = {
            "input_tokens": len(input_ids),
            "generated_tokens": len(generated_ids),
        }

        tts_bos_indices = []
        tts_eos_indices = []
        for i, x in enumerate(full_sequences[0]):
            if x == tts_bos_token:
                tts_bos_indices.append(i + 1)  # tts_bos + 1 才是第一个tts的位置，这样方便直接给tts去slice hidden states
            elif x == tts_eos_token:
                if teacher_forcing and i == len(full_sequences[0]) - 1:
                    continue
                tts_eos_indices.append(i)

        tts_bos_idx = tts_bos_indices[-1] if tts_bos_indices else -1
        # Use None instead of -1 when no EOS token found, so that slice [start:None]
        # means "to the end" rather than [start:-1] which excludes the last element
        tts_eos_idx = tts_eos_indices[-1] if tts_eos_indices else None

        tts_bound = (tts_bos_idx, tts_eos_idx)

        answer = res[0]
        if answer is not None:
            answer = answer.rstrip("<|tts_eos|>")

        generated_waveform = None
        if use_tts_template and generate_audio:
            try:
                # TTS ref audio 优先级：
                # 1. tts_ref_audio（显式指定的 TTS 参考音频，与 LLM ref audio 分离）
                # 2. input_audios_list[0][0]（messages 中第一个音频，即 LLM ref audio）
                # 3. None（无参考音频）
                if tts_ref_audio is not None:
                    _tts_audio_prompt = tts_ref_audio
                    logger.info(f"[Chat TTS] Using separate tts_ref_audio: {len(tts_ref_audio)} samples ({len(tts_ref_audio)/16000:.1f}s)")
                elif len(input_audios_list) > 0 and len(input_audios_list[0]) > 0:
                    _tts_audio_prompt = input_audios_list[0][0]
                    logger.info(f"[Chat TTS] Using LLM ref audio from messages: {len(_tts_audio_prompt)} samples ({len(_tts_audio_prompt)/16000:.1f}s)")
                else:
                    _tts_audio_prompt = None
                    logger.warning("[Chat TTS] No ref audio available for TTS")

                generated_waveform = self._generate_speech_non_streaming(
                    outputs=outputs,
                    tts_bound=tts_bound,
                    tts_proj_layer=tts_proj_layer,
                    audio_prompt=_tts_audio_prompt,
                    output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
                    tts_sampling_params=tts_sampling_params,
                )
                # 统一为 numpy array
                if isinstance(generated_waveform, torch.Tensor):
                    generated_waveform = generated_waveform.cpu().numpy()

                # 如果指定了保存路径，也保存到文件
                if output_audio_path and generated_waveform is not None:
                    import soundfile as sf
                    sf.write(output_audio_path, generated_waveform, samplerate=24000)
                    logger.debug(f"audio saved to {output_audio_path}")
            except:
                import traceback
                traceback.print_exc()
                generated_waveform = None

        if return_prompt:
            return answer, prompts_lists[0], generated_waveform
        elif generated_waveform is not None:
            return answer, generated_waveform
        else:
            return answer

    @torch.inference_mode()
    def _generate_speech_non_streaming(
        self,
        outputs,
        tts_bound,
        tts_proj_layer,
        audio_prompt,
        output_tts_inputs_embeds_path=None,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
    ):
        last_hidden_states = [hs[tts_proj_layer] for hs in outputs.hidden_states]
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])

        # FIX: 某些 pt 权重可能导致 hidden_states 和 sequences 长度不一致
        # 以 full_sequences 为准（这是实际要用的 tokens）
        full_seq_len = len(outputs["full_sequences"][0])
        if last_hidden_states.shape[0] != full_seq_len:
            logger.warning(f"TTS: hidden_states({last_hidden_states.shape[0]}) != full_sequences({full_seq_len}), truncating")
            last_hidden_states = last_hidden_states[:full_seq_len]

        spk_embeds = (
            torch.ones([0, self.tts.config.hidden_size]).to(last_hidden_states.device).to(last_hidden_states.dtype)
        )

        if self.tts.condition_type == "hidden_text_merge":
            llm_tokens = outputs["full_sequences"][0][tts_bound[0] : tts_bound[1]]
            llm_tokens = torch.tensor(llm_tokens, device=self.tts.emb_text.weight.device, dtype=torch.long)
            llm_embeds = self.tts.emb_text(llm_tokens)  # make sure emb_text is compatible with llm vocab size

            hidden_embeds = last_hidden_states[tts_bound[0] : tts_bound[1]]
            hidden_embeds = self.tts.projector_semantic(hidden_embeds)

            if self.tts.config.normalize_projected_hidden:
                hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

            tts_embeds = llm_embeds + hidden_embeds
            if self.tts.interleaved:
                chunks = []
                cond_length = tts_embeds.shape[0]
                for i in range(0, cond_length, 10):
                    chunks.append(tts_embeds[i : i + 10])
                tts_embeds = chunks
        else:
            raise NotImplementedError

        audio_bos = [self.tts.audio_bos_token_id]
        audio_bos = torch.tensor(audio_bos, device=self.tts.emb_text.weight.device, dtype=torch.long)

        audio_bos_embeds = self.tts.emb_text(audio_bos)

        text_eos_embed = self.tts.emb_text(
            torch.tensor(
                [self.tts.config.text_eos_token_id],
                device=self.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        )

        if self.tts.interleaved:

            tts_embeds[-1] = torch.cat([tts_embeds[-1], text_eos_embed], dim=0)
            for i in range(len(tts_embeds)):
                tts_embeds[i] = torch.cat([tts_embeds[i], audio_bos_embeds], dim=0).unsqueeze(0)
            outputs = self.tts.interleaved_generate(
                spk_embeds=spk_embeds,
                conditions=tts_embeds,
                temperature=0.8,
                repetition_penalty=1.05,
                eos_token=torch.tensor(
                    [self.tts.config.num_audio_tokens - 1],
                    dtype=torch.long,
                    device=self.tts.device,
                ),
            )
        else:
            if self.tts.condition_type == "tts_token":
                inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds], dim=0).unsqueeze(
                    0
                )
            elif self.tts.condition_type == "tts_token_streaming":
                tts_embeds[1] = spk_embeds.squeeze(0)  # apply speaker embedding
                inputs_embeds = tts_embeds.unsqueeze(0)
            else:  # modern case
                inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds], dim=0).unsqueeze(
                    0
                )

            # save inputs_embeds to file
            if output_tts_inputs_embeds_path:
                torch.save(inputs_embeds, output_tts_inputs_embeds_path)

            outputs = self.tts.generate(
                inputs_embeds=inputs_embeds,
                sampling_params=tts_sampling_params,
                eos_token=torch.tensor(
                    [self.tts.config.num_audio_tokens - 1],
                    dtype=torch.long,
                    device=self.tts.device,
                ),
            )

        if self.tts.config.audio_tokenizer_type == "s3tokenizer":
            # ========== CosyVoice2 vocoder 路径 ==========
            generated_tokens = outputs.new_ids.squeeze(-1)
            reference_audio = audio_prompt
            if reference_audio is not None:
                logger.debug("use reference audio in data to generate waveform")
                prompt_speech_16k = torch.tensor(reference_audio).unsqueeze(0)

            if self.tts.config.s3_stream_generate:
                waveform_pred = self.tts.audio_tokenizer.inference_token2wav(
                    speech_tokens=generated_tokens,
                    prompt_speech_16k=prompt_speech_16k,
                    prompt_speech=None,
                    stream=True,
                    n_timesteps=self.tts.config.s3_stream_n_timesteps,
                    code_chunk_size=self.tts.config.s3_stream_chunk_size,
                    chunk_prelook_size=self.tts.config.s3_stream_prelook_size,
                    use_attn_idx=False,
                )
                return waveform_pred[0]
            else:
                for i, j in enumerate(
                    self.tts.audio_tokenizer.token2wav(
                        speech_token=generated_tokens,
                        speech_token_len=torch.tensor([generated_tokens.shape[1]], device=generated_tokens.device),
                        prompt_speech_16k=prompt_speech_16k,
                        stream=False,
                    )
                ):
                    waveform_pred = j["tts_speech"]
                    waveform_sample_rate = self.tts.audio_tokenizer.sample_rate  # 24000 here, not 16000 input.
                return waveform_pred[0]

        elif self.tts.config.audio_tokenizer_type == "s3tokenizer_step_audio":
            # ========== Token2Wav vocoder 路径（非流式批量转换）==========
            generated_tokens = outputs.new_ids.squeeze(-1)
            token_ids = generated_tokens.reshape(-1).tolist()

            if not token_ids:
                logger.warning("Token2Wav non-streaming: 无 audio tokens 可转换")
                return None

            # Token2Wav 需要文件路径作为 prompt，将 ref audio 写入临时文件
            reference_audio = audio_prompt
            prompt_wav_path = None
            temp_file = None

            if reference_audio is not None:
                logger.debug("use reference audio in data to generate waveform (Token2Wav)")
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="t2w_ref_")
                sf.write(temp_file.name, reference_audio, 16000)
                prompt_wav_path = temp_file.name

            try:
                # 初始化 Token2Wav stream cache
                # set_stream_cache() 返回 (flow_cache, hift_cache)，需手动设置到实例属性
                self.tts.audio_tokenizer.cache = None
                flow_cache, hift_cache = self.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)
                self.tts.audio_tokenizer.stream_cache = flow_cache
                self.tts.audio_tokenizer.hift_cache_dict = hift_cache

                # Token2Wav 的 flow 模型有固定大小的 attention cache buffer，
                # 必须分块喂入（与 streaming 路径一致），否则长文本会溢出。
                CHUNK_SIZE = 25
                pre_lookahead = 3
                waveform_chunks = []
                buffer = list(token_ids)
                pos = 0

                while pos + CHUNK_SIZE + pre_lookahead <= len(buffer):
                    chunk = buffer[pos : pos + CHUNK_SIZE + pre_lookahead]
                    is_last = (pos + CHUNK_SIZE + pre_lookahead >= len(buffer))
                    wav_chunk = self.tts.audio_tokenizer.stream(
                        chunk, prompt_wav=prompt_wav_path,
                        last_chunk=is_last, return_waveform=True,
                    )
                    if wav_chunk is not None:
                        waveform_chunks.append(wav_chunk.squeeze())
                    pos += CHUNK_SIZE

                # flush 剩余 tokens
                if pos < len(buffer):
                    remaining = buffer[pos:]
                    wav_chunk = self.tts.audio_tokenizer.stream(
                        remaining, prompt_wav=prompt_wav_path,
                        last_chunk=True, return_waveform=True,
                    )
                    if wav_chunk is not None:
                        waveform_chunks.append(wav_chunk.squeeze())

                if waveform_chunks:
                    waveform = np.concatenate(waveform_chunks)
                    logger.info(
                        f"Token2Wav non-streaming: {len(token_ids)} tokens → "
                        f"{len(waveform)} samples ({len(waveform)/24000:.2f}s), "
                        f"{len(waveform_chunks)} chunks"
                    )
                    return waveform
                else:
                    logger.warning("Token2Wav non-streaming: 所有 chunks 返回空")
                    return None
            finally:
                # 清理临时文件
                if temp_file is not None:
                    try:
                        os.unlink(temp_file.name)
                    except OSError:
                        pass
        else:
            raise NotImplementedError(
                f"不支持的 audio_tokenizer_type: {self.tts.config.audio_tokenizer_type}"
            )

    @torch.inference_mode()
    def init_token2wav_cache(self, prompt_speech_16k):
        if hasattr(self.tts.audio_tokenizer, "set_stream_cache"):
            self.tts.audio_tokenizer.cache = None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                prompt_wav_path = tmp_wav.name
                sf.write(prompt_wav_path, prompt_speech_16k, 16000)
                flow_cache_base, hift_cache_base = self.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)

            self.token2wav_cache = {
                "flow_cache_base": torch_clone_recursive(flow_cache_base),
                "hift_cache_base": torch_clone_recursive(hift_cache_base),
            }
        else:
            model_input = self.tts.audio_tokenizer.frontend.frontend_token2wav(
                speech_tokens=torch.zeros(1, 1, dtype=torch.long, device=self.tts.device),
                speech_16k=None,
                prompt_speech_16k=prompt_speech_16k,
                resample_rate=self.tts.audio_tokenizer.sample_rate,
                prompt_speech=None,
            )

            prompt_token = model_input["flow_prompt_speech_token"]
            prompt_feat = model_input["prompt_speech_feat"]
            embedding = model_input["flow_embedding"]

            if self.tts.audio_tokenizer.fp16:
                prompt_feat = prompt_feat.to(torch.half)
                embedding = embedding.to(torch.half)

            prepared_cache = self.tts.audio_tokenizer.model.prepare_cache_from_prompt(
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                n_timesteps=self.tts.config.s3_stream_n_timesteps,
                code_chunk_size=self.tts.config.s3_stream_chunk_size,
                chunk_prelook_size=self.tts.config.s3_stream_prelook_size,
                use_attn_idx=False,
            )

            self.token2wav_cache = prepared_cache
