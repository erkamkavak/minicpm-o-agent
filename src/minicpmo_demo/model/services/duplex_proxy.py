"""Convenience pass-through methods for DuplexCapability."""

import logging
from typing import List
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DuplexProxyMixin:
    # ==================== Duplex 透传方法 ====================
    # 以下方法透传到 self.duplex，减少调用层级
    # 外部可以直接调用 model.duplex_prepare() 而不是 model.duplex.prepare()
    
    def duplex_prepare(
        self,
        prefix_system_prompt: Optional[str] = None,
        suffix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        context_previous_marker: str = "\n\nprevious: ",
    ):
        """准备双工会话（透传到 self.duplex.prepare）
    
        Args:
            prefix_system_prompt: system prompt 前缀
            suffix_system_prompt: system prompt 后缀
            ref_audio: 参考音频（16kHz numpy array）
            prompt_wav_path: TTS prompt 音频路径
            context_previous_marker: 上下文历史标记
    
        Returns:
            完整的 system prompt 字符串
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        return self.duplex.prepare(
            prefix_system_prompt=prefix_system_prompt,
            suffix_system_prompt=suffix_system_prompt,
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
            context_previous_marker=context_previous_marker,
        )
    
    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        max_slice_nums: int = 1,
    ):
        """预填充用户输入（透传到 self.duplex.streaming_prefill）
    
        Args:
            audio_waveform: 音频波形（16kHz numpy array）
            frame_list: 视频帧列表
            max_slice_nums: HD 图像切片数
    
        Returns:
            预填充结果 dict
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        return self.duplex.streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )
    
    def duplex_generate(
        self,
        decode_mode: str = "greedy",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        listen_prob_scale: Optional[float] = None,
        listen_top_k: int = 5,
        text_repetition_penalty: Optional[float] = None,
        text_repetition_window_size: Optional[int] = None,
        length_penalty: float = 1.1,
        force_listen_override: bool = False,
    ):
        """生成响应（透传到 self.duplex.streaming_generate）
    
        Args:
            decode_mode: 解码模式 ("greedy" 或 "sample")
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P 采样
            listen_prob_scale: Listen 概率缩放
            listen_top_k: Listen 判断的 top-k
            text_repetition_penalty: 文本重复惩罚
            text_repetition_window_size: 重复惩罚窗口大小
            length_penalty: 长度惩罚系数，>1.0 抑制 turn_eos 使输出更长
            force_listen_override: 前端 Force Listen 开关，强制本次生成为 listen
    
        Returns:
            生成结果 dict
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        return self.duplex.streaming_generate(
            decode_mode=decode_mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            listen_prob_scale=listen_prob_scale,
            listen_top_k=listen_top_k,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            length_penalty=length_penalty,
            force_listen_override=force_listen_override,
        )
    
    def duplex_finalize(self):
        """完成 streaming_generate 的延迟操作（透传到 self.duplex.finalize_unit）
    
        必须在 duplex_generate 之后、下一次 duplex_prefill 之前调用。
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        self.duplex.finalize_unit()
    
    def duplex_set_break(self):
        """设置打断信号（透传到 self.duplex.set_break_event）"""
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        self.duplex.set_break_event()
    
    def duplex_clear_break(self):
        """清除打断信号（透传到 self.duplex.clear_break_event）"""
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        self.duplex.clear_break_event()
    
    def duplex_stop(self):
        """停止当前会话（透传到 self.duplex.set_session_stop）"""
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
        self.duplex.set_session_stop()
    
    def duplex_is_break_set(self) -> bool:
        """检查是否设置了打断（透传到 self.duplex.is_break_set）"""
        if self.duplex is None:
            return False
        return self.duplex.is_break_set()
    
    def duplex_is_stopped(self) -> bool:
        """检查会话是否已停止（透传到 self.duplex.is_session_stop_set）"""
        if self.duplex is None:
            return False
        return self.duplex.is_session_stop_set()
    
    def duplex_chat(
        self,
        user_audio: np.ndarray,
        system_prompt: str = "You are a helpful assistant.",
        ref_audio: Optional[np.ndarray] = None,
        ref_audio_path: Optional[str] = None,
        image_list: Optional[List] = None,
        chunk_ms: int = 1000,
        sample_rate: int = 16000,
        generate_audio: bool = True,
        decode_mode: str = "greedy",
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.8,
        force_listen_count: int = 3,
    ) -> dict:
        """双工离线推理（便捷方法）
    
        对完整音频进行离线双工对话，一站式处理。
    
        适用场景：
        - 离线批量处理音频文件
        - 单元测试
        - 演示场景
    
        注意：这不是实时双工会话，而是对完整音频的离线处理。
        实时双工请使用 duplex_prepare/duplex_prefill/duplex_generate 原语。
    
        Args:
            user_audio: 用户音频波形（16kHz numpy array）
            system_prompt: 系统提示文本
            ref_audio: 参考音频波形（16kHz numpy array，用于 TTS）
            ref_audio_path: 参考音频路径（用于 TTS，与 ref_audio 二选一）
            image_list: 图像列表（视频双工，每个 chunk 一张 PIL Image）
            chunk_ms: 每个 chunk 的时长（毫秒）
            sample_rate: 音频采样率
            generate_audio: 是否生成音频
            decode_mode: 解码模式 ("greedy" 或 "sampling")
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P 采样
            force_listen_count: 强制 listen 的 chunk 数
    
        Returns:
            dict: {
                "success": bool,
                "full_text": str,
                "chunks": List[dict],
                "audio_chunks": List[np.ndarray],
                "error": Optional[str],
            }
    
        示例：
            >>> result = model.duplex_chat(
            ...     user_audio=audio_16k,
            ...     system_prompt="你是一个友好的助手。",
            ...     ref_audio_path="/path/to/ref.wav",
            ... )
            >>> print(result["full_text"])
        """
        if self.duplex is None:
            raise RuntimeError("Duplex 未初始化，请先调用 init_unified()")
    
        chunks = []
        full_text = ""
        audio_chunks = []
    
        try:
            # 准备会话
            self.duplex_prepare(
                prefix_system_prompt=system_prompt,
                ref_audio=ref_audio,
                prompt_wav_path=ref_audio_path,
            )
    
            # 配置 force_listen
            if hasattr(self.duplex, '_force_listen_count'):
                self.duplex._force_listen_count = force_listen_count
    
            # 分块处理音频
            chunk_samples = sample_rate * chunk_ms // 1000
            num_chunks = (len(user_audio) + chunk_samples - 1) // chunk_samples
    
            for i in range(num_chunks):
                # 获取音频块
                start_idx = i * chunk_samples
                end_idx = min(start_idx + chunk_samples, len(user_audio))
                audio_chunk = user_audio[start_idx:end_idx]
    
                # 补零到完整块
                if len(audio_chunk) < chunk_samples:
                    audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))
    
                # 获取图像帧（如果有）
                frame_list = None
                if image_list and i < len(image_list):
                    frame_list = [image_list[i]]
    
                # 预填充
                self.duplex_prefill(
                    audio_waveform=audio_chunk,
                    frame_list=frame_list,
                )
    
                # 生成
                result = self.duplex_generate(
                    decode_mode=decode_mode,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
    
                # 记录结果
                chunk_result = {
                    "chunk_idx": i,
                    "is_listen": result.get("is_listen", True),
                    "text": result.get("text", ""),
                    "has_audio": result.get("audio") is not None,
                    "end_of_turn": result.get("end_of_turn", False),
                }
                chunks.append(chunk_result)
    
                if not chunk_result["is_listen"]:
                    full_text += chunk_result["text"]
                    if result.get("audio") is not None:
                        audio_chunks.append(result["audio"])
    
                if chunk_result["end_of_turn"]:
                    break
    
            # 停止会话
            self.duplex_stop()
    
            return {
                "success": True,
                "full_text": full_text,
                "chunks": chunks,
                "audio_chunks": audio_chunks,
                "error": None,
            }
    
        except Exception as e:
            logger.error(f"duplex_chat 失败: {e}")
            return {
                "success": False,
                "full_text": full_text,
                "chunks": chunks,
                "audio_chunks": audio_chunks,
                "error": str(e),
            }

