"""Neural network building blocks used by the MiniCPM-o model wrappers.

The public component symbols are loaded lazily to avoid import cycles during
configuration loading. In particular, ``configuration_minicpmo`` imports
``components.vision_encoder`` for ``SiglipVisionConfig`` while ``tts`` imports
``MiniCPMTTSConfig`` back from ``configuration_minicpmo``.
"""

from importlib import import_module

__all__ = [
    "MiniCPMTTS",
    "MiniCPMTTSGenerationOutput",
    "MiniCPMMLP",
    "MiniCPMWhisperEncoder",
    "MiniCPMWhisperEncoderLayer",
    "MultiModalProjector",
    "Resampler",
    "gen_logits",
    "get_2d_sincos_pos_embed",
    "make_streaming_chunk_mask_inference",
    "prepare_inputs_for_generation",
]

_SYMBOL_MODULES = {
    "MiniCPMWhisperEncoder": ".audio_encoder",
    "MiniCPMWhisperEncoderLayer": ".audio_encoder",
    "prepare_inputs_for_generation": ".generation",
    "MiniCPMMLP": ".tts",
    "MiniCPMTTS": ".tts",
    "MiniCPMTTSGenerationOutput": ".tts",
    "MultiModalProjector": ".tts",
    "gen_logits": ".tts",
    "make_streaming_chunk_mask_inference": ".tts",
    "Resampler": ".vision",
    "get_2d_sincos_pos_embed": ".vision",
}


def __getattr__(name):
    if name not in _SYMBOL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_SYMBOL_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
