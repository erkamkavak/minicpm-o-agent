"""Neural network building blocks used by the MiniCPM-o model wrappers."""

from .audio_encoder import MiniCPMWhisperEncoder, MiniCPMWhisperEncoderLayer
from .generation import prepare_inputs_for_generation
from .tts import MiniCPMMLP
from .tts import MiniCPMTTS
from .tts import MiniCPMTTSGenerationOutput
from .tts import MultiModalProjector
from .tts import gen_logits
from .tts import make_streaming_chunk_mask_inference
from .vision import Resampler, get_2d_sincos_pos_embed

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
