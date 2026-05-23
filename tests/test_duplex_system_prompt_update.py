from types import SimpleNamespace
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from minicpmo_demo.model.capabilities.duplex import DuplexCapability
from minicpmo_demo.model.runtime.stream_decoder import StreamDecoder


class FakeTokenizer:
    eos_token_id = 0
    unk_token_id = -1
    all_special_ids = []
    all_special_tokens = []

    def convert_tokens_to_ids(self, token):
        return {
            "<|chunk_eos|>": 1,
            "<|chunk_tts_eos|>": 2,
            "<|turn_eos|>": 3,
            "<|speak|>": 4,
        }.get(token, 99)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 for c in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(str(i) for i in token_ids)


class FakeEmbeddingModel:
    def __init__(self, hidden_size):
        self.embedding = torch.nn.Embedding(256, hidden_size)
        with torch.no_grad():
            weight = torch.arange(256 * hidden_size, dtype=torch.float32).reshape(256, hidden_size)
            self.embedding.weight.copy_(weight / 1000.0)

    def embed_tokens(self, token_ids):
        return self.embedding(token_ids)


class FakeLLM:
    device = torch.device("cpu")

    def __init__(self, hidden_size=4):
        self.config = SimpleNamespace(hidden_size=hidden_size, rope_theta=10000.0)
        self.model = FakeEmbeddingModel(hidden_size)

    def __call__(
        self,
        inputs_embeds,
        position_ids,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del use_cache, return_dict
        positions = position_ids.to(inputs_embeds.dtype).unsqueeze(-1)
        keys = inputs_embeds.unsqueeze(1) + positions
        values = inputs_embeds.unsqueeze(1) - positions
        current = ((keys, values),)
        if past_key_values is None:
            cache = current
        else:
            cache = (
                (
                    torch.cat([past_key_values[0][0], current[0][0]], dim=2),
                    torch.cat([past_key_values[0][1], current[0][1]], dim=2),
                ),
            )
        return SimpleNamespace(past_key_values=cache)


def _make_cache(length, hidden_size=4):
    values = torch.arange(length * hidden_size, dtype=torch.float32).reshape(1, 1, length, hidden_size)
    keys = values + 0.5
    return ((keys, values),)


def test_update_system_prompt_rebuilds_system_span_and_preserves_units():
    decoder = StreamDecoder(FakeLLM(), FakeTokenizer())
    decoder.cache = _make_cache(6)
    decoder._preserve_prefix_length = 2
    decoder._previous_token_ids = [30]
    decoder._previous_content_length = 1
    decoder._suffix_token_ids = [40]
    decoder._system_preserve_length = 4
    decoder._unit_history = [{"unit_id": 0, "length": 2, "type": "audio"}]

    old_unit_values = decoder.cache[0][1][:, :, 4:6, :].clone()
    ref_audio_embeds = torch.ones(2, 4)

    updated = decoder.update_system_prompt(
        new_prefix_token_ids=[10, 11, 12],
        new_suffix_token_ids=[50],
        new_ref_audio_embeds=ref_audio_embeds,
    )

    assert updated is True
    assert decoder.get_cache_length() == 9
    assert decoder._preserve_prefix_length == 5
    assert decoder._previous_content_length == 1
    assert decoder._suffix_token_ids == [50]
    assert decoder._system_preserve_length == 7
    assert torch.equal(decoder.cache[0][1][:, :, 7:9, :], old_unit_values)


def test_update_system_prompt_supports_plain_protected_cache_layout():
    decoder = StreamDecoder(FakeLLM(), FakeTokenizer())
    decoder.cache = _make_cache(5)
    decoder._system_preserve_length = 3
    decoder._unit_history = [{"unit_id": 0, "length": 2, "type": "audio"}]

    old_unit_values = decoder.cache[0][1][:, :, 3:5, :].clone()

    updated = decoder.update_system_prompt(
        new_prefix_token_ids=[10],
        new_suffix_token_ids=[20],
    )

    assert updated is True
    assert decoder.get_cache_length() == 4
    assert decoder._preserve_prefix_length == 1
    assert decoder._system_preserve_length == 2
    assert torch.equal(decoder.cache[0][1][:, :, 2:4, :], old_unit_values)


class FakeDuplexDecoder:
    def __init__(self):
        self._window_config = SimpleNamespace(sliding_window_mode="off")
        self.updated_ref_audio_embeds = None
        self.feed_lengths = []

    def reset(self):
        self.feed_lengths.clear()

    def embed_tokens(self, token_ids):
        return torch.ones(len(token_ids), 4)

    def feed(self, embeds):
        self.feed_lengths.append(int(embeds.shape[0]))

    def register_system_prompt(self):
        pass

    def get_cache_length(self):
        return 1

    def update_system_prompt(
        self,
        new_prefix_token_ids,
        new_suffix_token_ids,
        new_ref_audio_embeds=None,
    ):
        self.updated_ref_audio_embeds = new_ref_audio_embeds
        return True


class FakeDuplexTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1] * len(text)


class FakeDuplexProcessor:
    def __init__(self):
        self.audio_batches = []

    def process_audio(self, audio_batch):
        self.audio_batches.append(audio_batch)
        return audio_batch


class FakeDuplexModel:
    def __init__(self):
        self.config = SimpleNamespace(audio_chunk_length=1)

    def init_streaming_processor(self):
        pass

    def get_audio_embedding(self, data, chunk_length):
        del data, chunk_length
        return [[torch.ones(3, 4)]]


def test_duplex_prepare_remembers_ref_audio_for_plain_window_update():
    capability = DuplexCapability.__new__(DuplexCapability)
    capability.break_event = threading.Event()
    capability.session_stop_event = threading.Event()
    capability.decoder = FakeDuplexDecoder()
    capability.tokenizer = FakeDuplexTokenizer()
    capability.processor = FakeDuplexProcessor()
    capability.model = FakeDuplexModel()
    capability.generate_audio = False
    capability.token2wav_initialized = False
    capability._pending_finalize = None

    ref_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    capability.prepare(
        prefix_system_prompt="old",
        suffix_system_prompt="suffix",
        ref_audio=ref_audio,
    )

    assert capability._ref_audio is ref_audio
    assert capability.update_system_prompt(prefix_system_prompt="new") is True
    assert capability.processor.audio_batches[-1][0] is ref_audio
    assert capability.decoder.updated_ref_audio_embeds is not None
    assert capability.decoder.updated_ref_audio_embeds.shape == (3, 4)
