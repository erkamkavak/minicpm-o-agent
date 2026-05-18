"""Operational mixins for the MiniCPM-o model wrapper."""

from .duplex_proxy import DuplexProxyMixin
from .generation import ChatGenerationMixin
from .media import MediaEmbeddingMixin
from .operations import UnifiedOperationsMixin
from .speculation import SpeculativeSnapshotMixin
from .session import StreamingSessionMixin
from .streaming import StreamingGenerationMixin

__all__ = [
    "ChatGenerationMixin",
    "DuplexProxyMixin",
    "MediaEmbeddingMixin",
    "SpeculativeSnapshotMixin",
    "StreamingGenerationMixin",
    "StreamingSessionMixin",
    "UnifiedOperationsMixin",
]
