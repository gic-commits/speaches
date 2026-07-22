from __future__ import annotations

import asyncio
from collections import OrderedDict
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)
from pydantic import BaseModel

from speaches.audio import Audio
from speaches.dependencies import get_executor_registry
from speaches.executors.shared.handler_protocol import TranscriptionRequest, VadRequest
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.realtime.utils import generate_item_id, task_done_callback
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.types.realtime import (
    ConversationItemContentInputAudio,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemMessage,
    ServerEvent,
    Session,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from speaches.executors.shared.registry import ExecutorRegistry
    from speaches.realtime.conversation_event_router import Conversation
    from speaches.realtime.pubsub import EventPubSub

SAMPLE_RATE = 16000
MS_SAMPLE_RATE = 16
MAX_VAD_WINDOW_SIZE_SAMPLES = 3000 * MS_SAMPLE_RATE

DEFAULT_VAD_OPTIONS = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30)

CHUNK_DURATION_SAMPLES = 72000  # 4.5 seconds at 16kHz

logger = logging.getLogger(__name__)

# Limit concurrent WhisperModel.transcribe() calls to avoid CPU oversubscription.
_MAX_CONCURRENT_TRANSCRIPTIONS = 1
_transcription_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_TRANSCRIPTIONS)


def preload_transcription_model(model: str) -> None:
    """Load the transcription model into memory without running inference (avoids ONNX concurrent access during first chunk)."""
    logger.info(f"Pre-loading transcription model: {model}")
    executor_registry = get_executor_registry()
    model_card_data = get_model_card_data_or_raise(model)
    transcription_executor = find_executor_for_model_or_raise(
        model, model_card_data, executor_registry.transcription
    )
    model_wrapper = transcription_executor.model_manager.load_model(model)
    with model_wrapper:
        pass
    logger.info(f"Transcription model pre-loaded: {model}")


def _transcribe_raw_sync(
    audio_data: NDArray[np.float32],
    model: str,
    language: str | None,
) -> str:
    executor_registry = get_executor_registry()
    audio = Audio(audio_data, sample_rate=16000)

    model_card_data = get_model_card_data_or_raise(model)
    transcription_executor = find_executor_for_model_or_raise(
        model, model_card_data, executor_registry.transcription
    )

    transcription_request = TranscriptionRequest(
        audio=audio,
        model=model,
        language=language,
        response_format="text",
        temperature=0.0,
        timestamp_granularities=["segment"],
        speech_segments=[],
        vad_options=DEFAULT_VAD_OPTIONS,
    )
    result = transcription_executor.model_manager.handle_transcription_request(transcription_request)

    if isinstance(result, tuple):
        return result[0]
    return str(result)


def _transcribe_sync(
    audio_data: NDArray[np.float32],
    model: str,
    language: str | None,
) -> str:
    executor_registry = get_executor_registry()
    audio = Audio(audio_data, sample_rate=16000)

    model_card_data = get_model_card_data_or_raise(model)
    transcription_executor = find_executor_for_model_or_raise(
        model, model_card_data, executor_registry.transcription
    )

    vad_request = VadRequest(audio=audio, vad_options=DEFAULT_VAD_OPTIONS)
    speech_segments = executor_registry.vad.model_manager.handle_vad_request(vad_request)

    transcription_request = TranscriptionRequest(
        audio=audio,
        model=model,
        language=language,
        response_format="text",
        temperature=0.0,
        timestamp_granularities=["segment"],
        speech_segments=speech_segments,
        vad_options=DEFAULT_VAD_OPTIONS,
    )
    result = transcription_executor.model_manager.handle_transcription_request(transcription_request)

    if isinstance(result, tuple):
        return result[0]
    return str(result)


# NOTE not in `src/speaches/realtime/input_audio_buffer_event_router.py` due to circular import
class VadState(BaseModel):
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    # TODO: consider keeping track of what was the last audio timestamp that was processed. This value could be used to control how often the VAD is run.


# TODO: use `np.int16` instead of `np.float32` for audio data
class InputAudioBuffer:
    def __init__(self, pubsub: EventPubSub) -> None:
        self.id = generate_item_id()
        self.data: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.vad_state = VadState()
        self.pubsub = pubsub

    @property
    def size(self) -> int:
        """Number of samples in the buffer."""
        return len(self.data)

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds."""
        return len(self.data) / SAMPLE_RATE

    @property
    def duration_ms(self) -> int:
        """Duration of the audio in milliseconds."""
        return len(self.data) // MS_SAMPLE_RATE

    def append(self, audio_chunk: NDArray[np.float32]) -> None:
        """Append an audio chunk to the buffer."""
        self.data = np.append(self.data, audio_chunk)

    # def commit(self) -> None:
    #     """Publish an event to indicate that the buffer is ready for processing."""
    #     self.pubsub.publish

    # TODO: come up with a better name
    @property
    def data_w_vad_applied(self) -> NDArray[np.float32]:
        if self.vad_state.audio_start_ms is None:
            return self.data
        audio_start = self.vad_state.audio_start_ms * MS_SAMPLE_RATE
        if self.vad_state.audio_end_ms is not None:
            audio_end = self.vad_state.audio_end_ms * MS_SAMPLE_RATE
        else:
            audio_end = len(self.data)
        return self.data[audio_start:audio_end]


class InputAudioBufferManager:
    def __init__(self, pubsub: EventPubSub) -> None:
        self._pubsub = pubsub
        initial = InputAudioBuffer(pubsub)
        self._buffers: OrderedDict[str, InputAudioBuffer] = OrderedDict({initial.id: initial})

    @property
    def current(self) -> InputAudioBuffer:
        buffer_id = next(reversed(self._buffers))
        return self._buffers[buffer_id]

    def get(self, buffer_id: str) -> InputAudioBuffer:
        return self._buffers[buffer_id]

    def rotate(self) -> InputAudioBuffer:
        new_buffer = InputAudioBuffer(self._pubsub)
        self._buffers[new_buffer.id] = new_buffer
        return new_buffer

    def clear_current(self) -> InputAudioBuffer:
        self._buffers.popitem()
        return self.rotate()


class InputAudioBufferTranscriber:
    def __init__(
        self,
        *,
        pubsub: EventPubSub,
        executor_registry: ExecutorRegistry,
        input_audio_buffer: InputAudioBuffer,
        session: Session,
        conversation: Conversation,
    ) -> None:
        self.pubsub = pubsub
        self.executor_registry = executor_registry
        self.input_audio_buffer = input_audio_buffer
        self.session = session
        self.conversation = conversation

        self.task: asyncio.Task[None] | None = None
        self.events = asyncio.Queue[ServerEvent]()

    async def _keep_alive(self) -> None:
        """Send empty delta events every 2s to prevent idle timeout during long chunk processing."""
        while True:
            await asyncio.sleep(2)
            self.pubsub.publish_nowait(
                ConversationItemInputAudioTranscriptionDeltaEvent(
                    item_id=self.input_audio_buffer.id,
                    delta="",
                )
            )

    async def _handler(self) -> None:
        audio_data = self.input_audio_buffer.data_w_vad_applied.copy()
        file_size = audio_data.nbytes * 2
        logger.info(f"Transcription _handler started: model={self.session.input_audio_transcription.model}, language={self.session.input_audio_transcription.language}, input_audio_duration={self.input_audio_buffer.duration:.2f}s, audio_size={file_size}")
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        transcript_parts: list[str] = []

        async with _transcription_semaphore:
            # Only the active handler sends keep-alive deltas; queued handlers don't flood the connection
            keep_alive = asyncio.create_task(self._keep_alive())
            # Send an immediate heartbeat delta so the client knows transcription has started
            self.pubsub.publish_nowait(
                ConversationItemInputAudioTranscriptionDeltaEvent(
                    item_id=self.input_audio_buffer.id,
                    delta="",
                )
            )
            try:
                chunk_size = CHUNK_DURATION_SAMPLES
                hop_size = chunk_size - 16000 // 2  # 4s hop (0.5s overlap)
                chunk_start = 0
                while chunk_start < len(audio_data):
                    chunk_end = min(chunk_start + chunk_size, len(audio_data))
                    chunk_audio = audio_data[chunk_start:chunk_end]
                    if len(chunk_audio) >= 16000:
                        try:
                            partial = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None,
                                    _transcribe_raw_sync,
                                    chunk_audio,
                                    self.session.input_audio_transcription.model,
                                    self.session.input_audio_transcription.language,
                                ),
                                timeout=300.0,
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"Delta transcription chunk timed out")
                            partial = ""
                        except Exception as e:
                            logger.exception(f"Delta transcription chunk failed: {e}")
                            partial = ""
                        if partial.strip():
                            transcript_parts.append(partial)
                            self.pubsub.publish_nowait(
                                ConversationItemInputAudioTranscriptionDeltaEvent(
                                    item_id=self.input_audio_buffer.id,
                                    delta=partial,
                                )
                            )
                    chunk_start += hop_size
                transcript = "".join(transcript_parts)
            finally:
                keep_alive.cancel()
                try:
                    await keep_alive
                except asyncio.CancelledError:
                    pass
        elapsed = time.perf_counter() - start
        logger.info(f"Transcription done in {elapsed:.2f}s, transcript='{transcript}'")

        content_item = ConversationItemContentInputAudio(transcript=transcript, type="input_audio")
        item = ConversationItemMessage(
            id=self.input_audio_buffer.id,
            role="user",
            content=[content_item],
            status="completed",
        )
        self.conversation.create_item(item)
        self.pubsub.publish_nowait(
            ConversationItemInputAudioTranscriptionCompletedEvent(
                item_id=item.id,
                transcript=transcript,
                usage=UsageTranscriptTextUsageDuration(
                    seconds=self.input_audio_buffer.duration,
                    type="duration",
                ),
            )
        )

    # TODO: add `timeout` parameter
    def start(self) -> None:
        assert self.task is None
        self.task = asyncio.create_task(self._handler())
        self.task.add_done_callback(task_done_callback)
