from pathlib import Path

import anyio
from httpx import AsyncClient
import pytest

from speaches.executors.silero_vad_v5 import MEL_FRAME_SAMPLES, VadOptions, merge_segments
from speaches.routers.vad import MODEL_ID, SpeechTimestamp

FILE_PATH = "audio.wav"
ENDPOINT = "/v1/audio/speech/timestamps"


@pytest.mark.asyncio
async def test_speech_timestamps_basic(aclient: AsyncClient) -> None:
    extension = Path(FILE_PATH).suffix[1:]
    async with await anyio.open_file(FILE_PATH, "rb") as f:
        data = await f.read()
    res = await aclient.post(
        ENDPOINT, files={"file": (f"audio.{extension}", data, f"audio/{extension}")}, data={"model": MODEL_ID}
    )
    res.raise_for_status()
    data = res.json()
    speech_timestamps = [SpeechTimestamp.model_validate(x) for x in data]
    assert len(speech_timestamps) == 1


def test_merge_segments_aligns_to_mel_frame_boundaries() -> None:
    options = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30.0)
    segments_list = [
        SpeechTimestamp(start=5376, end=314880),  # 0.336s / 19.68s
        SpeechTimestamp(start=314880, end=326656),  # 19.68s / 20.416s
        SpeechTimestamp(start=326656, end=399600),  # 20.416s / 24.975s
    ]
    merged = merge_segments(segments_list, options)
    assert len(merged) == 1
    segment = merged[0]
    assert segment["start"] % MEL_FRAME_SAMPLES == 0
    assert segment["end"] % MEL_FRAME_SAMPLES == 0
    assert segment["start"] < segments_list[0].start
    assert segment["end"] >= segments_list[-1].end


def test_merge_segments_expands_clips_only() -> None:
    options = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30.0)
    segments_list = [
        SpeechTimestamp(start=100, end=1000),
        SpeechTimestamp(start=2000, end=3000),
    ]
    merged = merge_segments(segments_list, options)
    assert len(merged) == 1
    segment = merged[0]
    assert segment["start"] % MEL_FRAME_SAMPLES == 0
    assert segment["end"] % MEL_FRAME_SAMPLES == 0
    assert segment["start"] <= segments_list[0].start
    assert segment["end"] >= segments_list[-1].end


def test_merge_segments_clamps_end_to_audio_length() -> None:
    options = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30.0)
    audio_length_samples = 313120  # not a multiple of MEL_FRAME_SAMPLES
    segments_list = [
        SpeechTimestamp(start=0, end=audio_length_samples),
    ]
    merged = merge_segments(segments_list, options, audio_length_samples=audio_length_samples)
    assert len(merged) == 1
    segment = merged[0]
    assert segment["end"] == audio_length_samples
    assert segment["end"] <= audio_length_samples
    # without audio_length the aligned end may exceed the audio length
    merged_no_clamp = merge_segments(segments_list, options)
    assert merged_no_clamp[0]["end"] > audio_length_samples


def test_merge_segments_merges_overlapping_clips() -> None:
    options = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30.0)
    audio_length_samples = 529600
    # Adjacent segments where frame alignment pushes the first clip's end past the
    # second clip's start. Overlapping clips crash faster-whisper, so they must be
    # merged into a single contiguous clip.
    segments_list = [
        SpeechTimestamp(start=0, end=13568),
        SpeechTimestamp(start=13568, end=82176),
        SpeechTimestamp(start=87296, end=123904),
        SpeechTimestamp(start=123904, end=160512),
        SpeechTimestamp(start=160512, end=193536),
        SpeechTimestamp(start=193536, end=252416),
        SpeechTimestamp(start=252416, end=299008),
        SpeechTimestamp(start=299008, end=339200),
        SpeechTimestamp(start=339200, end=354816),
        SpeechTimestamp(start=354816, end=393472),
        SpeechTimestamp(start=393984, end=409600),
        SpeechTimestamp(start=409600, end=441344),
        SpeechTimestamp(start=441344, end=480512),
        SpeechTimestamp(start=480512, end=529600),
    ]
    merged = merge_segments(segments_list, options, audio_length_samples=audio_length_samples)
    assert len(merged) == 1
    assert merged[0]["end"] <= audio_length_samples
    # total clip span must not exceed the audio length
    total_span = sum(segment["end"] - segment["start"] for segment in merged)
    assert total_span <= audio_length_samples


def test_merge_segments_keeps_non_overlapping_clips_separate() -> None:
    options = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30.0)
    # Two segments far apart with a large gap should stay separate.
    segments_list = [
        SpeechTimestamp(start=0, end=10000),
        SpeechTimestamp(start=50000, end=60000),
    ]
    merged = merge_segments(segments_list, options, audio_length_samples=70000)
    assert len(merged) >= 1
    for i in range(len(merged) - 1):
        assert merged[i]["end"] <= merged[i + 1]["start"]


def test_min_speech_duration_ms_default_is_250() -> None:
    assert VadOptions().min_speech_duration_ms == 250


def test_min_speech_duration_filters_isolated_short_segments() -> None:
    import numpy as np

    from speaches.executors.silero_vad_v5 import SAMPLE_RATE, get_speech_timestamps

    audio = np.zeros(30000, dtype=np.float32)

    class FakeModel:
        def __init__(self, probs: np.ndarray) -> None:
            self.probs = probs

        def __call__(self, _x: np.ndarray) -> np.ndarray:
            return self.probs.reshape(1, -1)

    class FakeModelManager:
        def __init__(self, probs: np.ndarray) -> None:
            self.probs = probs

        def load_model(self, _model_id: str) -> "FakeModelManager":
            return self

        def __enter__(self) -> FakeModel:
            return FakeModel(self.probs)

        def __exit__(self, *args: object) -> None:
            return None

    def seg_count(min_speech_duration_ms: int, short_window: slice) -> int:
        probs = np.full(60, 0.05, dtype=np.float32)
        probs[5:21] = 0.9  # main speech segment (16 windows = 512ms)
        probs[short_window] = 0.9  # isolated short segment
        options = VadOptions(
            threshold=0.5,
            min_silence_duration_ms=160,
            min_speech_duration_ms=min_speech_duration_ms,
            speech_pad_ms=400,
        )
        segs = get_speech_timestamps(audio, options, FakeModelManager(probs), sampling_rate=SAMPLE_RATE)
        return len(segs)

    # short segment 192ms (< 250ms threshold): filtered at 250ms
    assert seg_count(0, slice(29, 35)) == 2
    assert seg_count(250, slice(29, 35)) == 1

    # short segment 320ms (> 250ms threshold): kept at 250ms
    assert seg_count(0, slice(29, 39)) == 2
    assert seg_count(250, slice(29, 39)) == 2
    assert seg_count(400, slice(29, 39)) == 1


# TODO: add more tests
