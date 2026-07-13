"""Unit tests for Deepgram router core logic.

Tests pure functions directly without needing heavy ML dependencies,
thanks to the lazy imports in deepgram.py.
"""

import pytest


class TestDeepgramApiKeyVerification:
    def test_no_auth_when_no_api_key(self):
        from speaches.routers.deepgram import verify_deepgram_api_key

        config = type("Config", (), {"api_key": None})()
        verify_deepgram_api_key(config, None)
        verify_deepgram_api_key(config, "Token some-key")

    def test_missing_auth_returns_403(self):
        from fastapi import HTTPException
        from pydantic import SecretStr

        from speaches.routers.deepgram import verify_deepgram_api_key

        config = type("Config", (), {"api_key": SecretStr("test-key")})()
        with pytest.raises(HTTPException) as exc_info:
            verify_deepgram_api_key(config, None)
        assert exc_info.value.status_code == 403

    def test_wrong_scheme_returns_403(self):
        from fastapi import HTTPException
        from pydantic import SecretStr

        from speaches.routers.deepgram import verify_deepgram_api_key

        config = type("Config", (), {"api_key": SecretStr("test-key")})()
        with pytest.raises(HTTPException) as exc_info:
            verify_deepgram_api_key(config, "Basic dGVzdDp0ZXN0")
        assert exc_info.value.status_code == 403

    def test_token_scheme_with_correct_key(self):
        from pydantic import SecretStr

        from speaches.routers.deepgram import verify_deepgram_api_key

        config = type("Config", (), {"api_key": SecretStr("test-key")})()
        verify_deepgram_api_key(config, "Token test-key")

    def test_bearer_scheme_with_correct_key(self):
        from pydantic import SecretStr

        from speaches.routers.deepgram import verify_deepgram_api_key

        config = type("Config", (), {"api_key": SecretStr("test-key")})()
        verify_deepgram_api_key(config, "Bearer test-key")

    def test_wrong_key_returns_403(self):
        from fastapi import HTTPException
        from pydantic import SecretStr

        from speaches.routers.deepgram import verify_deepgram_api_key

        config = type("Config", (), {"api_key": SecretStr("test-key")})()
        with pytest.raises(HTTPException) as exc_info:
            verify_deepgram_api_key(config, "Token wrong-key")
        assert exc_info.value.status_code == 403


class TestDeepgramResponseBuilder:
    def test_basic_transcription_structure(self):
        import openai.types.audio

        from speaches.routers.deepgram import _build_deepgram_response

        res = openai.types.audio.Transcription(text="Hello world")
        result = _build_deepgram_response(res, 2.5, "test-id", "en")

        assert result["metadata"]["request_id"] == "test-id"
        assert result["metadata"]["duration"] == 2.5
        assert result["metadata"]["channels"] == 1
        alt = result["results"]["channels"][0]["alternatives"][0]
        assert alt["transcript"] == "Hello world"
        assert alt["words"] == []

    def test_transcription_with_words(self):
        import openai.types.audio

        from speaches.routers.deepgram import _build_deepgram_response

        res = openai.types.audio.TranscriptionVerbose(
            text="Hello world",
            language="en",
            duration=2.5,
            segments=[
                openai.types.audio.TranscriptionSegment(
                    id=0, seek=0, start=0.0, end=1.0, text="Hello world",
                    tokens=[], temperature=0, avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0,
                )
            ],
            words=[
                openai.types.audio.TranscriptionWord(word="Hello", start=0.0, end=0.4),
                openai.types.audio.TranscriptionWord(word="world", start=0.5, end=1.0),
            ],
        )
        result = _build_deepgram_response(res, 2.5, "test-id", "en")
        alt = result["results"]["channels"][0]["alternatives"][0]
        assert alt["transcript"] == "Hello world"
        assert len(alt["words"]) == 2
        assert alt["confidence"] > 0

    def test_confidence_from_avg_logprob(self):
        import math

        import openai.types.audio

        from speaches.routers.deepgram import _build_deepgram_response

        res = openai.types.audio.TranscriptionVerbose(
            text="Test",
            language="en",
            duration=1.0,
            segments=[
                openai.types.audio.TranscriptionSegment(
                    id=0, seek=0, start=0.0, end=0.5, text="Test",
                    tokens=[], temperature=0, avg_logprob=-0.2, compression_ratio=1.0, no_speech_prob=0.0,
                )
            ],
            words=[],
        )
        result = _build_deepgram_response(res, 1.0, "test-id", "en")
        expected = round(math.exp(-0.2), 4)
        assert result["results"]["channels"][0]["alternatives"][0]["confidence"] == expected


class TestDeepgramAudioDecoding:
    def test_linear16_decode(self):
        import numpy as np

        from speaches.routers.deepgram import _decode_ws_audio_frame

        samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        result = _decode_ws_audio_frame(samples.tobytes(), "linear16", 16000)
        assert result is not None
        assert len(result) == 5
        assert abs(result[0]) < 0.001
        assert abs(result[3] - 1.0) < 0.001
        assert abs(result[4] + 1.0) < 0.001
