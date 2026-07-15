from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    status,
)

from speaches.config import Config
from speaches.model_aliases import ModelId, resolve_model_id_alias

logger = logging.getLogger(__name__)

router = APIRouter(tags=["deepgram"])

SAMPLE_RATE = 16000
DEEPGRAM_WS_CLOSE_TIMEOUT = 5.0


def get_config() -> Config:
    return Config()


async def get_config_async() -> Config:
    return get_config()


ConfigDependency = Annotated[Config, Depends(get_config_async)]


async def get_executor_registry_async():
    from speaches.dependencies import get_executor_registry
    return get_executor_registry()


ExecutorRegistryDependency = Annotated[object, Depends(get_executor_registry_async)]


def verify_deepgram_api_key(config, authorization: str | None) -> None:
    if config.api_key is None:
        return
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key required. Provide using Authorization: Token <key>",
        )
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() not in ("token", "bearer"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authorization scheme must be 'Token' or 'Bearer'",
        )
    if credentials != config.api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


def _build_word_confidence(res, words):
    import math
    import openai.types.audio
    segments = res.segments if isinstance(res, openai.types.audio.TranscriptionVerbose) and res.segments else []
    word_conf = []
    for w in words:
        seg_conf = 0.0
        for s in segments:
            if s.start is not None and s.end is not None and s.start <= w.start <= s.end:
                if s.avg_logprob is not None:
                    seg_conf = round(float(math.exp(s.avg_logprob)), 4)
                break
        word_conf.append(seg_conf)
    return word_conf


def _build_overall_confidence(res):
    import math
    import numpy as np
    import openai.types.audio
    segments = res.segments if isinstance(res, openai.types.audio.TranscriptionVerbose) and res.segments else []
    if not segments:
        return 0.0
    avg_logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
    return round(float(np.exp(np.mean(avg_logprobs))), 4) if avg_logprobs else 0.0


def _build_words(res, res_words):
    word_conf = _build_word_confidence(res, res_words)
    return [
        {
            "word": w.word,
            "start": round(w.start, 3),
            "end": round(w.end, 3),
            "confidence": word_conf[i],
            "punctuated_word": w.word,
        }
        for i, w in enumerate(res_words)
    ]


def _build_deepgram_metadata(
    request_id: str, duration: float, audio_sha256: str | None = None
) -> dict:
    model_uuid = "00000000-0000-0000-0000-000000000001"
    return {
        "transaction_key": "deprecated",
        "request_id": request_id,
        "sha256": audio_sha256 or "",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "duration": round(duration, 3),
        "channels": 1,
        "models": [model_uuid],
        "model_info": {
            model_uuid: {
                "name": "faster-whisper",
                "version": "latest",
                "arch": "whisper",
            }
        },
    }


def _build_deepgram_response(
    res,
    duration: float,
    request_id: str,
    audio_sha256: str | None = None,
    detected_language: str | None = None,
) -> dict:
    import openai.types.audio

    transcript = res.text.strip() if res.text else ""

    res_words = res.words if isinstance(res, openai.types.audio.TranscriptionVerbose) and res.words else []
    words = _build_words(res, res_words)
    confidence = _build_overall_confidence(res)

    return {
        "metadata": _build_deepgram_metadata(request_id, duration, audio_sha256),
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": transcript,
                            "confidence": confidence,
                            "words": words,
                        }
                    ],
                    "detected_language": detected_language,
                }
            ]
        },
    }


async def _decode_audio_bytes(raw_bytes: bytes) -> Audio:
    import numpy as np
    import soundfile as sf

    from speaches.audio import Audio

    audio_data, audio_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)
    if audio_sr != SAMPLE_RATE:
        from speaches.audio import resample_audio_data
        audio_data = resample_audio_data(audio_data, audio_sr, SAMPLE_RATE)
    return Audio(audio_data, sample_rate=SAMPLE_RATE)


async def _decode_audio_from_request(request: Request) -> Audio:
    raw_bytes = await request.body()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty request body")

    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        try:
            body = json.loads(raw_bytes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        url = body.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="Missing 'url' field in JSON body")
        import httpx
        try:
            resp = await httpx.AsyncClient(timeout=300).get(url)
            resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch audio from URL: {e}") from e
        raw_bytes = resp.content

    elif content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
            for _, field in form.items():
                if hasattr(field, "read"):
                    file_bytes = await field.read()
                    if file_bytes:
                        raw_bytes = file_bytes
                    break
        except Exception:
            pass

    try:
        return await _decode_audio_bytes(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Failed to decode audio: {e}") from e


def _transcribe(
    audio: Audio,
    model: str,
    language: str | None,
    executor_registry,
):
    from speaches.executors.shared.handler_protocol import (
        TranscriptionRequest,
        VadRequest,
    )
    from speaches.routers.stt import DEFAULT_VAD_OPTIONS
    from speaches.routers.utils import (
        find_executor_for_model_or_raise,
        get_model_card_data_or_raise,
    )

    import openai.types.audio

    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.transcription)

    vad_request = VadRequest(audio=audio, vad_options=DEFAULT_VAD_OPTIONS)
    speech_segments = executor_registry.vad.model_manager.handle_vad_request(vad_request)

    transcription_request = TranscriptionRequest(
        audio=audio,
        model=model,
        language=language,
        prompt=None,
        response_format="verbose_json",
        temperature=0.0,
        timestamp_granularities=["word"],
        stream=False,
        hotwords=None,
        speech_segments=speech_segments,
        vad_options=DEFAULT_VAD_OPTIONS,
        without_timestamps=False,
    )
    res = executor.model_manager.handle_non_streaming_transcription_request(transcription_request)
    if not isinstance(res, (openai.types.audio.TranscriptionVerbose, openai.types.audio.Transcription)):
        raise HTTPException(status_code=500, detail="Unexpected transcription response type")
    return res  # pyrefly: ignore[bad-return]


@router.get("/v1/projects")
async def deepgram_projects(request: Request) -> dict:
    config = get_config()
    verify_deepgram_api_key(config, request.headers.get("authorization"))
    return {
        "projects": [
            {
                "project_id": "speaches",
                "name": "speaches",
            }
        ]
    }


@router.post("/v1/listen")
async def deepgram_listen_http(
    request: Request,
    executor_registry: ExecutorRegistryDependency,
    model: ModelId = Query("whisper-1"),
    language: str | None = Query(None),
    encoding: str | None = Query(None),
    sample_rate: int | None = Query(None),
    punctuate: bool = Query(False),
    smart_format: bool = Query(False),
    diarize: bool = Query(False),
    utterances: bool = Query(False),
    paragraphs: bool = Query(False),
    numerals: bool = Query(False),
    profanity_filter: bool = Query(False),
    multichannel: bool = Query(False),
    dictation: bool = Query(False),
    filler_words: bool = Query(False),
    detect_language: bool = Query(False),
) -> dict:
    import hashlib

    model = resolve_model_id_alias(model)
    config = get_config()
    verify_deepgram_api_key(config, request.headers.get("authorization"))

    audio = await _decode_audio_from_request(request)
    audio_sha256 = hashlib.sha256(audio.data.tobytes()).hexdigest()
    res = _transcribe(audio, model, language, executor_registry)

    detected_language = res.language if hasattr(res, "language") else None
    duration = res.duration if hasattr(res, "duration") and res.duration is not None else audio.duration
    request_id = str(uuid4())
    return _build_deepgram_response(res, duration, request_id, audio_sha256, detected_language)


def _decode_ws_audio_frame(
    raw_bytes: bytes, encoding: str, sample_rate: int
):
    import numpy as np

    if encoding == "linear16":
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
    else:
        import soundfile as sf
        try:
            audio_float32, _ = sf.read(
                io.BytesIO(raw_bytes), dtype="float32", channels=1, samplerate=sample_rate
            )
        except Exception:
            logger.warning(f"Failed to decode audio with encoding {encoding}")
            return None
    if sample_rate != SAMPLE_RATE:
        from speaches.audio import resample_audio_bytes
        audio_int16 = (audio_float32 * 32767).astype(np.int16)
        resampled = resample_audio_bytes(audio_int16.tobytes(), sample_rate, SAMPLE_RATE)
        audio_float32 = np.frombuffer(resampled, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_float32


def _transcribe_audio_ws(
    audio_data,
    model: str,
    language: str | None,
    executor_registry,
):
    import numpy as np
    import openai.types.audio

    from speaches.audio import Audio
    audio = Audio(audio_data, sample_rate=SAMPLE_RATE)
    res = _transcribe(audio, model, language, executor_registry)
    transcript = res.text.strip() if res.text else ""

    res_words = res.words if isinstance(res, openai.types.audio.TranscriptionVerbose) and res.words else []
    words = _build_words(res, res_words)
    confidence = _build_overall_confidence(res)
    duration = res.duration if hasattr(res, "duration") and res.duration is not None else len(audio_data) / SAMPLE_RATE
    return transcript, confidence, words, duration


@router.websocket("/v1/listen")
async def deepgram_listen_ws(
    websocket: WebSocket,
    executor_registry: ExecutorRegistryDependency,
    config: ConfigDependency,
    model: ModelId = Query("whisper-1"),
    language: str | None = Query(None),
    encoding: str = Query("linear16"),
    sample_rate: int = Query(16000),
    punctuate: bool = Query(False),
    interim_results: bool = Query(False),
    utterance_end_ms: str = Query("1000"),
    vad_turnoff: bool = Query(False),
    smart_format: bool = Query(False),
    diarize: bool = Query(False),
    multichannel: bool = Query(False),
    numerals: bool = Query(False),
    profanity_filter: bool = Query(False),
    detect_entities: bool = Query(False),
    endpointing: str = Query("10"),
    channels: str = Query("1"),
    vad_events: bool = Query(False),
) -> None:
    model = resolve_model_id_alias(model)
    await websocket.accept()

    request_id = str(uuid4())
    all_audio_chunks = []

    await websocket.send_text(json.dumps({
        "type": "Metadata",
        "transaction_key": "deprecated",
        "request_id": request_id,
        "sha256": "",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "duration": 0.0,
        "channels": 1,
    }))

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=DEEPGRAM_WS_CLOSE_TIMEOUT)
            except TimeoutError:
                if all_audio_chunks:
                    break
                continue

            if isinstance(message, dict) and message.get("type") == "websocket.disconnect":
                break

            if isinstance(message, dict) and message.get("type") == "websocket.receive":
                data = message
                if "text" in data:
                    text_data = data["text"]
                    try:
                        msg = json.loads(text_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON message: {text_data}")
                        continue
                    if msg.get("type") in ("CloseStream", "Finalize"):
                        break
                    continue

                if "bytes" in data:
                    raw_bytes: bytes = data["bytes"]
                    audio_float32 = _decode_ws_audio_frame(raw_bytes, encoding, sample_rate)
                    if audio_float32 is not None:
                        all_audio_chunks.append(audio_float32)

    except Exception:
        logger.exception("WebSocket error")
    finally:
        if all_audio_chunks:
            import numpy as np
            accumulated = np.concatenate(all_audio_chunks)
            try:
                transcript, confidence, words, duration = _transcribe_audio_ws(
                    accumulated, model, language, executor_registry
                )
                model_uuid = "00000000-0000-0000-0000-000000000001"
                result_msg = {
                    "type": "Results",
                    "channel_index": [0, 1],
                    "duration": round(duration, 3),
                    "start": 0.0,
                    "is_final": True,
                    "speech_final": True,
                    "channel": {
                        "alternatives": [{"transcript": transcript, "confidence": confidence, "words": words}],
                    },
                    "metadata": {
                        "request_id": request_id,
                        "model_uuid": model_uuid,
                        "model_info": {
                            "name": "faster-whisper",
                            "version": "latest",
                            "arch": "whisper",
                        },
                    },
                }
                await websocket.send_text(json.dumps(result_msg))
            except Exception:
                logger.exception("Final transcription failed")

        try:
            await websocket.close()
        except Exception:
            pass
