from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Optional

# conditional import for dict loading via URL
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────

DEFAULT_SENTENCE_GAP = 0.80
DEFAULT_CLAUSE_GAP = 0.25
DEFAULT_ALPHA_LOW = 0.60
DEFAULT_ALPHA_HIGH = 1.50
DEFAULT_WINDOW_SIZE = 5

MAX_DICT_SIZE_BYTES = 50 * 1024 * 1024
MAX_DICT_SOURCES = 20

FUNC_WORDS = {
    "的", "了", "在", "中", "和", "于", "之",
    "等", "由", "其", "被", "向", "以", "与",
    "而", "或", "但", "是", "有", "不", "也",
    "就", "这", "那", "都", "还", "很", "更",
    "将", "把", "从", "对", "为", "上", "下",
    "到", "让", "给", "用", "能", "会", "要",
}


@dataclass
class SegmentationConfig:
    sentence_gap: float = DEFAULT_SENTENCE_GAP
    clause_gap: float = DEFAULT_CLAUSE_GAP
    alpha_low: float = DEFAULT_ALPHA_LOW
    alpha_high: float = DEFAULT_ALPHA_HIGH
    window_size: int = DEFAULT_WINDOW_SIZE
    use_jieba: bool = True
    user_dict_paths: list[str] = field(default_factory=list)


# ─── Data Types ────────────────────────────────────────────


@dataclass
class WordEntry:
    char: str
    start: float
    end: float


@dataclass
class WordGroup:
    text: str
    start: float
    end: float
    chars: list[WordEntry] = field(default_factory=list)


@dataclass
class PunctuationLabel:
    position: int
    punct: str


@dataclass
class ProcessedResult:
    text: str
    word_groups: list[WordGroup]
    punctuation: list[PunctuationLabel]


# ─── jieba 增强 ────────────────────────────────────────────

_JIEBA_INITIALIZED = False


def _ensure_jieba_loaded() -> None:
    global _JIEBA_INITIALIZED
    if _JIEBA_INITIALIZED:
        return
    try:
        import jieba
    except ImportError:
        logger.warning("jieba not available, CJK post-processing disabled")
        raise

    dict_manager.initialize()
    _JIEBA_INITIALIZED = True


# ─── DictManager ────────────────────────────────────────────


@dataclass
class DictSource:
    name: str
    path: str
    entries: int
    loaded_at: str
    type: str  # "builtin" | "user_path" | "external"
    source_url: str | None = None


@dataclass
class DictLoadResult:
    status: str  # "loaded" | "error"
    name: str
    source: str
    entries_loaded: int
    entries_total: int
    duplicates_skipped: int
    load_duration_ms: int
    error: str | None = None


class DictManager:
    def __init__(self) -> None:
        self.builtin_path = Path(__file__).parent / "jieba_domain_dict.txt"
        self.cache_dir = Path.home() / ".cache" / "speaches" / "domain_dicts"
        self.sources: dict[str, DictSource] = {}
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        import jieba

        # Load built-in domain dictionary
        if self.builtin_path.exists():
            jieba.load_userdict(str(self.builtin_path))
            self.sources["_builtin"] = DictSource(
                name="_builtin",
                path=str(self.builtin_path),
                entries=_count_entries(self.builtin_path),
                loaded_at=_now_iso(),
                type="builtin",
            )
            logger.info(f"Loaded builtin domain dictionary: {self.builtin_path} ({self.sources['_builtin'].entries} entries)")
        else:
            logger.warning(f"Builtin domain dictionary not found: {self.builtin_path}")

        self._initialized = True

    def load_url(
        self,
        url: str,
        name: str | None = None,
    ) -> DictLoadResult:
        import jieba

        if httpx is None:
            return DictLoadResult(
                status="error", name=name or url, source=url,
                entries_loaded=0, entries_total=0, duplicates_skipped=0,
                load_duration_ms=0, error="httpx is not installed",
            )

        if len(self.sources) >= MAX_DICT_SOURCES:
            return DictLoadResult(
                status="error", name=name or url, source=url,
                entries_loaded=0, entries_total=0, duplicates_skipped=0,
                load_duration_ms=0, error=f"max sources ({MAX_DICT_SOURCES}) exceeded",
            )

        source_name = name or _name_from_url(url)
        if source_name in self.sources:
            _ = self.unload(source_name)

        start = time.perf_counter()
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            content = resp.text
        except Exception as e:
            return DictLoadResult(
                status="error", name=source_name, source=url,
                entries_loaded=0, entries_total=0, duplicates_skipped=0,
                load_duration_ms=int((time.perf_counter() - start) * 1000),
                error=f"download_failed: {e}",
            )

        raw_bytes = len(content.encode("utf-8"))
        if raw_bytes > MAX_DICT_SIZE_BYTES:
            return DictLoadResult(
                status="error", name=source_name, source=url,
                entries_loaded=0, entries_total=0, duplicates_skipped=0,
                load_duration_ms=int((time.perf_counter() - start) * 1000),
                error=f"dict_too_large: {raw_bytes} bytes exceeds {MAX_DICT_SIZE_BYTES}",
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = self.cache_dir / f"{source_name}.txt"
        local_path.write_text(content, encoding="utf-8")

        total_lines = _count_lines(content)
        total_entries = _count_entries_lines(content)

        try:
            jieba.load_userdict(str(local_path))
        except Exception as e:
            return DictLoadResult(
                status="error", name=source_name, source=url,
                entries_loaded=0, entries_total=total_entries, duplicates_skipped=0,
                load_duration_ms=int((time.perf_counter() - start) * 1000),
                error=f"invalid_dict_format: {e}",
            )

        elapsed = int((time.perf_counter() - start) * 1000)
        self.sources[source_name] = DictSource(
            name=source_name,
            path=str(local_path),
            entries=total_entries,
            loaded_at=_now_iso(),
            type="external",
            source_url=url,
        )
        logger.info(f"Downloaded and loaded domain dictionary '{source_name}': {total_entries} entries in {elapsed}ms")

        return DictLoadResult(
            status="loaded",
            name=source_name,
            source=url,
            entries_loaded=total_entries,
            entries_total=total_entries,
            duplicates_skipped=0,
            load_duration_ms=elapsed,
        )

    def load_user_paths(self, paths: list[str]) -> None:
        import jieba

        for path_str in paths:
            p = Path(path_str)
            if not p.exists():
                logger.warning(f"User dictionary path not found, skipping: {p}")
                continue
            name = f"_user_{p.stem}"
            if name in self.sources:
                continue
            try:
                jieba.load_userdict(str(p))
            except Exception as e:
                logger.warning(f"Failed to load user dictionary {p}: {e}")
                continue
            self.sources[name] = DictSource(
                name=name,
                path=str(p),
                entries=_count_entries(p),
                loaded_at=_now_iso(),
                type="user_path",
            )
            logger.info(f"Loaded user dictionary: {p} ({self.sources[name].entries} entries)")

    def unload(self, name: str) -> dict:
        if name == "_builtin":
            return {"status": "error", "error": "cannot unload builtin dictionary"}

        if name not in self.sources:
            return {"status": "error", "error": f"source '{name}' not found"}

        removed = self.sources.pop(name)
        self._reinitialize_all()
        return {"status": "unloaded", "name": name, "entries_removed": removed.entries}

    def _reinitialize_all(self) -> None:
        import jieba

        jieba.del_word = lambda x: None  # no-op, jieba doesn't support true removal

        # The only reliable way: reinitialize jieba from scratch
        # jieba has no official API to clear user dicts, so we reload
        jieba.initialize()  # reloads the built-in dict
        # Re-apply all remaining sources
        for source in self.sources.values():
            jieba.load_userdict(str(source.path))

    def export_merged(self) -> str:
        lines: set[str] = set()
        for source in self.sources.values():
            p = Path(source.path)
            if p.exists():
                lines.update(p.read_text(encoding="utf-8").splitlines())
        return "\n".join(sorted(lines))

    def get_sources(self) -> list[DictSource]:
        return list(self.sources.values())

    def get_total_entries(self) -> int:
        return sum(s.entries for s in self.sources.values())


# ─── Helpers ────────────────────────────────────────────────


def _count_entries(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#"))
    except Exception:
        return 0


def _count_entries_lines(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.strip() and not line.strip().startswith("#"))


def _count_lines(content: str) -> int:
    return len([ln for ln in content.splitlines() if ln.strip()])


def _name_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1].rsplit(".", 1)[0] if "/" in url else url


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


dict_manager = DictManager()


# ─── Main Entry Point ─────────────────────────────────────


def post_process_transcription(
    words: list[WordEntry],
    language: str | None,
    config: SegmentationConfig | None = None,
) -> ProcessedResult:
    if language is None or not language.startswith("zh") or len(words) < 2:
        return _passthrough(words)

    config = config or SegmentationConfig()

    punctuation = _detect_punctuation(words, config)
    punct_positions = {p.position for p in punctuation}

    if config.use_jieba:
        try:
            _ensure_jieba_loaded()
            if config.user_dict_paths:
                dict_manager.load_user_paths(config.user_dict_paths)
            jieba_groups = _jieba_segment(words)
        except ImportError:
            jieba_groups = None

        if jieba_groups is not None:
            final_groups = _acoustic_verify(words, jieba_groups, config, punct_positions)
        else:
            final_groups = _acoustic_only(words, config)
    else:
        final_groups = _acoustic_only(words, config)

    result = _build_output(words, final_groups, punctuation)
    return result


# ─── Layer 1: Gap Punctuation ─────────────────────────────


def _detect_punctuation(
    words: list[WordEntry],
    config: SegmentationConfig,
) -> list[PunctuationLabel]:
    result: list[PunctuationLabel] = []
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap >= config.sentence_gap:
            result.append(PunctuationLabel(position=i, punct="。"))
        elif gap >= config.clause_gap:
            result.append(PunctuationLabel(position=i, punct="，"))
    return result


# ─── Layer 2: jieba Segmentation ──────────────────────────


def _jieba_segment(
    words: list[WordEntry],
) -> list[tuple[int, int, str]]:
    import jieba

    text = "".join(w.char for w in words)
    positions = list(jieba.tokenize(text))

    groups: list[tuple[int, int, str]] = []
    for start_pos, end_pos, word in positions:
        groups.append((start_pos, end_pos, word))

    return groups


# ─── Layer 3: Acoustic Verification + OOV Fallback ────────


def _is_single_char(idx: int, groups: list[tuple[int, int, str]]) -> bool:
    if idx < 0 or idx >= len(groups):
        return False
    start, end, _ = groups[idx]
    return (end - start) == 1


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def _try_merge_oov_span(
    words: list[WordEntry],
    span_start_idx: int,
    span_end_idx: int,
    config: SegmentationConfig,
) -> list[tuple[int, int, str]]:
    if span_end_idx - span_start_idx < 2:
        return [(span_start_idx, span_end_idx, words[span_start_idx].char)]

    spans = words[span_start_idx:span_end_idx]
    durations = [w.end - w.start for w in spans]
    boundaries: set[int] = set()

    for i in range(1, len(durations)):
        ratio = durations[i] / durations[i - 1] if durations[i - 1] > 0 else 999.0

        half = config.window_size // 2
        left = max(0, i - half)
        right = min(len(durations), i + half + 1)
        window = [d for d in durations[left:right] if d > 0]

        if len(window) < 3:
            if ratio >= 0.8:
                boundaries.add(i)
            continue

        med = _median(window)
        if med == 0:
            continue

        if ratio < config.alpha_low:
            pass
        elif ratio > config.alpha_high:
            boundaries.add(i)
        elif durations[i] < config.alpha_low * med:
            pass
        else:
            boundaries.add(i)

    if not boundaries:
        merged_text = "".join(w.char for w in spans)
        return [(span_start_idx, span_end_idx, merged_text)]

    result: list[tuple[int, int, str]] = []
    seg_start = 0
    for b in sorted(boundaries):
        if b > seg_start:
            txt = "".join(w.char for w in spans[seg_start:b])
            result.append((span_start_idx + seg_start, span_start_idx + b, txt))
        seg_start = b
    if seg_start < len(spans):
        txt = "".join(w.char for w in spans[seg_start:])
        result.append((span_start_idx + seg_start, span_end_idx, txt))

    return result


def _acoustic_verify(
    words: list[WordEntry],
    jieba_groups: list[tuple[int, int, str]],
    config: SegmentationConfig,
    punct_positions: set[int],
) -> list[tuple[int, int, str]]:
    def spans_punct(g: tuple[int, int, str]) -> bool:
        s, e, _ = g
        return any(p in punct_positions for p in range(s, e))

    result: list[tuple[int, int, str]] = []
    i = 0
    while i < len(jieba_groups):
        start, end, word = jieba_groups[i]

        if (end - start) >= 2 or word in FUNC_WORDS or spans_punct(jieba_groups[i]):
            result.append((start, end, word))
            i += 1
        else:
            span_start = i
            while (
                i < len(jieba_groups)
                and (jieba_groups[i][1] - jieba_groups[i][0]) == 1
                and jieba_groups[i][2] not in FUNC_WORDS
                and not spans_punct(jieba_groups[i])
            ):
                i += 1
            span_end = i

            merged = _try_merge_oov_span(
                words,
                jieba_groups[span_start][0],
                jieba_groups[span_end - 1][1],
                config,
            )
            result.extend(merged)

    return result


def _acoustic_only(
    words: list[WordEntry],
    config: SegmentationConfig,
) -> list[tuple[int, int, str]]:
    groups = [(i, i + 1, words[i].char) for i in range(len(words))]
    return _acoustic_verify(words, groups, config, set())


# ─── Layer 4: Output Reconstruction ───────────────────────


def _build_output(
    words: list[WordEntry],
    final_groups: list[tuple[int, int, str]],
    punctuation: list[PunctuationLabel],
) -> ProcessedResult:
    punct_map = {p.position: p.punct for p in punctuation}

    word_groups: list[WordGroup] = []
    for start_idx, end_idx, text in final_groups:
        chars = words[start_idx:end_idx]
        if not chars:
            continue
        group = WordGroup(
            text=text,
            start=chars[0].start,
            end=chars[-1].end,
            chars=chars,
        )
        word_groups.append(group)

    for group in word_groups:
        last_idx = _find_word_index(words, group.chars[-1])
        if last_idx in punct_map:
            group.text += punct_map[last_idx]

    text_parts = [g.text for g in word_groups]
    final_text = " ".join(text_parts)

    return ProcessedResult(
        text=final_text,
        word_groups=word_groups,
        punctuation=punctuation,
    )


def _find_word_index(words: list[WordEntry], target: WordEntry) -> int:
    for i, w in enumerate(words):
        if w is target:
            return i
    return -1


def _passthrough(words: list[WordEntry]) -> ProcessedResult:
    text = "".join(w.char for w in words)
    groups = [WordGroup(text=w.char, start=w.start, end=w.end, chars=[w]) for w in words]
    return ProcessedResult(text=text, word_groups=groups, punctuation=[])


# ─── Export: apply to OpenAI TranscriptionVerbose ─────────


def apply_to_verbose_json(
    response_data: dict,
    language: str | None,
) -> dict:
    """Post-process an already-assembled verbose_json dict in-place.

    This is the integration point for whisper.py.  It modifies the
    *top-level* ``text`` and ``words`` keys and the nested segment
    ``text`` fields so that the response carries word-grouped output.

    Returns the same dict (mutated) so callers can chain it.
    """
    raw_words = response_data.get("words")
    if not raw_words:
        return response_data

    if not language or not language.startswith("zh"):
        return response_data

    entries = [WordEntry(char=w["word"], start=w["start"], end=w["end"]) for w in raw_words]

    result = post_process_transcription(entries, language)

    # Update top-level text
    response_data["text"] = result.text

    if result.word_groups:
        # Update words array: group entries
        grouped: list[dict] = []
        for g in result.word_groups:
            if not g.text:
                continue
            clean = g.text.replace("。", "").replace("，", "")
            if clean:
                grouped.append({"word": clean, "start": g.start, "end": g.end})

        if not has_punctuation_only(grouped, raw_words):
            response_data["words"] = grouped

    # Update segment texts
    segments = response_data.get("segments")
    if segments:
        _patch_segment_texts(segments, result.word_groups)

    return response_data


def has_punctuation_only(grouped: list[dict], original: list[dict]) -> bool:
    return len(grouped) == 0 and len(original) > 0


def _patch_segment_texts(
    segments: list[dict],
    word_groups: list[WordGroup],
) -> None:
    if not segments:
        return
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        parts: list[str] = []
        for g in word_groups:
            if g.start >= seg_start and g.end <= seg_end:
                parts.append(g.text)
            elif g.start >= seg_end:
                break
        if parts:
            seg["text"] = " ".join(parts)
