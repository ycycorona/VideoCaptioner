"""Headless CLI entrypoint for VideoCaptioner.

This module provides a pure-CLI interface that can run on Linux servers without GUI.

Usage examples:
  python -m app.cli full --config config.example.toml --input /path/video.mp4
  python -m app.cli transcribe --config config.example.toml --input /path/video.mp4
  python -m app.cli subtitle --config config.example.toml --input /path/subtitle.srt --video /path/video.mp4
  python -m app.cli synthesize --config config.example.toml --video /path/video.mp4 --subtitle /path/subtitle.ass
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Type, TypeVar

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    try:  # Python 3.10/3.9 fallback
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore

from app.config import MODEL_PATH, SUBTITLE_STYLE_PATH, WORK_PATH
from app.core.entities import (
    FasterWhisperModelEnum,
    SubtitleConfig,
    SubtitleLayoutEnum,
    SynthesisConfig,
    TranscribeConfig,
    TranscribeModelEnum,
    TranscribeOutputFormatEnum,
    TranslatorServiceEnum,
    VadMethodEnum,
    VideoQualityEnum,
    WhisperModelEnum,
)
from app.core.translate.types import TargetLanguage
from app.core.utils.logger import setup_logger
from app.core.utils.video_utils import video2audio

logger = setup_logger("videocaptioner_cli")

T = TypeVar("T")

DEFAULT_MEDIA_EXTS = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".m4v",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".opus",
    ".ogg",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base and return a new dict."""
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str, profile: Optional[str]) -> Dict[str, Any]:
    if tomllib is None:  # pragma: no cover
        raise RuntimeError(
            "TOML parser not available. Use Python 3.11+ (tomllib) or install tomli for Python < 3.11: `pip install tomli`."
        )
    raw = Path(config_path).expanduser().resolve()
    if not raw.exists():
        raise FileNotFoundError(f"Config file not found: {raw}")

    data = tomllib.loads(raw.read_text(encoding="utf-8"))
    profiles = data.get("profiles", {})
    if profile:
        if profile not in profiles:
            available = ", ".join(sorted(profiles.keys())) or "(none)"
            raise ValueError(
                f"Profile not found: {profile}. Available profiles: {available}"
            )
        data = _deep_merge(data, profiles[profile])
    return data


def _require_str(d: Dict[str, Any], key: str) -> str:
    val = d.get(key, None)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"Missing or invalid string config: {key}")
    return val.strip()


def _get_str(d: Dict[str, Any], key: str, default: str = "") -> str:
    val = d.get(key, None)
    if val is None:
        return default
    if not isinstance(val, str):
        raise ValueError(f"Invalid string config: {key}")
    return val.strip()


def _get_bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
    val = d.get(key, None)
    if val is None:
        return default
    if not isinstance(val, bool):
        raise ValueError(f"Invalid boolean config: {key}")
    return val


def _get_int(d: Dict[str, Any], key: str, default: int) -> int:
    val = d.get(key, None)
    if val is None:
        return default
    if not isinstance(val, int):
        raise ValueError(f"Invalid int config: {key}")
    return val


def _get_float(d: Dict[str, Any], key: str, default: float) -> float:
    val = d.get(key, None)
    if val is None:
        return default
    if not isinstance(val, (int, float)):
        raise ValueError(f"Invalid float config: {key}")
    return float(val)


E = TypeVar("E")


def _parse_enum_by_name(enum_cls: Type[E], raw: str, field: str) -> E:
    try:
        return enum_cls[raw]  # type: ignore[index]
    except Exception as e:  # pragma: no cover
        options = ", ".join([m.name for m in enum_cls])  # type: ignore[arg-type]
        raise ValueError(f"Invalid {field}: {raw}. Options: {options}") from e


def _parse_enum_by_value(enum_cls: Type[E], raw: str, field: str) -> E:
    for member in enum_cls:  # type: ignore[assignment]
        if str(getattr(member, "value", "")).lower() == raw.lower():
            return member
    options = ", ".join([str(m.value) for m in enum_cls])  # type: ignore[arg-type]
    raise ValueError(f"Invalid {field}: {raw}. Options: {options}")


def _parse_enum_by_name_or_value(enum_cls: Type[E], raw: str, field: str) -> E:
    """Parse enum member by NAME (case-insensitive) or by value string."""
    if not raw:
        raise ValueError(f"Invalid {field}: empty")
    try:
        return _parse_enum_by_name(enum_cls, raw.upper(), field)
    except Exception:
        return _parse_enum_by_value(enum_cls, raw, field)


def _normalize_exts(raw_exts: Optional[Sequence[str]]) -> set[str]:
    if not raw_exts:
        return set()
    exts: set[str] = set()
    for e in raw_exts:
        if not e:
            continue
        s = e.strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        exts.add(s)
    return exts


def _iter_files_in_dir(dir_path: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from (p for p in dir_path.rglob("*") if p.is_file())
    else:
        yield from (p for p in dir_path.iterdir() if p.is_file())


def _collect_inputs(
    inputs: Optional[Sequence[str]],
    globs: Optional[Sequence[str]],
    input_dir: Optional[str],
    recursive: bool,
    exts: set[str],
) -> list[str]:
    collected: list[Path] = []

    if inputs:
        for p in inputs:
            collected.append(Path(p).expanduser())

    if globs:
        for pattern in globs:
            for m in glob.glob(pattern, recursive=True):
                collected.append(Path(m).expanduser())

    if input_dir:
        d = Path(input_dir).expanduser()
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"--input-dir is not a directory: {d}")
        for p in _iter_files_in_dir(d, recursive=recursive):
            collected.append(p)

    if not collected:
        raise ValueError("No inputs found. Use --input/--glob/--input-dir.")

    seen: set[Path] = set()
    out: list[str] = []
    saw_dir = False
    for p in collected:
        rp = p.resolve()
        if not rp.exists():
            continue
        if rp.is_dir():
            saw_dir = True
            for child in _iter_files_in_dir(rp, recursive=recursive):
                crp = child.resolve()
                if exts and crp.suffix.lower() not in exts:
                    continue
                if crp in seen:
                    continue
                seen.add(crp)
                out.append(str(crp))
            continue
        if not rp.is_file():
            continue
        if exts and rp.suffix.lower() not in exts:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        out.append(str(rp))

    if not out:
        if saw_dir and exts:
            raise ValueError(
                "No valid input files found in directory inputs after filtering. "
                "Try adding `--recursive` or extending `--ext`."
            )
        if saw_dir:
            raise ValueError(
                "No valid input files found in directory inputs. "
                "Try adding `--recursive` or check the directory contains media files."
            )
        raise ValueError("No valid input files after filtering.")

    return sorted(out)


def _load_subtitle_style(style_name: str = "", style_file: str = "") -> str:
    if style_file:
        p = Path(style_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Subtitle style file not found: {p}")
        return p.read_text(encoding="utf-8")

    name = style_name or "default"
    style_path = SUBTITLE_STYLE_PATH / f"{name}.txt"
    if style_path.exists():
        return style_path.read_text(encoding="utf-8")
    return ""


def _apply_llm_env(llm_cfg: Dict[str, Any]) -> Tuple[str, str, str, bool]:
    base_url = _get_str(llm_cfg, "base_url", "")
    api_key = _get_str(llm_cfg, "api_key", "")
    model = _get_str(llm_cfg, "model", "")
    reflect = _get_bool(llm_cfg, "reflect", False)

    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    return base_url, api_key, model, reflect


def build_transcribe_config(cfg: Dict[str, Any]) -> TranscribeConfig:
    transcribe_cfg = cfg.get("transcribe", {})
    if not isinstance(transcribe_cfg, dict):
        raise ValueError("Config [transcribe] must be a table")

    model_name = _require_str(transcribe_cfg, "model").upper()
    transcribe_model = _parse_enum_by_name(
        TranscribeModelEnum, model_name, "transcribe.model"
    )

    language = _get_str(transcribe_cfg, "language", "auto").lower()
    if language == "auto":
        language = ""

    need_word_time_stamp = _get_bool(transcribe_cfg, "need_word_time_stamp", True)

    chunk_length_sec = transcribe_cfg.get("chunk_length_sec", None)
    if chunk_length_sec is not None and not isinstance(chunk_length_sec, int):
        raise ValueError("Invalid int config: transcribe.chunk_length_sec")
    chunk_overlap_sec = transcribe_cfg.get("chunk_overlap_sec", None)
    if chunk_overlap_sec is not None and not isinstance(chunk_overlap_sec, int):
        raise ValueError("Invalid int config: transcribe.chunk_overlap_sec")
    chunk_concurrency = transcribe_cfg.get("chunk_concurrency", None)
    if chunk_concurrency is not None and not isinstance(chunk_concurrency, int):
        raise ValueError("Invalid int config: transcribe.chunk_concurrency")

    output_format_raw = _get_str(transcribe_cfg, "output_format", "SRT").upper()
    # TranscribeOutputFormatEnum uses values like "SRT"/"All" but we prefer names here
    if output_format_raw == "ALL":
        output_format = TranscribeOutputFormatEnum.ALL
    else:
        output_format = _parse_enum_by_name(
            TranscribeOutputFormatEnum,
            output_format_raw,
            "transcribe.output_format",
        )

    whisper_cfg = transcribe_cfg.get("whisper", {})
    if whisper_cfg is None:
        whisper_cfg = {}
    if not isinstance(whisper_cfg, dict):
        raise ValueError("Config [transcribe.whisper] must be a table")

    whisper_model_raw = _get_str(whisper_cfg, "model", "")
    whisper_model = (
        _parse_enum_by_name(WhisperModelEnum, whisper_model_raw.upper(), "transcribe.whisper.model")
        if whisper_model_raw
        else None
    )

    whisper_api_cfg = transcribe_cfg.get("whisper_api", {})
    if whisper_api_cfg is None:
        whisper_api_cfg = {}
    if not isinstance(whisper_api_cfg, dict):
        raise ValueError("Config [transcribe.whisper_api] must be a table")

    faster_cfg = transcribe_cfg.get("faster_whisper", {})
    if faster_cfg is None:
        faster_cfg = {}
    if not isinstance(faster_cfg, dict):
        raise ValueError("Config [transcribe.faster_whisper] must be a table")

    faster_model_raw = _get_str(faster_cfg, "model", "")
    faster_model = (
        _parse_enum_by_name_or_value(
            FasterWhisperModelEnum,
            faster_model_raw,
            "transcribe.faster_whisper.model",
        )
        if faster_model_raw
        else None
    )
    vad_method_raw = _get_str(faster_cfg, "vad_method", "")
    vad_method = (
        _parse_enum_by_name_or_value(
            VadMethodEnum, vad_method_raw, "transcribe.faster_whisper.vad_method"
        )
        if vad_method_raw
        else None
    )

    model_dir = _get_str(faster_cfg, "model_dir", str(MODEL_PATH))

    return TranscribeConfig(
        transcribe_model=transcribe_model,
        transcribe_language=language,
        need_word_time_stamp=need_word_time_stamp,
        chunk_length_sec=chunk_length_sec,
        chunk_overlap_sec=chunk_overlap_sec,
        chunk_concurrency=chunk_concurrency,
        output_format=output_format,
        whisper_model=whisper_model,
        whisper_api_key=_get_str(whisper_api_cfg, "api_key", ""),
        whisper_api_base=_get_str(whisper_api_cfg, "base_url", ""),
        whisper_api_model=_get_str(whisper_api_cfg, "model", ""),
        whisper_api_prompt=_get_str(whisper_api_cfg, "prompt", ""),
        faster_whisper_program=_get_str(faster_cfg, "program", ""),
        faster_whisper_model=faster_model,
        faster_whisper_model_dir=model_dir,
        faster_whisper_device=_get_str(faster_cfg, "device", "cuda"),
        faster_whisper_vad_filter=_get_bool(faster_cfg, "vad_filter", True),
        faster_whisper_vad_threshold=_get_float(faster_cfg, "vad_threshold", 0.4),
        faster_whisper_vad_method=vad_method,
        faster_whisper_ff_mdx_kim2=_get_bool(faster_cfg, "ff_mdx_kim2", False),
        faster_whisper_one_word=_get_bool(faster_cfg, "one_word", True),
        faster_whisper_prompt=_get_str(faster_cfg, "prompt", ""),
    )


def build_subtitle_config(cfg: Dict[str, Any]) -> SubtitleConfig:
    subtitle_cfg = cfg.get("subtitle", {})
    if not isinstance(subtitle_cfg, dict):
        raise ValueError("Config [subtitle] must be a table")

    llm_cfg = cfg.get("llm", {})
    if llm_cfg is None:
        llm_cfg = {}
    if not isinstance(llm_cfg, dict):
        raise ValueError("Config [llm] must be a table")

    base_url, api_key, model, reflect = _apply_llm_env(llm_cfg)

    translator_raw = _get_str(subtitle_cfg, "translator_service", "")
    translator_service = (
        _parse_enum_by_name(TranslatorServiceEnum, translator_raw.upper(), "subtitle.translator_service")
        if translator_raw
        else None
    )

    target_lang_raw = _get_str(subtitle_cfg, "target_language", "")
    target_language = (
        _parse_enum_by_name(TargetLanguage, target_lang_raw.upper(), "subtitle.target_language")
        if target_lang_raw
        else None
    )

    layout_raw = _get_str(subtitle_cfg, "subtitle_layout", "TRANSLATE_ON_TOP")
    subtitle_layout = _parse_enum_by_name(
        SubtitleLayoutEnum, layout_raw.upper(), "subtitle.subtitle_layout"
    )

    style_name = _get_str(subtitle_cfg, "subtitle_style_name", "")
    style_file = _get_str(subtitle_cfg, "subtitle_style_file", "")
    style_str = _load_subtitle_style(style_name=style_name, style_file=style_file)

    return SubtitleConfig(
        base_url=base_url,
        api_key=api_key,
        llm_model=model,
        deeplx_endpoint=_get_str(subtitle_cfg, "deeplx_endpoint", ""),
        translator_service=translator_service,
        need_split=_get_bool(subtitle_cfg, "need_split", True),
        need_optimize=_get_bool(subtitle_cfg, "need_optimize", False),
        need_translate=_get_bool(subtitle_cfg, "need_translate", False),
        need_reflect=reflect,
        thread_num=_get_int(subtitle_cfg, "thread_num", 8),
        batch_size=_get_int(subtitle_cfg, "batch_size", 5),
        subtitle_layout=subtitle_layout,
        max_word_count_cjk=_get_int(subtitle_cfg, "max_word_count_cjk", 25),
        max_word_count_english=_get_int(subtitle_cfg, "max_word_count_english", 20),
        target_language=target_language,
        subtitle_style=style_str,
        custom_prompt_text=_get_str(subtitle_cfg, "custom_prompt_text", ""),
    )


def build_synthesis_config(cfg: Dict[str, Any]) -> SynthesisConfig:
    synthesis_cfg = cfg.get("synthesis", {})
    if synthesis_cfg is None:
        synthesis_cfg = {}
    if not isinstance(synthesis_cfg, dict):
        raise ValueError("Config [synthesis] must be a table")

    quality_raw = _get_str(synthesis_cfg, "video_quality", "MEDIUM")
    video_quality = _parse_enum_by_name(
        VideoQualityEnum, quality_raw.upper(), "synthesis.video_quality"
    )

    return SynthesisConfig(
        need_video=_get_bool(synthesis_cfg, "need_video", True),
        soft_subtitle=_get_bool(synthesis_cfg, "soft_subtitle", False),
        video_quality=video_quality,
    )


@dataclass(frozen=True)
class RuntimePaths:
    work_dir: Path

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "RuntimePaths":
        paths_cfg = cfg.get("paths", {})
        if paths_cfg is None:
            paths_cfg = {}
        if not isinstance(paths_cfg, dict):
            raise ValueError("Config [paths] must be a table")
        work_dir_raw = _get_str(paths_cfg, "work_dir", str(WORK_PATH))
        work_dir = Path(work_dir_raw).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        return RuntimePaths(work_dir=work_dir)


def _default_run_dir(work_dir: Path, input_path: str) -> Path:
    stem = Path(input_path).stem
    run_dir = work_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _progress_printer(prefix: str):
    def _cb(progress: int, message: str):
        # progress from various modules can be in different scales; keep it simple
        logger.info("%s %s%% %s", prefix, int(progress), message)

    return _cb


def run_transcribe(
    input_path: str,
    transcribe_config: TranscribeConfig,
    run_dir: Path,
    output_base: Optional[Path] = None,
    ensure_srt_for_next: bool = False,
) -> Path:
    from app.core.asr.transcribe import transcribe  # lazy import (no GUI dependency)

    in_path = Path(input_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    subtitle_dir = run_dir / "subtitle"
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    if output_base is None:
        output_base = subtitle_dir / in_path.stem

    logger.info(transcribe_config.print_config())
    logger.info("Extracting audio via ffmpeg...")

    # Convert to temporary wav for ASR
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_audio_path = tmp.name

    try:
        if not video2audio(str(in_path), output=temp_audio_path, audio_track_index=0):
            raise RuntimeError("Audio extraction failed (ffmpeg).")

        logger.info("Running ASR...")
        asr_data = transcribe(
            temp_audio_path,
            transcribe_config,
            callback=_progress_printer("ASR"),
        )

        # Decide export formats
        formats_to_export: list[str] = []
        fmt = transcribe_config.output_format
        if fmt is None:
            formats_to_export = ["srt"]
        elif fmt == TranscribeOutputFormatEnum.ALL:
            formats_to_export = [
                f.value.lower()
                for f in TranscribeOutputFormatEnum
                if f != TranscribeOutputFormatEnum.ALL
            ]
        else:
            formats_to_export = [fmt.value.lower()]

        if ensure_srt_for_next and "srt" not in formats_to_export:
            formats_to_export.append("srt")

        # Export
        srt_path: Optional[Path] = None
        for ext in sorted(set(formats_to_export)):
            out_path = output_base.with_suffix(f".{ext}")
            asr_data.save(str(out_path))
            logger.info("Saved %s: %s", ext.upper(), out_path)
            if ext == "srt":
                srt_path = out_path

        if srt_path is None:
            # Should not happen if ensure_srt_for_next is True or output_format includes SRT
            srt_path = output_base.with_suffix(".srt")
            asr_data.save(str(srt_path))
            logger.info("Saved SRT: %s", srt_path)

        return srt_path
    finally:
        Path(temp_audio_path).unlink(missing_ok=True)


def _ensure_llm_ready(subtitle_config: SubtitleConfig, skip_check: bool) -> None:
    from app.core.llm.check_llm import check_llm_connection

    # Only required when split/optimize/LLM translate is needed.
    if not subtitle_config.llm_model:
        raise ValueError("LLM model is required but not configured (llm.model).")
    if not subtitle_config.base_url or not subtitle_config.api_key:
        raise ValueError("LLM base_url/api_key are required but not configured (llm.base_url/llm.api_key).")

    if skip_check:
        return

    ok, msg = check_llm_connection(
        subtitle_config.base_url,
        subtitle_config.api_key,
        subtitle_config.llm_model,
    )
    if not ok:
        raise RuntimeError(f"LLM API check failed: {msg or ''}".strip())


def run_subtitle(
    subtitle_path: str,
    subtitle_config: SubtitleConfig,
    run_dir: Path,
    output_path: Optional[str] = None,
    video_path: Optional[str] = None,
    skip_llm_check: bool = False,
    export_layouts: bool = False,
) -> Path:
    from app.core.asr.asr_data import ASRData
    from app.core.optimize.optimize import SubtitleOptimizer
    from app.core.split.split import SubtitleSplitter
    from app.core.translate import (
        BingTranslator,
        DeepLXTranslator,
        GoogleTranslator,
        LLMTranslator,
    )

    in_path = Path(subtitle_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {in_path}")

    subtitle_dir = run_dir / "subtitle"
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        out_path = Path(output_path).expanduser().resolve()
    else:
        default_ext = "ass" if subtitle_config.subtitle_style else "srt"
        out_path = subtitle_dir / f"{in_path.stem}.processed.{default_ext}"

    logger.info(subtitle_config.print_config())

    asr_data = ASRData.from_subtitle_file(str(in_path))

    # 1) Optionally convert segments to word timestamps for splitting
    if subtitle_config.need_split and not asr_data.is_word_timestamp():
        logger.info("Converting subtitle to word-level segments...")
        asr_data.split_to_word_segments()
        words_path = subtitle_dir / f"{in_path.stem}.words.srt"
        asr_data.save(str(words_path))
        logger.info("Saved word-level subtitle: %s", words_path)

    # Determine if we need LLM
    needs_llm = (
        subtitle_config.need_optimize
        or asr_data.is_word_timestamp()
        or (
            subtitle_config.need_translate
            and subtitle_config.translator_service
            not in [
                TranslatorServiceEnum.DEEPLX,
                TranslatorServiceEnum.BING,
                TranslatorServiceEnum.GOOGLE,
            ]
        )
    )
    if needs_llm:
        _ensure_llm_ready(subtitle_config, skip_check=skip_llm_check)

    # 2) Split (LLM) if we have word timestamps
    if asr_data.is_word_timestamp():
        logger.info("Splitting (LLM) into sentence segments...")
        splitter = SubtitleSplitter(
            thread_num=subtitle_config.thread_num,
            model=subtitle_config.llm_model or "",
            max_word_count_cjk=subtitle_config.max_word_count_cjk,
            max_word_count_english=subtitle_config.max_word_count_english,
        )
        asr_data = splitter.split_subtitle(asr_data)
        split_path = subtitle_dir / f"{in_path.stem}.split.srt"
        asr_data.save(str(split_path))
        logger.info("Saved split subtitle: %s", split_path)

    # 3) Optimize (LLM)
    if subtitle_config.need_optimize:
        logger.info("Optimizing subtitle (LLM)...")
        optimizer = SubtitleOptimizer(
            thread_num=subtitle_config.thread_num,
            batch_num=subtitle_config.batch_size,
            model=subtitle_config.llm_model or "",
            custom_prompt=subtitle_config.custom_prompt_text or "",
            update_callback=None,
        )
        asr_data = optimizer.optimize_subtitle(asr_data)
        asr_data.remove_punctuation()
        opt_path = subtitle_dir / f"{in_path.stem}.optimized.srt"
        asr_data.save(str(opt_path))
        logger.info("Saved optimized subtitle: %s", opt_path)

    # 4) Translate
    if subtitle_config.need_translate:
        if not subtitle_config.target_language:
            raise ValueError("subtitle.target_language is required when need_translate=true")
        if not subtitle_config.translator_service:
            raise ValueError("subtitle.translator_service is required when need_translate=true")

        logger.info("Translating subtitle...")
        translator_service = subtitle_config.translator_service
        custom_prompt = subtitle_config.custom_prompt_text or ""

        if translator_service == TranslatorServiceEnum.OPENAI:
            translator = LLMTranslator(
                thread_num=subtitle_config.thread_num,
                batch_num=subtitle_config.batch_size,
                target_language=subtitle_config.target_language,
                model=subtitle_config.llm_model or "",
                custom_prompt=custom_prompt,
                is_reflect=subtitle_config.need_reflect,
                update_callback=None,
            )
        elif translator_service == TranslatorServiceEnum.GOOGLE:
            translator = GoogleTranslator(
                thread_num=subtitle_config.thread_num,
                batch_num=5,
                target_language=subtitle_config.target_language,
                timeout=20,
                update_callback=None,
            )
        elif translator_service == TranslatorServiceEnum.BING:
            translator = BingTranslator(
                thread_num=subtitle_config.thread_num,
                batch_num=10,
                target_language=subtitle_config.target_language,
                update_callback=None,
            )
        elif translator_service == TranslatorServiceEnum.DEEPLX:
            if subtitle_config.deeplx_endpoint:
                os.environ["DEEPLX_ENDPOINT"] = subtitle_config.deeplx_endpoint
            translator = DeepLXTranslator(
                thread_num=subtitle_config.thread_num,
                batch_num=5,
                target_language=subtitle_config.target_language,
                timeout=20,
                update_callback=None,
            )
        else:
            raise ValueError(f"Unsupported translator service: {translator_service}")

        asr_data = translator.translate_subtitle(asr_data)
        asr_data.remove_punctuation()
        tr_path = subtitle_dir / f"{in_path.stem}.translated.srt"
        asr_data.save(str(tr_path))
        logger.info("Saved translated subtitle: %s", tr_path)

        if export_layouts and video_path:
            vstem = Path(video_path).stem
            for layout in SubtitleLayoutEnum:
                layout_path = subtitle_dir / f"{vstem}.{layout.name.lower()}.srt"
                asr_data.save(
                    save_path=str(layout_path),
                    ass_style=subtitle_config.subtitle_style or "",
                    layout=layout,
                )
                logger.info("Saved layout subtitle: %s", layout_path)

    # 5) Save final output (SRT/ASS based on extension)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asr_data.save(
        save_path=str(out_path),
        ass_style=subtitle_config.subtitle_style or "",
        layout=subtitle_config.subtitle_layout,
    )
    logger.info("Saved processed subtitle: %s", out_path)
    return out_path


def run_synthesis(
    video_path: str,
    subtitle_path: str,
    synthesis_config: SynthesisConfig,
    run_dir: Path,
    output_path: Optional[str] = None,
) -> Optional[Path]:
    from app.core.utils.video_utils import add_subtitles

    if not synthesis_config.need_video:
        logger.info("synthesis.need_video=false, skipping video synthesis")
        return None

    vpath = Path(video_path).expanduser().resolve()
    spath = Path(subtitle_path).expanduser().resolve()
    if not vpath.exists():
        raise FileNotFoundError(f"Video not found: {vpath}")
    if not spath.exists():
        raise FileNotFoundError(f"Subtitle not found: {spath}")

    synthesis_dir = run_dir / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        out = Path(output_path).expanduser().resolve()
    else:
        out = synthesis_dir / f"{vpath.stem}.final{vpath.suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(synthesis_config.print_config())
    add_subtitles(
        str(vpath),
        str(spath),
        str(out),
        crf=synthesis_config.video_quality.get_crf(),
        preset=synthesis_config.video_quality.get_preset(),
        soft_subtitle=synthesis_config.soft_subtitle,
        progress_callback=_progress_printer("FFmpeg"),
    )
    logger.info("Saved synthesized video: %s", out)
    return out


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    parser.add_argument("--profile", default=None, help="Optional profile name under [profiles.<name>]")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="videocaptioner", description="VideoCaptioner headless CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_transcribe = sub.add_parser("transcribe", help="Transcribe a video/audio to subtitle files")
    _add_common_args(p_transcribe)
    p_transcribe.add_argument(
        "--input",
        action="append",
        help="Input video/audio file path (repeatable). Directories are allowed (treated like --input-dir).",
    )
    p_transcribe.add_argument("--glob", action="append", help="Glob pattern (repeatable, e.g. '/data/in/*.mp4')")
    p_transcribe.add_argument("--input-dir", default=None, help="Input directory for batch processing")
    p_transcribe.add_argument("--recursive", action="store_true", help="Scan --input-dir recursively")
    p_transcribe.add_argument(
        "--ext",
        action="append",
        help="File extension filter (repeatable, default in batch: common media extensions)",
    )
    p_transcribe.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining inputs if one fails",
    )
    p_transcribe.add_argument(
        "--out-dir",
        default=None,
        help="Directory to place exported subtitles (default: run_dir/subtitle)",
    )
    p_transcribe.add_argument(
        "--run-dir",
        default=None,
        help="Override run dir (single input: exact dir; batch: used as base dir)",
    )

    p_subtitle = sub.add_parser("subtitle", help="Split/optimize/translate a subtitle file")
    _add_common_args(p_subtitle)
    p_subtitle.add_argument("--input", required=True, help="Input subtitle file path (.srt/.vtt/.ass/.json)")
    p_subtitle.add_argument("--video", default=None, help="Optional video path (used for layout exports)")
    p_subtitle.add_argument("--output", default=None, help="Output subtitle path (default: run_dir/subtitle/<stem>.processed.ass)")
    p_subtitle.add_argument("--run-dir", default=None, help="Override run dir (default: work_dir/<stem>)")
    p_subtitle.add_argument("--skip-llm-check", action="store_true", help="Skip LLM connectivity check")
    p_subtitle.add_argument("--export-layouts", action="store_true", help="Export all layout SRTs (requires --video)")

    p_synth = sub.add_parser("synthesize", help="Synthesize video with subtitle (ffmpeg)")
    _add_common_args(p_synth)
    p_synth.add_argument("--video", required=True, help="Input video file")
    p_synth.add_argument("--subtitle", required=True, help="Input subtitle file (.srt/.ass)")
    p_synth.add_argument("--output", default=None, help="Output video file path")
    p_synth.add_argument("--run-dir", default=None, help="Override run dir (default: work_dir/<stem>)")

    p_full = sub.add_parser("full", help="Full pipeline: transcribe -> subtitle -> synthesize")
    _add_common_args(p_full)
    p_full.add_argument(
        "--input",
        action="append",
        help="Input video file path (repeatable). Directories are allowed (treated like --input-dir).",
    )
    p_full.add_argument("--glob", action="append", help="Glob pattern (repeatable, e.g. '/data/in/*.mp4')")
    p_full.add_argument("--input-dir", default=None, help="Input directory for batch processing")
    p_full.add_argument("--recursive", action="store_true", help="Scan --input-dir recursively")
    p_full.add_argument(
        "--ext",
        action="append",
        help="File extension filter (repeatable, default in batch: common media extensions)",
    )
    p_full.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining inputs if one fails",
    )
    p_full.add_argument("--output-video", default=None, help="Output video file path (single input only)")
    p_full.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place final videos (batch-friendly; default: run_dir/synthesis)",
    )
    p_full.add_argument(
        "--run-dir",
        default=None,
        help="Override run dir (single input: exact dir; batch: used as base dir)",
    )
    p_full.add_argument("--skip-llm-check", action="store_true", help="Skip LLM connectivity check")
    p_full.add_argument("--export-layouts", action="store_true", help="Export all layout SRTs")

    return p


def _resolve_run_dir(paths: RuntimePaths, input_path: str, override: Optional[str]) -> Path:
    return _resolve_run_dir_for_input(paths, input_path, override, batch=False)


def _resolve_run_dir_for_input(
    paths: RuntimePaths, input_path: str, override: Optional[str], batch: bool
) -> Path:
    if override:
        base = Path(override).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        if batch:
            run_dir = base / Path(input_path).stem
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        return base
    return _default_run_dir(paths.work_dir, input_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.profile)
    paths = RuntimePaths.from_cfg(cfg)

    try:
        if args.cmd == "transcribe":
            has_dir_input = any(
                Path(p).expanduser().exists() and Path(p).expanduser().is_dir()
                for p in (args.input or [])
            )
            batch_mode = bool(args.input_dir or args.glob or has_dir_input) or (args.input and len(args.input) > 1)
            exts = _normalize_exts(args.ext) if args.ext else (DEFAULT_MEDIA_EXTS if batch_mode else set())
            input_paths = _collect_inputs(
                inputs=args.input,
                globs=args.glob,
                input_dir=args.input_dir,
                recursive=bool(args.recursive),
                exts=exts,
            )

            tcfg = build_transcribe_config(cfg)
            failures = 0
            for i, input_path in enumerate(input_paths, 1):
                try:
                    run_dir = _resolve_run_dir_for_input(paths, input_path, args.run_dir, batch=len(input_paths) > 1)
                    output_base = None
                    if args.out_dir:
                        out_dir = Path(args.out_dir).expanduser().resolve()
                        out_dir.mkdir(parents=True, exist_ok=True)
                        output_base = out_dir / Path(input_path).stem

                    logger.info("(%s/%s) Transcribe: %s", i, len(input_paths), input_path)
                    run_transcribe(
                        input_path=input_path,
                        transcribe_config=tcfg,
                        run_dir=run_dir,
                        output_base=output_base,
                        ensure_srt_for_next=False,
                    )
                except Exception as e:
                    failures += 1
                    logger.exception("Transcribe failed for %s: %s", input_path, e)
                    if not args.continue_on_error:
                        raise

            return 2 if failures else 0

        if args.cmd == "subtitle":
            run_dir = _resolve_run_dir(paths, args.input, args.run_dir)
            scfg = build_subtitle_config(cfg)
            run_subtitle(
                subtitle_path=args.input,
                subtitle_config=scfg,
                run_dir=run_dir,
                output_path=args.output,
                video_path=args.video,
                skip_llm_check=args.skip_llm_check,
                export_layouts=args.export_layouts,
            )
            return 0

        if args.cmd == "synthesize":
            run_dir = _resolve_run_dir(paths, args.video, args.run_dir)
            ycfg = build_synthesis_config(cfg)
            run_synthesis(
                video_path=args.video,
                subtitle_path=args.subtitle,
                synthesis_config=ycfg,
                run_dir=run_dir,
                output_path=args.output,
            )
            return 0

        if args.cmd == "full":
            tcfg = build_transcribe_config(cfg)
            scfg = build_subtitle_config(cfg)
            ycfg = build_synthesis_config(cfg)

            has_dir_input = any(
                Path(p).expanduser().exists() and Path(p).expanduser().is_dir()
                for p in (args.input or [])
            )
            batch_mode = bool(args.input_dir or args.glob or has_dir_input) or (args.input and len(args.input) > 1)
            exts = _normalize_exts(args.ext) if args.ext else (DEFAULT_MEDIA_EXTS if batch_mode else set())
            input_paths = _collect_inputs(
                inputs=args.input,
                globs=args.glob,
                input_dir=args.input_dir,
                recursive=bool(args.recursive),
                exts=exts,
            )

            if len(input_paths) > 1 and args.output_video:
                raise ValueError("--output-video can only be used with a single input. Use --output-dir for batch.")
            if args.output_video and args.output_dir:
                raise ValueError("Cannot use --output-video and --output-dir together.")

            output_dir: Optional[Path] = None
            if args.output_dir:
                output_dir = Path(args.output_dir).expanduser().resolve()
                output_dir.mkdir(parents=True, exist_ok=True)

            failures = 0
            for i, input_path in enumerate(input_paths, 1):
                try:
                    logger.info("(%s/%s) Full pipeline: %s", i, len(input_paths), input_path)
                    run_dir = _resolve_run_dir_for_input(paths, input_path, args.run_dir, batch=len(input_paths) > 1)

                    # Ensure we have SRT output for the next stage
                    srt_path = run_transcribe(
                        input_path=input_path,
                        transcribe_config=tcfg,
                        run_dir=run_dir,
                        ensure_srt_for_next=True,
                    )

                    processed_sub_path = run_subtitle(
                        subtitle_path=str(srt_path),
                        subtitle_config=scfg,
                        run_dir=run_dir,
                        output_path=None,
                        video_path=input_path,
                        skip_llm_check=args.skip_llm_check,
                        export_layouts=args.export_layouts,
                    )

                    output_video_path: Optional[str] = None
                    if output_dir is not None:
                        v = Path(input_path)
                        output_video_path = str(output_dir / f"{v.stem}.final{v.suffix}")
                    elif len(input_paths) == 1 and args.output_video:
                        output_video_path = args.output_video

                    run_synthesis(
                        video_path=input_path,
                        subtitle_path=str(processed_sub_path),
                        synthesis_config=ycfg,
                        run_dir=run_dir,
                        output_path=output_video_path,
                    )
                except Exception as e:
                    failures += 1
                    logger.exception("Full pipeline failed for %s: %s", input_path, e)
                    if not args.continue_on_error:
                        raise

            return 2 if failures else 0

        raise ValueError(f"Unknown command: {args.cmd}")
    except Exception as e:
        logger.exception("CLI failed: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
