"""翻译模块（懒加载）。

提供多种翻译服务：LLM、Google、Bing、DeepLX。

为避免在导入 `app.core.translate.types` 等轻量子模块时触发大量依赖导入，
此包使用 PEP 562 的 `__getattr__` 做按需加载。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseTranslator",
    "SubtitleProcessData",
    "TranslatorFactory",
    "TranslatorType",
    "TargetLanguage",
    "BingTranslator",
    "DeepLXTranslator",
    "GoogleTranslator",
    "LLMTranslator",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "BaseTranslator":
        from .base import BaseTranslator

        return BaseTranslator
    if name == "SubtitleProcessData":
        from app.core.entities import SubtitleProcessData

        return SubtitleProcessData
    if name == "TranslatorFactory":
        from .factory import TranslatorFactory

        return TranslatorFactory
    if name == "TranslatorType":
        from .types import TranslatorType

        return TranslatorType
    if name == "TargetLanguage":
        from .types import TargetLanguage

        return TargetLanguage
    if name == "BingTranslator":
        from .bing_translator import BingTranslator

        return BingTranslator
    if name == "DeepLXTranslator":
        from .deeplx_translator import DeepLXTranslator

        return DeepLXTranslator
    if name == "GoogleTranslator":
        from .google_translator import GoogleTranslator

        return GoogleTranslator
    if name == "LLMTranslator":
        from .llm_translator import LLMTranslator

        return LLMTranslator

    raise AttributeError(name)
