"""Тесты конфигурации LLM runtime через env."""

from __future__ import annotations

import pytest

from mcd_voice.dialog.pipeline import DialogPipeline
from mcd_voice.llm.agent import (
    _normalize_base_url,
    _resolve_model,
    ensure_llm_credentials,
    get_llm_runtime_config,
)


def test_normalize_base_url_keeps_v1_root() -> None:
    assert _normalize_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_normalize_base_url_strips_chat_completions() -> None:
    assert (
        _normalize_base_url("http://localhost:11434/v1/chat/completions")
        == "http://localhost:11434/v1"
    )


def test_resolve_model_prefers_explicit(monkeypatch) -> None:
    monkeypatch.setenv("API_MODEL", "env-model")
    assert _resolve_model("explicit-model") == "explicit-model"


def test_resolve_model_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("API_MODEL", "qwen3:1.7b")
    assert _resolve_model(None) == "qwen3:1.7b"


def test_runtime_config_for_openai(monkeypatch) -> None:
    monkeypatch.delenv("API_PROVIDER", raising=False)
    monkeypatch.delenv("API_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    cfg = get_llm_runtime_config()
    assert cfg["provider"] == "openai"
    assert cfg["model"]
    assert cfg["base_url"] == ""


def test_runtime_config_for_ollama(monkeypatch) -> None:
    monkeypatch.setenv("API_PROVIDER", "ollama")
    monkeypatch.setenv("API_MODEL", "qwen3:1.7b")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")

    cfg = get_llm_runtime_config()
    assert cfg == {
        "provider": "ollama",
        "model": "qwen3:1.7b",
        "base_url": "http://localhost:11434/v1",
    }


def test_dialog_pipeline_uses_env_model(monkeypatch) -> None:
    monkeypatch.setenv("API_MODEL", "qwen3:1.7b")

    pipeline = DialogPipeline()

    assert pipeline.model == "qwen3:1.7b"


def test_ensure_llm_credentials_requires_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("API_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        ensure_llm_credentials()


def test_ensure_llm_credentials_ollama_requires_url(monkeypatch) -> None:
    monkeypatch.setenv("API_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        ensure_llm_credentials()


def test_ensure_llm_credentials_neutral_llm_env(monkeypatch) -> None:
    monkeypatch.delenv("API_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    ensure_llm_credentials()
