"""Central configuration for ResolveAI.

Every value comes from an environment variable (``.env`` is loaded once).
The module also applies the LangSmith tracing variables and reports whether
the LLM is configured; without a key the app runs in offline mode instead of
pretending to call a model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    raw = _env(name)
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    chroma_db_path: Path = field(default_factory=lambda: _path("CHROMA_DB_PATH", PROJECT_ROOT / "chroma_db"))
    knowledge_base_path: Path = field(
        default_factory=lambda: _path("KNOWLEDGE_BASE_PATH", PROJECT_ROOT / "data" / "knowledge_base")
    )
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", DEFAULT_MODEL))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    confidence_threshold: float = field(default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.65))
    max_react_rounds: int = field(default_factory=lambda: _env_int("MAX_REACT_ROUNDS", 3))
    langsmith_api_key: str = field(default_factory=lambda: _env("LANGSMITH_API_KEY"))
    langsmith_project: str = field(default_factory=lambda: _env("LANGSMITH_PROJECT", "ResolveAI"))
    langsmith_endpoint: str = field(default_factory=lambda: _env("LANGSMITH_ENDPOINT"))
    langsmith_tracing: bool = field(
        default_factory=lambda: _env_bool("LANGSMITH_TRACING", bool(_env("LANGSMITH_API_KEY")))
    )

    @property
    def llm_available(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langsmith_api_key) and self.langsmith_tracing

    def apply_tracing_env(self) -> None:
        """Hand the resolved tracing settings to LangChain/LangSmith."""
        os.environ["LANGSMITH_TRACING"] = "true" if self.langsmith_enabled else "false"
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        if self.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.top_k < 1 or self.top_k > 10:
            errors.append("TOP_K must be between 1 and 10.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0.")
        if self.max_react_rounds < 1:
            errors.append("MAX_REACT_ROUNDS must be at least 1.")
        if not self.knowledge_base_path.exists():
            errors.append(f"KNOWLEDGE_BASE_PATH does not exist: {self.knowledge_base_path}")
        if self.langsmith_tracing and not self.langsmith_api_key:
            errors.append("LANGSMITH_TRACING=true requires LANGSMITH_API_KEY.")
        return errors

    def describe(self) -> dict[str, str]:
        """Non-secret summary for the UI and logs."""
        return {
            "Answer mode": f"LLM ({self.llm_model})" if self.llm_available else "offline extractive (no ANTHROPIC_API_KEY)",
            "Retrieval": f"ChromaDB, top {self.top_k}",
            "Escalation threshold": f"{self.confidence_threshold:.2f}",
            "Tracing": f"LangSmith '{self.langsmith_project}'" if self.langsmith_enabled else "disabled",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.apply_tracing_env()
    return settings
