"""Configuration management.

Supports any OpenAI-compatible LLM provider (DeepSeek, OpenAI, Anthropic via
openai-compatible proxy, local Ollama/vLLM, etc.).

All settings are read from environment variables (via .env or shell).
Provider-specific defaults are applied automatically.

Quick reference:
    # DeepSeek (默认)
    LLM_PROVIDER=deepseek
    LLM_API_KEY=sk-xxx

    # OpenAI
    LLM_PROVIDER=openai
    LLM_API_KEY=sk-xxx

    # Anthropic (via OpenAI-compatible proxy)
    LLM_PROVIDER=custom
    LLM_API_KEY=sk-xxx
    LLM_BASE_URL=https://your-proxy.com/v1
    LLM_MODEL=claude-sonnet-4-6

    # Local Ollama / vLLM
    LLM_PROVIDER=custom
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_MODEL=qwen3:latest
    LLM_API_KEY=ollama  # Ollama doesn't need a real key, but the SDK requires one
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (if it exists)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Provider presets — one dict per known provider
# ═══════════════════════════════════════════════════════════════════════════════

PROVIDER_PRESETS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
        "flash_model": "deepseek-v4-flash",
        "supports_thinking": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1",
        "flash_model": "gpt-4.1-mini",
        "supports_thinking": False,
    },
    # Anthropic via OpenAI-compatible proxy (e.g. deepseek anthropic endpoint, litellm, etc.)
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-6",
        "flash_model": "claude-haiku-4-5",
        "supports_thinking": True,
    },
}


class Config:
    # ═══════════════════════════════════════════════════════════════════
    # LLM Provider / 模型提供商
    # ═══════════════════════════════════════════════════════════════════

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek").lower()

    # ── Main model ──
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")

    # ── Flash/cheaper model (for bulk summarization) ──
    LLM_FLASH_MODEL: str = os.getenv("LLM_FLASH_MODEL", "")

    # ═══════════════════════════════════════════════════════════════════
    # Legacy compat — if user set DeepSeek-specific vars, auto-detect
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def _apply_legacy_compat(cls):
        """If user has old DEEPSEEK_* vars set, migrate to generic vars."""
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        if deepseek_key and not cls.LLM_API_KEY:
            cls.LLM_API_KEY = deepseek_key
            cls.LLM_PROVIDER = "deepseek"
        if os.getenv("DEEPSEEK_MODEL") and not cls.LLM_MODEL:
            cls.LLM_MODEL = os.getenv("DEEPSEEK_MODEL")

    @classmethod
    def _apply_preset(cls):
        """Fill in defaults from provider preset if not explicitly set."""
        preset = PROVIDER_PRESETS.get(cls.LLM_PROVIDER, {})
        if not cls.LLM_BASE_URL:
            cls.LLM_BASE_URL = preset.get("base_url", "https://api.openai.com/v1")
        if not cls.LLM_MODEL:
            cls.LLM_MODEL = preset.get("model", "gpt-4.1")
        if not cls.LLM_FLASH_MODEL:
            cls.LLM_FLASH_MODEL = preset.get("flash_model", cls.LLM_MODEL)

    @classmethod
    def resolve(cls):
        """Call once after all env vars loaded. Applies legacy compat + presets."""
        cls._apply_legacy_compat()
        cls._apply_preset()

    # ═══════════════════════════════════════════════════════════════════
    # Kaggle / Kaggle 凭证
    # ═══════════════════════════════════════════════════════════════════

    KAGGLE_USERNAME: str = os.getenv("KAGGLE_USERNAME", "")
    KAGGLE_KEY: str = os.getenv("KAGGLE_KEY", "")

    # ═══════════════════════════════════════════════════════════════════
    # Project paths / 项目路径
    # ═══════════════════════════════════════════════════════════════════

    BASE_DIR: Path = Path(__file__).parent.parent
    COMPETITIONS_DIR: Path = BASE_DIR / "competitions"
    CHECKPOINT_DB: str = str(BASE_DIR / "checkpoints.db")

    # ═══════════════════════════════════════════════════════════════════
    # Agent tuning / Agent 参数
    # ═══════════════════════════════════════════════════════════════════

    MAX_RESEARCH_NOTEBOOKS: int = int(os.getenv("MAX_RESEARCH_NOTEBOOKS", "20"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    DEFAULT_CV_FOLDS: int = int(os.getenv("DEFAULT_CV_FOLDS", "5"))
    SCRIPT_TIMEOUT_SECONDS: int = int(os.getenv("SCRIPT_TIMEOUT_SECONDS", "600"))

    # ═══════════════════════════════════════════════════════════════════
    # Validation + Display / 验证 + 显示
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def validate(cls) -> list[str]:
        """Check required configuration. Returns list of missing items."""
        missing = []
        if not cls.LLM_API_KEY:
            missing.append("LLM_API_KEY — set in .env or environment")
        if not cls.KAGGLE_USERNAME:
            # Try reading from kaggle.json
            try:
                import json as _json
                kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
                if kaggle_json.exists():
                    data = _json.loads(kaggle_json.read_text())
                    if data.get("username"):
                        cls.KAGGLE_USERNAME = data["username"]
            except Exception:
                pass
            if not cls.KAGGLE_USERNAME:
                missing.append(
                    "Kaggle credentials — set KAGGLE_USERNAME in .env "
                    "or place kaggle.json in ~/.kaggle/"
                )
        return missing

    @classmethod
    def summary(cls) -> str:
        """One-line status for CLI display (no secrets)."""
        key_status = "✓" if cls.LLM_API_KEY else "✗"
        kaggle_status = "✓" if cls.KAGGLE_USERNAME else "✗"
        return (
            f"Provider: {cls.LLM_PROVIDER} ({cls.LLM_MODEL})  "
            f"Key: {key_status}  "
            f"Kaggle: {kaggle_status}  "
            f"Base: {cls.BASE_DIR}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton — import this everywhere
# ═══════════════════════════════════════════════════════════════════════════════

config = Config()
config.resolve()
