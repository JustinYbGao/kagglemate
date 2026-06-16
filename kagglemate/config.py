"""Configuration management.

All settings are read from environment variables (via .env or shell).
Sensitive defaults are empty — the app validates at startup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (if it exists)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)


class Config:
    # ── DeepSeek API ──
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    DEEPSEEK_FLASH_MODEL: str = os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")

    # ── Kaggle ──
    KAGGLE_USERNAME: str = os.getenv("KAGGLE_USERNAME", "")
    KAGGLE_KEY: str = os.getenv("KAGGLE_KEY", "")

    # ── Project paths ──
    BASE_DIR: Path = Path(__file__).parent.parent
    COMPETITIONS_DIR: Path = BASE_DIR / "competitions"
    CHECKPOINT_DB: str = str(BASE_DIR / "checkpoints.db")

    # ── Agent tuning ──
    MAX_RESEARCH_NOTEBOOKS: int = int(os.getenv("MAX_RESEARCH_NOTEBOOKS", "20"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    DEFAULT_CV_FOLDS: int = int(os.getenv("DEFAULT_CV_FOLDS", "5"))
    SCRIPT_TIMEOUT_SECONDS: int = int(os.getenv("SCRIPT_TIMEOUT_SECONDS", "600"))

    @classmethod
    def validate(cls) -> list[str]:
        """Check that required configuration is present.

        Returns a list of missing items (empty list → all good).
        """
        missing = []
        if not cls.DEEPSEEK_API_KEY:
            missing.append("DEEPSEEK_API_KEY — set in .env or environment")
        if not cls.KAGGLE_USERNAME:
            missing.append("KAGGLE_USERNAME — set in .env or ensure ~/.kaggle/kaggle.json exists")
        return missing

    @classmethod
    def summary(cls) -> str:
        """One-line summary for CLI display (no secrets)."""
        key_status = "✓" if cls.DEEPSEEK_API_KEY else "✗"
        kaggle_status = "✓" if cls.KAGGLE_USERNAME else "✗"
        return (
            f"DeepSeek: {key_status}  "
            f"Model: {cls.DEEPSEEK_MODEL}  "
            f"Kaggle: {kaggle_status}  "
            f"Base: {cls.BASE_DIR}"
        )


# Singleton — import this everywhere
config = Config()
