"""
Application Configuration Module
Loads environment variables using python-dotenv and provides configuration
classes for Flask and MySQL database settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Config:
    """Base application configuration."""

    # Project Paths
    BASE_DIR = BASE_DIR

    # Flask Core Settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-secret-key-change-me")
    DEBUG = False
    TESTING = False

    # Session & Security Settings (Phase 5)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "False").lower() in ("true", "1", "yes")
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = int(os.environ.get("WTF_CSRF_TIME_LIMIT", 3600))  # 1 hour token validity

    # MySQL Database Connection Settings
    DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.environ.get("DB_PORT", 3306))
    DB_NAME = os.environ.get("DB_NAME", "plant_disease_db")
    DB_USER = os.environ.get("DB_USER", "plant_app")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    DB_POOL_NAME = os.environ.get("DB_POOL_NAME", "plant_db_pool")
    DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", 5))

    # Prediction & Upload Settings (Phase 2)
    UPLOAD_FOLDER = BASE_DIR / "frontend" / "static" / "uploads"
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 5 * 1024 * 1024))  # 5 MB max upload size
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_IMAGE_DIMENSION = 5000  # Max width or height in pixels to avoid decompression bombs
    MAX_IMAGE_PIXELS = 25_000_000  # Max total pixels (5000 x 5000)

    # Dataset Paths (Phase 2 & 3)
    DATASET_DIR = BASE_DIR / "dataset"
    REPORTS_DIR = BASE_DIR / "docs"

    # Model Artifacts (Phases 3, 4 & 8 Multi-Dataset Pipeline)
    MODEL_DIR = BASE_DIR / "models"
    MODELS_DIR = BASE_DIR / "models"
    MODEL_PATH = Path(os.environ.get(
        "MODEL_PATH",
        str(MODELS_DIR / "best_model.keras" if (MODELS_DIR / "best_model.keras").exists() else MODEL_DIR / "saved_model" / "plant_disease_model.keras")
    ))
    CLASS_INDICES_PATH = Path(os.environ.get(
        "CLASS_INDICES_PATH",
        str(MODELS_DIR / "class_names.json" if (MODELS_DIR / "class_names.json").exists() else MODEL_DIR / "class_indices.json")
    ))
    MODEL_METADATA_PATH = Path(os.environ.get(
        "MODEL_METADATA_PATH",
        str(MODELS_DIR / "model_metadata.json" if (MODELS_DIR / "model_metadata.json").exists() else MODEL_DIR / "model_metadata.json")
    ))
    DISEASE_MAPPING_PATH = Path(os.environ.get("DISEASE_MAPPING_PATH", str(MODEL_DIR / "disease_class_mapping.json")))

    # Gemini API Settings (Supplementary Multimodal Plant & Symptom Analysis)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    GEMINI_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", 10))
    GEMINI_ENABLED = os.environ.get("GEMINI_ENABLED", "True").lower() in ("true", "1", "yes")

    @classmethod
    def get_db_config(cls) -> dict:
        """Returns database connection parameters as a dictionary."""
        return {
            "host": cls.DB_HOST,
            "port": cls.DB_PORT,
            "database": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "pool_name": cls.DB_POOL_NAME,
            "pool_size": cls.DB_POOL_SIZE,
            "charset": "utf8mb4",
            "collation": "utf8mb4_unicode_ci",
            "autocommit": False,
        }

    @classmethod
    def validate_db_config(cls) -> list[str]:
        """
        Validates essential database configuration variables.
        Returns a list of error messages (empty if valid).
        Does NOT expose sensitive passwords in returned errors.
        """
        errors = []
        if not cls.DB_HOST:
            errors.append("DB_HOST is required but not configured.")
        if not cls.DB_PORT:
            errors.append("DB_PORT is required but not configured.")
        if not cls.DB_NAME:
            errors.append("DB_NAME is required but not configured.")
        if not cls.DB_USER:
            errors.append("DB_USER is required but not configured.")
        if not cls.DB_PASSWORD or cls.DB_PASSWORD == "replace_with_your_password":
            errors.append(
                "DB_PASSWORD is missing or set to the default placeholder. "
                "Please configure valid credentials in your .env file."
            )
        return errors


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t", "yes")


class TestingConfig(Config):
    """Testing environment configuration with mock or isolated database."""
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    DB_NAME = os.environ.get("TEST_DB_NAME", "plant_disease_test_db")
    DB_POOL_NAME = "plant_test_pool"
    DB_POOL_SIZE = 2


class ProductionConfig(Config):
    """Production environment configuration with strict validations."""
    DEBUG = False
    TESTING = False


# Map of environment names to config classes
CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(env_name: str | None = None) -> type[Config]:
    """Retrieve configuration class based on environment variable or explicit name."""
    if not env_name:
        env_name = os.environ.get("FLASK_ENV", "development").lower()
    return CONFIG_MAP.get(env_name, DevelopmentConfig)
