"""
Application configuration management.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    PROJECT_NAME: str = "Chat Bot API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    # IMPORTANT: Use a stable secret key so JWTs remain valid across restarts.
    # Loaded from environment (.env) if provided; otherwise uses a fixed dev default.
    # Replace the default in production or set SECRET_KEY in your environment.
    SECRET_KEY: str = Field("change-me-in-.env", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    
    # Database
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # CORS - Allow frontend origins
    # In Railway, set FRONTEND_URL environment variable
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:8000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8000",
            "https://localhost:5173",
            "https://localhost:8000",
            "https://127.0.0.1:5173",
            "https://127.0.0.1:8000",
        ]
    )
    
    # Additional origins from environment (for Railway deployment)
    FRONTEND_URL: Optional[str] = os.getenv("FRONTEND_URL")
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_FOLDER: str = "./uploads"
    
    # Background Tasks
    ENABLE_BACKGROUND_TASKS: bool = True
    
    # AI API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()