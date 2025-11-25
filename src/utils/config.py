import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logging import get_logger

load_dotenv()
logger = get_logger(__name__)

class AIConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore",
    )
    FRAUD_MODEL_LOCAL_PATH: str = os.getenv("FRAUD_MODEL_LOCAL_PATH") 
    FRAUD_MODEL_REPO_ID: str = os.getenv("FRAUD_MODEL_REPO_ID")
    FRAUD_THRESHOLD: float = float(os.getenv("FRAUD_THRESHOLD"))
    
    HOST: str = os.getenv("HOST")
    PORT: int = int(os.getenv("PORT"))

ai_config = AIConfig()
logger.info("Successfully loaded AI config")
