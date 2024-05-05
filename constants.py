import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    Chroma_db_impl: str
    persist_directory: str
    anonymized_telemetry: bool

    class Config:
        env_prefix = "CHROMA_"


CHROMA_SETTINGS = Settings(
    Chroma_db_impl='duckdb+parquet',
    persist_directory="db",
    anonymized_telemetry=False
)
