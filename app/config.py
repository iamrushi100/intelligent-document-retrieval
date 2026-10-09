"""Environment-backed settings for the API and ingestion pipeline."""
from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    database_path: Path = Path("data/documents.db")
    max_upload_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 1000
    chunk_overlap: int = 150

    @classmethod
    def from_environment(cls) -> "Settings":
        """Read optional overrides while keeping useful local defaults."""
        return cls(
            database_path=Path(os.getenv("DATABASE_PATH", "data/documents.db")),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        )
