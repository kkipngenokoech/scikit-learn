import os
from pathlib import Path
from typing import Optional

class Config:
    """Configuration management for the application."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    @property
    def database_url(self) -> str:
        """Get database URL from environment or default to SQLite."""
        return os.getenv("DATABASE_URL", f"sqlite:///{self.data_dir}/app.db")
    
    @property
    def log_level(self) -> str:
        """Get log level from environment or default to INFO."""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def debug(self) -> bool:
        """Get debug mode from environment."""
        return os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    
    @property
    def secret_key(self) -> str:
        """Get secret key from environment or generate default."""
        return os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# Global config instance
config = Config()
