import os
from typing import Optional

class Config:
    def __init__(self):
        self.api_key: Optional[str] = os.getenv('API_KEY')
        self.debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
        self.database_url: str = os.getenv('DATABASE_URL', 'sqlite:///app.db')
        self.port: int = int(os.getenv('PORT', '8000'))
        self.host: str = os.getenv('HOST', '0.0.0.0')
    
    def validate(self) -> bool:
        """Validate required configuration values."""
        if not self.api_key:
            return False
        return True

config = Config()