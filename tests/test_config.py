import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.config import Config


class TestConfig:
    """Test configuration management."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.secret_key == "dev-secret-key-change-in-production"
        assert "sqlite://" in config.database_url
    
    def test_environment_overrides(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            "LOG_LEVEL": "DEBUG",
            "DEBUG": "true",
            "SECRET_KEY": "test-secret",
            "DATABASE_URL": "postgresql://test"
        }):
            config = Config()
            
            assert config.log_level == "DEBUG"
            assert config.debug is True
            assert config.secret_key == "test-secret"
            assert config.database_url == "postgresql://test"
    
    def test_debug_flag_variations(self):
        """Test different ways to set debug flag."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("", False)
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"DEBUG": env_value}):
                config = Config()
                assert config.debug == expected
    
    def test_directories_created(self):
        """Test that required directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the base directory to use temp directory
            with patch.object(Path, "parent", new_callable=lambda: Path(temp_dir)):
                config = Config()
                
                assert config.data_dir.exists()
                assert config.logs_dir.exists()
