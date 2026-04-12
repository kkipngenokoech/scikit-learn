import os
import pytest
from unittest.mock import patch
from src.core.config import Config

class TestConfig:
    def test_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.api_key is None
            assert config.debug is False
            assert config.database_url == 'sqlite:///app.db'
            assert config.port == 8000
            assert config.host == '0.0.0.0'
    
    def test_environment_variables(self):
        env_vars = {
            'API_KEY': 'test-key-123',
            'DEBUG': 'true',
            'DATABASE_URL': 'postgresql://localhost/test',
            'PORT': '3000',
            'HOST': '127.0.0.1'
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.api_key == 'test-key-123'
            assert config.debug is True
            assert config.database_url == 'postgresql://localhost/test'
            assert config.port == 3000
            assert config.host == '127.0.0.1'
    
    def test_debug_false_values(self):
        test_cases = ['false', 'False', 'FALSE', '0', 'no']
        for value in test_cases:
            with patch.dict(os.environ, {'DEBUG': value}, clear=True):
                config = Config()
                assert config.debug is False
    
    def test_validate_with_api_key(self):
        with patch.dict(os.environ, {'API_KEY': 'valid-key'}, clear=True):
            config = Config()
            assert config.validate() is True
    
    def test_validate_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.validate() is False