import os
import pytest
from unittest.mock import patch
from src.core.config import AppConfig, DatabaseConfig


class TestAppConfig:
    def test_default_values(self):
        config = AppConfig()
        assert config.debug is False
        assert config.secret_key == "dev-secret-key"
        assert isinstance(config.database, DatabaseConfig)
        assert config.database.host == "localhost"
    
    def test_custom_values(self):
        db_config = DatabaseConfig(host="custom-host")
        config = AppConfig(
            debug=True,
            secret_key="custom-key",
            database=db_config
        )
        assert config.debug is True
        assert config.secret_key == "custom-key"
        assert config.database.host == "custom-host"
    
    @patch.dict(os.environ, {
        'DEBUG': 'true',
        'SECRET_KEY': 'env-secret',
        'DB_HOST': 'env-db-host'
    })
    def test_from_env_with_vars(self):
        config = AppConfig.from_env()
        assert config.debug is True
        assert config.secret_key == "env-secret"
        assert config.database.host == "env-db-host"
    
    @patch.dict(os.environ, {'DEBUG': 'false'}, clear=True)
    def test_from_env_debug_false(self):
        config = AppConfig.from_env()
        assert config.debug is False
    
    @patch.dict(os.environ, {'DEBUG': 'TRUE'}, clear=True)
    def test_from_env_debug_case_insensitive(self):
        config = AppConfig.from_env()
        assert config.debug is True
    
    @patch.dict(os.environ, {'DEBUG': 'invalid'}, clear=True)
    def test_from_env_debug_invalid_value(self):
        config = AppConfig.from_env()
        assert config.debug is False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_no_vars(self):
        config = AppConfig.from_env()
        assert config.debug is False
        assert config.secret_key == "dev-secret-key"
        assert config.database.host == "localhost"