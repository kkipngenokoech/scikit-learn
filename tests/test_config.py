import os
import pytest
from unittest.mock import patch
from src.core.config import DatabaseConfig, RedisConfig, AppConfig


class TestDatabaseConfig:
    def test_default_values(self):
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "myapp"
        assert config.user == "postgres"
        assert config.password == ""
    
    def test_custom_values(self):
        config = DatabaseConfig(
            host="db.example.com",
            port=3306,
            name="production",
            user="admin",
            password="secret"
        )
        assert config.host == "db.example.com"
        assert config.port == 3306
        assert config.name == "production"
        assert config.user == "admin"
        assert config.password == "secret"
    
    @patch.dict(os.environ, {
        'DB_HOST': 'test-host',
        'DB_PORT': '3306',
        'DB_NAME': 'test-db',
        'DB_USER': 'test-user',
        'DB_PASSWORD': 'test-pass'
    })
    def test_from_env(self):
        config = DatabaseConfig.from_env()
        assert config.host == "test-host"
        assert config.port == 3306
        assert config.name == "test-db"
        assert config.user == "test-user"
        assert config.password == "test-pass"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        config = DatabaseConfig.from_env()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "myapp"
        assert config.user == "postgres"
        assert config.password == ""


class TestRedisConfig:
    def test_default_values(self):
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
    
    def test_custom_values(self):
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="redis-pass"
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "redis-pass"
    
    @patch.dict(os.environ, {
        'REDIS_HOST': 'test-redis',
        'REDIS_PORT': '6380',
        'REDIS_DB': '2',
        'REDIS_PASSWORD': 'redis-secret'
    })
    def test_from_env(self):
        config = RedisConfig.from_env()
        assert config.host == "test-redis"
        assert config.port == 6380
        assert config.db == 2
        assert config.password == "redis-secret"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        config = RedisConfig.from_env()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None


class TestAppConfig:
    def test_default_values(self):
        config = AppConfig()
        assert config.debug is False
        assert config.secret_key == "dev-secret-key"
        assert config.log_level == "INFO"
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.redis, RedisConfig)
    
    def test_custom_values(self):
        db_config = DatabaseConfig(host="custom-db")
        redis_config = RedisConfig(host="custom-redis")
        config = AppConfig(
            debug=True,
            secret_key="custom-key",
            log_level="DEBUG",
            database=db_config,
            redis=redis_config
        )
        assert config.debug is True
        assert config.secret_key == "custom-key"
        assert config.log_level == "DEBUG"
        assert config.database.host == "custom-db"
        assert config.redis.host == "custom-redis"
    
    @patch.dict(os.environ, {
        'DEBUG': 'true',
        'SECRET_KEY': 'env-secret',
        'LOG_LEVEL': 'WARNING',
        'DB_HOST': 'env-db',
        'REDIS_HOST': 'env-redis'
    })
    def test_from_env(self):
        config = AppConfig.from_env()
        assert config.debug is True
        assert config.secret_key == "env-secret"
        assert config.log_level == "WARNING"
        assert config.database.host == "env-db"
        assert config.redis.host == "env-redis"
    
    @patch.dict(os.environ, {'DEBUG': 'false'})
    def test_debug_false_from_env(self):
        config = AppConfig.from_env()
        assert config.debug is False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        config = AppConfig.from_env()
        assert config.debug is False
        assert config.secret_key == "dev-secret-key"
        assert config.log_level == "INFO"
        assert config.database.host == "localhost"
        assert config.redis.host == "localhost"
