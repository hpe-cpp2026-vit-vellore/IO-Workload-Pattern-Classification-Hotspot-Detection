from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Literal
import os

class Settings(BaseSettings):
    # Environment Toggle
    environment: Literal["development", "production", "testing"] = "development"
    
    # Message Bus
    bus_type: Literal["redis", "kafka"] = "redis"
    redis_url: str = "redis://redis:6379/0"
    kafka_brokers: str = "kafka-cluster:9092"
    
    # Security
    api_key_secret: str = "super-secret-local-key"

    @classmethod
    def _detect_redis_host_local(cls) -> str:
        """Auto-detect the best Redis host address (handles WSL2 networking).
        
        This mimics the original _detect_redis_host implementation to preserve
        WSL2 and local development fallback capability.
        """
        import subprocess, socket
        # If running inside docker container, 'redis' hostname resolves.
        # But if running locally outside docker, we want the detected host/IP.
        candidates = []
        try:
            result = subprocess.run(
                ["wsl", "hostname", "-I"],
                capture_output=True, text=True, timeout=3
            )
            wsl_ip = result.stdout.strip().split()[0] if result.returncode == 0 else None
            if wsl_ip:
                candidates.append(wsl_ip)
        except Exception:
            pass
        candidates.append("127.0.0.1")

        for host in candidates:
            try:
                # We verify connectivity to port 6379
                with socket.create_connection((host, 6379), timeout=1):
                    return host
            except OSError:
                continue
        return "127.0.0.1"

    @model_validator(mode="before")
    @classmethod
    def resolve_redis_compatibility(cls, data: any) -> any:
        if isinstance(data, dict):
            # If redis_url is not set, or is the default, resolve dynamically
            url_val = data.get("redis_url")
            # If redis_url is set as env variable REDIS_URL, pydantic-settings loads it.
            # Otherwise we check for REDIS_HOST env var, then fallback to auto-detection.
            env_url = os.environ.get("REDIS_URL")
            env_host = os.environ.get("REDIS_HOST")
            
            if env_url:
                data["redis_url"] = env_url
            elif not url_val or url_val == "redis://redis:6379/0":
                if env_host:
                    data["redis_url"] = f"redis://{env_host}:6379/0"
                else:
                    detected_host = cls._detect_redis_host_local()
                    data["redis_url"] = f"redis://{detected_host}:6379/0"
        return data

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"

settings = Settings()
