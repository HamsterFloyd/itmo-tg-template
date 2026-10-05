"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_project: str = ""
    llm_timeout: float = 30.0
    llm_max_tokens: int = 1500
    history_max_messages: int = 20
    history_max_chars: int = 12000

    @property
    def llm_model_name(self) -> str:
        """Имя модели для пользователя без ID каталога: gpt://b1g…/yandexgpt-lite/latest."""
        scheme, separator, rest = self.llm_model.partition("://")
        if not separator:
            return self.llm_model
        return rest.partition("/")[2] or scheme

    @classmethod
    def load(
        cls, env_file: Path | str = ".env", *, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        def number(key: str, default: str, minimum: float, maximum: float, kind: type) -> float:
            try:
                result = kind(value(key, default))
                if not minimum <= result <= maximum:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно число от {minimum} до {maximum}.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        base_url = value("LLM_BASE_URL").rstrip("/")
        parsed_url = urlsplit(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ConfigError(
                "LLM_BASE_URL: укажите адрес OpenAI-совместимого API, "
                "например https://ai.api.cloud.yandex.net/v1."
            )
        if parsed_url.username or parsed_url.password or parsed_url.query:
            raise ConfigError("LLM_BASE_URL: адрес не должен содержать ключ или параметры.")
        api_key = value("LLM_API_KEY")
        if not api_key:
            raise ConfigError("LLM_API_KEY: ключ доступа к языковой модели не задан.")
        model = value("LLM_MODEL")
        if not model:
            raise ConfigError(
                "LLM_MODEL: укажите модель, например gpt://<ID каталога>/yandexgpt-lite/latest."
            )
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            llm_base_url=base_url,
            llm_api_key=api_key,
            llm_model=model,
            llm_project=value("LLM_PROJECT"),
            llm_timeout=number("LLM_TIMEOUT", "30", 1, 300, float),
            llm_max_tokens=number("LLM_MAX_TOKENS", "1500", 1, 32000, int),
            history_max_messages=number("HISTORY_MAX_MESSAGES", "20", 2, 200, int),
            # Нижняя граница — лимит сообщения Telegram: текущий запрос всегда помещается.
            history_max_chars=number("HISTORY_MAX_CHARS", "12000", 4096, 200000, int),
        )
