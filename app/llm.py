"""Клиент OpenAI-совместимого Chat Completions API (YandexGPT, локальные модели и др.).

Модуль ничего не знает о Telegram и PostgreSQL: обработчики получают либо
LLMResult, либо LLMError с причиной, которую можно безопасно показать пользователю.
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

import aiohttp

logger = logging.getLogger("app.llm")

Role = Literal["system", "user", "assistant"]
RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMErrorReason(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    BAD_RESPONSE = "bad_response"
    EMPTY = "empty"


class LLMError(Exception):
    """Ошибка вызова модели. Текст исключения не содержит ключей и данных диалога."""

    def __init__(self, reason: LLMErrorReason, status: int | None = None):
        super().__init__(reason.value if status is None else f"{reason.value} (HTTP {status})")
        self.reason = reason
        self.status = status


class LLMClient(Protocol):
    async def complete(
        self, messages: Sequence[ChatMessage], *, temperature: float, request_id: str
    ) -> LLMResult: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        model_name: str,
        project: str = "",
        timeout: float = 30.0,
        max_tokens: int = 1500,
        max_retries: int = 1,
        retry_delay: float = 1.0,
    ):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        if project:
            # Yandex AI Studio получает ID каталога в заголовке OpenAI-Project.
            self._headers["OpenAI-Project"] = project
        self._model = model
        self._model_name = model_name
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        # trust_env=False: прокси Telegram и системные HTTP_PROXY к модели не применяются.
        self._session = aiohttp.ClientSession(trust_env=False)

    async def close(self) -> None:
        await self._session.close()

    async def complete(
        self, messages: Sequence[ChatMessage], *, temperature: float, request_id: str
    ) -> LLMResult:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        started = time.monotonic()
        logger.info(
            "LLM %s: запрос к %s, temperature=%.1f, сообщений=%d",
            request_id,
            self._model_name,
            temperature,
            len(messages),
        )
        attempt = 0
        while True:
            try:
                result = await self._request(payload)
            except LLMError as exc:
                # Повторяем только временные сбои: сеть, таймаут, 429 и 5xx шлюза.
                retryable = exc.status in RETRY_STATUSES or (
                    exc.status is None
                    and exc.reason in {LLMErrorReason.TIMEOUT, LLMErrorReason.UNAVAILABLE}
                )
                if retryable and attempt < self._max_retries:
                    attempt += 1
                    delay = self._retry_delay * 2 ** (attempt - 1)
                    logger.warning(
                        "LLM %s: временная ошибка %s, повтор %d через %.1f с",
                        request_id,
                        exc,
                        attempt,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "LLM %s: ошибка %s за %d мс",
                    request_id,
                    exc,
                    (time.monotonic() - started) * 1000,
                )
                raise
            logger.info(
                "LLM %s: ответ за %d мс, finish_reason=%s, токены: запрос=%s, ответ=%s, всего=%s",
                request_id,
                (time.monotonic() - started) * 1000,
                result.finish_reason,
                result.prompt_tokens,
                result.completion_tokens,
                result.total_tokens,
            )
            return result

    async def _request(self, payload: dict) -> LLMResult:
        try:
            async with self._session.post(
                self._url, json=payload, headers=self._headers, timeout=self._timeout
            ) as response:
                if response.status != 200:
                    raise LLMError(_status_reason(response.status), response.status)
                try:
                    body = await response.json(content_type=None)
                except ValueError:
                    raise LLMError(LLMErrorReason.BAD_RESPONSE, response.status) from None
        except TimeoutError:
            raise LLMError(LLMErrorReason.TIMEOUT) from None
        except aiohttp.ClientError:
            # Текст исключения aiohttp может содержать адрес; наружу отдаём только причину.
            raise LLMError(LLMErrorReason.UNAVAILABLE) from None
        return _parse_result(body)


def _status_reason(status: int) -> LLMErrorReason:
    if status in {401, 403}:
        return LLMErrorReason.AUTH
    if status == 429:
        return LLMErrorReason.RATE_LIMIT
    if status in {408, 504}:
        return LLMErrorReason.TIMEOUT
    if 400 <= status < 500:
        return LLMErrorReason.BAD_REQUEST
    return LLMErrorReason.UNAVAILABLE


def _parse_result(body: object) -> LLMResult:
    try:
        choice = body["choices"][0]  # type: ignore[index]
        text = choice["message"].get("content")
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise LLMError(LLMErrorReason.BAD_RESPONSE) from None
    if text is not None and not isinstance(text, str):
        raise LLMError(LLMErrorReason.BAD_RESPONSE)
    if not text or not text.strip():
        raise LLMError(LLMErrorReason.EMPTY)
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}

    def tokens(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) else None

    return LLMResult(
        text=text,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        prompt_tokens=tokens("prompt_tokens"),
        completion_tokens=tokens("completion_tokens"),
        total_tokens=tokens("total_tokens"),
    )
