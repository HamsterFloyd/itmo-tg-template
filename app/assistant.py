"""Сценарии ассистента: связывает хранилище, режимы и клиент модели."""

import uuid
from decimal import Decimal, InvalidOperation
from typing import Protocol

from app import texts
from app.context import HistoryLimits, build_messages
from app.llm import ChatMessage, LLMClient, LLMError
from app.prompts import Mode, get_mode
from app.storage import UserSettings


class SettingsStore(Protocol):
    async def get_settings(self, chat_id: int) -> UserSettings: ...
    async def set_mode(self, chat_id: int, mode: str) -> None: ...
    async def set_temperature(self, chat_id: int, temperature: float) -> None: ...
    async def clear_history(self, chat_id: int) -> None: ...
    async def load_history(self, chat_id: int, limit: int) -> list[ChatMessage]: ...
    async def save_exchange(
        self, chat_id: int, context_version: int, question: str, answer: str, *, keep_last: int
    ) -> bool: ...


def parse_temperature(raw: str) -> float | None:
    """Возвращает одно из разрешённых значений или None; проверка выполняется до вызова API."""
    try:
        value = Decimal(raw.strip().replace(",", "."))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    for choice in texts.TEMPERATURE_CHOICES:
        if value == Decimal(str(choice)):
            return choice
    return None


class Assistant:
    def __init__(
        self, *, store: SettingsStore, llm: LLMClient, limits: HistoryLimits, model_name: str
    ):
        self.store = store
        self.llm = llm
        self.limits = limits
        self.model_name = model_name

    async def current_mode(self, chat_id: int) -> Mode:
        return get_mode((await self.store.get_settings(chat_id)).mode)

    async def switch_mode(self, chat_id: int, mode: Mode) -> None:
        await self.store.set_mode(chat_id, mode.command)

    async def reset(self, chat_id: int) -> None:
        await self.store.clear_history(chat_id)

    async def settings_text(self, chat_id: int) -> str:
        current = await self.store.get_settings(chat_id)
        return texts.settings(get_mode(current.mode), self.model_name, current.temperature)

    async def set_temperature(self, chat_id: int, raw: str) -> str:
        value = parse_temperature(raw)
        if value is None:
            return texts.TEMPERATURE_INVALID
        await self.store.set_temperature(chat_id, value)
        return texts.temperature_saved(value)

    async def answer(self, chat_id: int, text: str) -> str:
        """Возвращает ответ модели либо безопасное сообщение об ошибке; всегда непустой текст."""
        current = await self.store.get_settings(chat_id)
        history = await self.store.load_history(chat_id, self.limits.max_messages)
        messages = build_messages(get_mode(current.mode), history, text, self.limits)
        try:
            result = await self.llm.complete(
                messages, temperature=current.temperature, request_id=uuid.uuid4().hex[:12]
            )
        except LLMError as exc:
            # Сообщение ассистента при ошибке не сохраняется, вопрос тоже: история
            # остаётся парами «вопрос — успешный ответ».
            return texts.LLM_ERRORS[exc.reason]
        await self.store.save_exchange(
            chat_id,
            current.context_version,
            text,
            result.text,
            keep_last=self.limits.max_messages,
        )
        return result.text + (texts.TRUNCATED_NOTICE if result.truncated else "")
