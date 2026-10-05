"""Общие тестовые замены: хранилище в памяти, управляемая модель и отправка в Telegram."""

from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from app.assistant import Assistant
from app.context import HistoryLimits
from app.handlers.assistant import create_router
from app.llm import ChatMessage, LLMError, LLMResult
from app.storage import UserSettings

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
LLM_ENV = {
    "LLM_BASE_URL": "https://llm.example/v1",
    "LLM_API_KEY": "llm-secret-key",
    "LLM_MODEL": "gpt://b1gfolder/yandexgpt-lite/latest",
}


class FakeStore:
    """Повторяет поведение Storage без PostgreSQL; SQL проверяется интеграционными тестами."""

    def __init__(self):
        self.settings: dict[int, UserSettings] = {}
        self.history: dict[int, list[ChatMessage]] = {}

    async def get_settings(self, chat_id: int) -> UserSettings:
        return self.settings.setdefault(chat_id, UserSettings())

    async def set_mode(self, chat_id: int, mode: str) -> None:
        current = await self.get_settings(chat_id)
        self.settings[chat_id] = replace(
            current, mode=mode, context_version=current.context_version + 1
        )
        self.history.pop(chat_id, None)

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        self.settings[chat_id] = replace(await self.get_settings(chat_id), temperature=temperature)

    async def clear_history(self, chat_id: int) -> None:
        current = await self.get_settings(chat_id)
        self.settings[chat_id] = replace(current, context_version=current.context_version + 1)
        self.history.pop(chat_id, None)

    async def load_history(self, chat_id: int, limit: int) -> list[ChatMessage]:
        return list(self.history.get(chat_id, [])[-limit:])

    async def save_exchange(
        self, chat_id: int, context_version: int, question: str, answer: str, *, keep_last: int
    ) -> bool:
        if (await self.get_settings(chat_id)).context_version != context_version:
            return False
        history = self.history.setdefault(chat_id, [])
        history += [ChatMessage("user", question), ChatMessage("assistant", answer)]
        del history[:-keep_last]
        return True


class FakeLLM:
    """Возвращает заранее заданные ответы и запоминает фактические запросы."""

    def __init__(self, *responses: LLMResult | LLMError | str):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, messages, *, temperature: float, request_id: str) -> LLMResult:
        self.calls.append({"messages": list(messages), "temperature": temperature})
        response = self.responses.pop(0) if self.responses else "Ответ модели"
        if isinstance(response, LLMError):
            raise response
        return LLMResult(response) if isinstance(response, str) else response


class BotHarness:
    """Прогоняет обновления через настоящий Dispatcher; запросы к Telegram подменены."""

    def __init__(self, llm: FakeLLM, store: FakeStore, limits: HistoryLimits):
        self.llm = llm
        self.store = store
        self.assistant = Assistant(
            store=store, llm=llm, limits=limits, model_name="yandexgpt-lite/latest"
        )
        self.bot = Bot(TOKEN)
        self.bot.session = AsyncMock()
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(create_router())
        self.update_id = 0

    def sent(self) -> list[SendMessage]:
        return [
            call.args[1]
            for call in self.bot.session.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]

    def sent_texts(self, chat_id: int | None = None) -> list[str]:
        return [m.text for m in self.sent() if chat_id is None or m.chat_id == chat_id]

    async def feed(self, update: Update) -> None:
        await self.dispatcher.feed_update(self.bot, update, assistant=self.assistant)

    async def send(self, chat_id: int, text: str | None = None, chat_type: str = "private"):
        self.update_id += 1
        message = Message(
            message_id=self.update_id,
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type=chat_type),
            from_user=User(id=chat_id, is_bot=False, first_name="Студент"),
            text=text,
        )
        await self.feed(Update(update_id=self.update_id, message=message))

    async def press(self, chat_id: int, data: str) -> None:
        self.update_id += 1
        message = Message(
            message_id=self.update_id,
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type="private"),
            text="Текущие настройки",
        )
        callback = CallbackQuery(
            id=str(self.update_id),
            from_user=User(id=chat_id, is_bot=False, first_name="Студент"),
            chat_instance="test",
            message=message,
            data=data,
        )
        await self.feed(Update(update_id=self.update_id, callback_query=callback))


@pytest.fixture
def limits() -> HistoryLimits:
    return HistoryLimits(max_messages=20, max_chars=12000)


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def harness(llm, store, limits) -> BotHarness:
    return BotHarness(llm, store, limits)
