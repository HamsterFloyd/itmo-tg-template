"""Обработчики Telegram: разбирают команды и отправляют ответы.

HTTP-запросы к модели, SQL и тексты промптов находятся в других модулях;
сервис Assistant передаётся в обработчики через аргумент assistant.
"""

import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender

from app import texts
from app.assistant import Assistant
from app.prompts import MODES
from app.splitting import split_message

logger = logging.getLogger("app.handlers")

TEMPERATURE_PREFIX = "temperature:"


def temperature_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=texts.temperature_text(value),
            callback_data=TEMPERATURE_PREFIX + texts.temperature_text(value),
        )
        for value in texts.TEMPERATURE_CHOICES
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


async def send_long(message: Message, text: str) -> None:
    for part in split_message(text):
        # Ответ модели отправляется как обычный текст: разметка не ломается при разбиении.
        await message.answer(part, parse_mode=None)


def create_router() -> Router:
    """Новый Router при каждом вызове: aiogram не позволяет подключить один Router дважды."""
    router = Router(name="assistant")
    router.message.filter(F.chat.type == "private")
    router.callback_query.filter(F.message.chat.type == "private")

    @router.message(CommandStart())
    async def start(message: Message, assistant: Assistant) -> None:
        mode = await assistant.current_mode(message.chat.id)
        await message.answer(texts.start(mode), parse_mode=None)

    @router.message(Command(*MODES))
    async def switch_mode(message: Message, command: CommandObject, assistant: Assistant) -> None:
        mode = MODES[command.command]
        await assistant.switch_mode(message.chat.id, mode)
        await message.answer(texts.mode_switched(mode), parse_mode=None)

    @router.message(Command("reset"))
    async def reset(message: Message, assistant: Assistant) -> None:
        await assistant.reset(message.chat.id)
        await message.answer(texts.RESET_DONE, parse_mode=None)

    @router.message(Command("settings"))
    async def settings(message: Message, command: CommandObject, assistant: Assistant) -> None:
        if command.args:
            reply = await assistant.set_temperature(message.chat.id, command.args)
            await message.answer(reply, parse_mode=None)
            return
        await message.answer(
            await assistant.settings_text(message.chat.id),
            parse_mode=None,
            reply_markup=temperature_keyboard(),
        )

    @router.callback_query(F.data.startswith(TEMPERATURE_PREFIX))
    async def temperature_button(callback: CallbackQuery, bot: Bot, assistant: Assistant) -> None:
        # callback_data приходит от клиента и проверяется так же, как текст команды.
        chat_id = callback.message.chat.id
        reply = await assistant.set_temperature(chat_id, callback.data[len(TEMPERATURE_PREFIX) :])
        await callback.answer()
        await bot.send_message(chat_id, reply, parse_mode=None)

    @router.message(F.text.startswith("/"))
    async def unknown_command(message: Message) -> None:
        await message.answer(texts.UNKNOWN_COMMAND, parse_mode=None)

    @router.message(F.text)
    async def text_message(message: Message, bot: Bot, assistant: Assistant) -> None:
        # Индикатор «печатает…» повторяется, пока модель формирует ответ.
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            reply = await assistant.answer(message.chat.id, message.text)
        await send_long(message, reply)

    @router.message()
    async def not_text(message: Message) -> None:
        await message.answer(texts.TEXT_ONLY, parse_mode=None)

    @router.errors()
    async def on_error(event: ErrorEvent) -> None:
        # Ошибки БД и прочие сбои: подробности только в журнале (через фильтр секретов).
        logger.error("Ошибка обработки обновления: %s", type(event.exception).__name__)
        message = event.update.message
        if message is not None:
            await message.answer(texts.INTERNAL_ERROR, parse_mode=None)

    return router
