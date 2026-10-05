import argparse
import asyncio
import logging

from aiogram import Dispatcher
from aiogram.types import BotCommand

from app.assistant import Assistant
from app.config import ConfigError, Settings
from app.context import HistoryLimits
from app.db import create_pool
from app.handlers.assistant import create_router
from app.health import HealthState, start_health_server
from app.llm import OpenAICompatibleClient
from app.logging_setup import configure_logging
from app.migrations import apply_migrations
from app.prompts import MODES
from app.storage import Storage
from app.telegram import create_bot

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    llm = None
    try:
        state.pool = await create_pool(settings)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        await apply_migrations(state.pool)
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        await set_commands(bot)
        llm = OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            model_name=settings.llm_model_name,
            project=settings.llm_project,
            timeout=settings.llm_timeout,
            max_tokens=settings.llm_max_tokens,
        )
        assistant = Assistant(
            store=Storage(state.pool),
            llm=llm,
            limits=HistoryLimits(settings.history_max_messages, settings.history_max_chars),
            model_name=settings.llm_model_name,
        )
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router())
        runner = await start_health_server(state, settings.health_port)
        state.polling_task = asyncio.create_task(
            dispatcher.start_polling(
                bot,
                db=state.pool,
                assistant=assistant,
                allowed_updates=dispatcher.resolve_used_update_types(),
                close_bot_session=False,
            )
        )
        state.initialized = True
        await state.polling_task
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        if runner:
            await runner.cleanup()
        if llm:
            await llm.close()
        await bot.session.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


async def set_commands(bot) -> None:
    """Меню команд в Telegram. Сбой не мешает работе: команды доступны и без меню."""
    commands = [BotCommand(command="start", description="Возможности и команды")]
    commands += [BotCommand(command=m.command, description=m.description) for m in MODES.values()]
    commands += [
        BotCommand(command="settings", description="Режим, модель и temperature"),
        BotCommand(command="reset", description="Очистить историю диалога"),
    ]
    try:
        async with asyncio.timeout(10):
            await bot.set_my_commands(commands)
    except Exception as exc:
        logger.warning("Не удалось обновить меню команд: %s", type(exc).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram-бот «AI-ассистент студента»")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен и прокси.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
