"""Хранение настроек пользователей и истории диалога в PostgreSQL."""

from dataclasses import dataclass

import asyncpg

from app.llm import ChatMessage
from app.prompts import DEFAULT_MODE

DEFAULT_TEMPERATURE = 0.3


@dataclass(frozen=True)
class UserSettings:
    mode: str = DEFAULT_MODE
    temperature: float = DEFAULT_TEMPERATURE
    context_version: int = 0


class Storage:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def get_settings(self, chat_id: int) -> UserSettings:
        # Строка создаётся при первом обращении со значениями по умолчанию из схемы.
        await self._pool.execute(
            "INSERT INTO user_settings (chat_id) VALUES ($1) ON CONFLICT (chat_id) DO NOTHING",
            chat_id,
        )
        row = await self._pool.fetchrow(
            "SELECT mode, temperature, context_version FROM user_settings WHERE chat_id = $1",
            chat_id,
        )
        return UserSettings(row["mode"], row["temperature"], row["context_version"])

    async def set_mode(self, chat_id: int, mode: str) -> None:
        """Сохраняет режим и очищает историю предыдущего режима одной транзакцией."""
        async with self._pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO user_settings (chat_id, mode) VALUES ($1, $2)
                ON CONFLICT (chat_id) DO UPDATE
                SET mode = EXCLUDED.mode,
                    context_version = user_settings.context_version + 1,
                    updated_at = now()
                """,
                chat_id,
                mode,
            )
            await connection.execute("DELETE FROM dialog_messages WHERE chat_id = $1", chat_id)

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        await self._pool.execute(
            """
            INSERT INTO user_settings (chat_id, temperature) VALUES ($1, $2)
            ON CONFLICT (chat_id) DO UPDATE
            SET temperature = EXCLUDED.temperature, updated_at = now()
            """,
            chat_id,
            temperature,
        )

    async def clear_history(self, chat_id: int) -> None:
        """Удаляет историю; режим и temperature сохраняются."""
        async with self._pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """
                UPDATE user_settings
                SET context_version = context_version + 1, updated_at = now()
                WHERE chat_id = $1
                """,
                chat_id,
            )
            await connection.execute("DELETE FROM dialog_messages WHERE chat_id = $1", chat_id)

    async def load_history(self, chat_id: int, limit: int) -> list[ChatMessage]:
        rows = await self._pool.fetch(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM dialog_messages
                WHERE chat_id = $1 ORDER BY id DESC LIMIT $2
            ) AS latest
            ORDER BY id
            """,
            chat_id,
            limit,
        )
        return [ChatMessage(row["role"], row["content"]) for row in rows]

    async def save_exchange(
        self, chat_id: int, context_version: int, question: str, answer: str, *, keep_last: int
    ) -> bool:
        """Сохраняет вопрос и успешный ответ парой и удаляет записи старше keep_last.

        Если пока модель отвечала, пользователь выполнил /reset или сменил режим,
        context_version уже изменился: ответ отправляется, но в новую историю не попадает.
        context_version берётся из get_settings, которая и создаёт строку пользователя.
        """
        async with self._pool.acquire() as connection, connection.transaction():
            current = await connection.fetchval(
                "SELECT context_version FROM user_settings WHERE chat_id = $1 FOR UPDATE",
                chat_id,
            )
            if current != context_version:
                return False
            await connection.executemany(
                "INSERT INTO dialog_messages (chat_id, role, content) VALUES ($1, $2, $3)",
                [(chat_id, "user", question), (chat_id, "assistant", answer)],
            )
            await connection.execute(
                """
                DELETE FROM dialog_messages
                WHERE chat_id = $1 AND id <= (
                    SELECT id FROM dialog_messages WHERE chat_id = $1
                    ORDER BY id DESC OFFSET $2 LIMIT 1
                )
                """,
                chat_id,
                keep_last,
            )
        return True
