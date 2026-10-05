"""Версионированная схема БД, применяемая при каждом запуске бота.

Каждая миграция выполняется один раз и записывается в schema_migrations.
Уже опубликованные миграции не изменяются: для новой таблицы или столбца
добавьте в конец списка следующую версию.
"""

import logging

import asyncpg

logger = logging.getLogger("app.migrations")

# Произвольный постоянный ключ: два процесса бота не применяют миграции одновременно.
LOCK_KEY = 726_110_001

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE user_settings (
            chat_id BIGINT PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'study',
            temperature DOUBLE PRECISION NOT NULL DEFAULT 0.3
                CHECK (temperature >= 0 AND temperature <= 1),
            -- Увеличивается при /reset и смене режима; см. Storage.save_exchange.
            context_version INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE dialog_messages (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            chat_id BIGINT NOT NULL REFERENCES user_settings (chat_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX dialog_messages_chat_id_id ON dialog_messages (chat_id, id);
        """,
    ),
]


async def apply_migrations(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection, connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock($1)", LOCK_KEY)
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        rows = await connection.fetch("SELECT version FROM schema_migrations")
        applied = {row["version"] for row in rows}
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            await connection.execute(sql)
            await connection.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
            logger.info("Применена миграция БД %d.", version)
