"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import json
import os
import secrets
import socket
import time
from dataclasses import replace

import asyncpg
import pytest

from app.config import Settings
from app.db import create_pool
from app.health import HealthState, health_result
from scripts.common import CommandError, run_command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]


@pytest.fixture
def database():
    name = "itmo-test-" + secrets.token_hex(6)
    volume = name + "-data"
    password = secrets.token_urlsafe(24)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def docker(*args):
        return run_command(["docker", *args], label="Интеграционный Docker-тест", timeout=180)

    def wait_until_ready():
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                return
            except CommandError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    docker("info")
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:5432",
            "-e",
            "POSTGRES_DB=bot",
            "-e",
            "POSTGRES_USER=bot",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/postgresql/data",
            "postgres:16-bookworm",
        )
        info = json.loads(docker("inspect", name))[0]
        assert int(info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]) == port
        wait_until_ready()
        yield (
            Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password=password,
                postgres_port=port,
            ),
            docker,
            wait_until_ready,
            name,
        )
    finally:
        try:
            docker("rm", "-f", name)
        finally:
            docker("volume", "rm", volume)


async def test_real_db_password_restart_health_and_persistence(database):
    # Arrange
    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        await pool.execute("CREATE TABLE integration_probe (value text NOT NULL)")
        await pool.execute("INSERT INTO integration_probe VALUES ($1)", "Сохранилось 👋")
        # Act / Assert: реальное подключение и ошибка пароля.
        assert (await health_result(state))[0] == 200
        with pytest.raises(asyncpg.InvalidPasswordError):
            await create_pool(replace(settings, postgres_password="wrong-password"))
        # Act / Assert: потеря соединения и сохранность данных после запуска.
        await asyncio.to_thread(docker, "stop", name)
        assert (await health_result(state))[0] == 503
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        for _ in range(60):
            if (await health_result(state))[0] == 200:
                break
            await asyncio.sleep(1)
        assert (await health_result(state))[0] == 200
        assert await pool.fetchval("SELECT value FROM integration_probe") == "Сохранилось 👋"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


async def test_storage_separates_users_survives_restart_and_guards_history(database):
    # Arrange
    from app.llm import ChatMessage
    from app.migrations import apply_migrations
    from app.storage import Storage

    settings, *_ = database
    alice, bob = 101, 202
    pool = await create_pool(settings)
    try:
        await apply_migrations(pool)
        await apply_migrations(pool)  # повторный запуск не меняет схему
        storage = Storage(pool)
        first = await storage.get_settings(alice)
        await storage.get_settings(bob)
        # Act: история двух пользователей, лимит хранения и смена настроек.
        for i in range(3):
            await storage.save_exchange(alice, 0, f"вопрос {i}", f"ответ {i}", keep_last=4)
        await storage.save_exchange(bob, 0, "вопрос Боба", "ответ Бобу", keep_last=4)
        await storage.set_temperature(alice, 0.7)
        # Assert
        assert (first.mode, first.temperature, first.context_version) == ("study", 0.3, 0)
        assert await storage.load_history(alice, 20) == [
            ChatMessage("user", "вопрос 1"),
            ChatMessage("assistant", "ответ 1"),
            ChatMessage("user", "вопрос 2"),
            ChatMessage("assistant", "ответ 2"),
        ]
        assert await storage.load_history(alice, 2) == [
            ChatMessage("user", "вопрос 2"),
            ChatMessage("assistant", "ответ 2"),
        ]
        assert len(await storage.load_history(bob, 20)) == 2
        assert await pool.fetchval("SELECT count(*) FROM schema_migrations") == 1
    finally:
        await pool.close()

    # Act: «перезапуск» приложения — новый пул, те же данные.
    pool = await create_pool(settings)
    try:
        storage = Storage(pool)
        restored = await storage.get_settings(alice)
        history_before_reset = await storage.load_history(alice, 20)
        await storage.clear_history(alice)
        stale_saved = await storage.save_exchange(alice, 0, "поздний", "ответ", keep_last=4)
        await storage.set_mode(bob, "translate")
        # Assert
        assert (restored.mode, restored.temperature) == ("study", 0.7)
        assert len(history_before_reset) == 4
        assert stale_saved is False
        assert await storage.load_history(alice, 20) == []
        alice_settings = await storage.get_settings(alice)
        assert (alice_settings.temperature, alice_settings.context_version) == (0.7, 1)
        assert await storage.load_history(bob, 20) == []
        assert (await storage.get_settings(bob)).mode == "translate"
    finally:
        await pool.close()
