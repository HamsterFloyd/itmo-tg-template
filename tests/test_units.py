"""Чистые функции: разбиение ответа, сокращение истории, настройки LLM."""

import pytest
from conftest import LLM_ENV, TOKEN

from app.assistant import parse_temperature
from app.config import ConfigError, Settings
from app.context import HistoryLimits, trim_history
from app.llm import ChatMessage
from app.splitting import split_message


def utf16(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def test_short_text_is_one_part():
    # Arrange / Act
    parts = split_message("Короткий ответ")
    # Assert
    assert parts == ["Короткий ответ"]


def test_split_prefers_paragraph_boundaries():
    # Arrange
    text = "а" * 3000 + "\n\n" + "б" * 3000 + "\nв" + " г" * 10
    # Act
    parts = split_message(text)
    # Assert
    assert parts[0] == "а" * 3000 + "\n\n"
    assert "".join(parts) == text
    assert all(len(part) <= 4096 for part in parts)


def test_text_without_spaces_is_cut_hard():
    # Arrange
    text = "x" * 10000
    # Act
    parts = split_message(text)
    # Assert
    assert [len(part) for part in parts] == [4096, 4096, 1808]


def test_emoji_are_counted_in_utf16_and_never_broken():
    # Arrange: каждый эмодзи занимает две единицы UTF-16.
    text = "😀" * 3000
    # Act
    parts = split_message(text)
    # Assert
    assert all(utf16(part) <= 4096 for part in parts)
    assert "".join(parts) == text
    assert [len(part) for part in parts] == [2048, 952]


def test_whitespace_only_parts_are_not_sent():
    # Arrange
    text = "слово" + " " * 5000
    # Act
    parts = split_message(text, limit=100)
    # Assert
    assert parts == ["слово" + " " * 95]


def test_trim_keeps_contiguous_tail_and_stops_at_oversized_message():
    # Arrange
    history = [
        ChatMessage("user", "старый"),
        ChatMessage("assistant", "о" * 500),
        ChatMessage("user", "вопрос"),
        ChatMessage("assistant", "ответ"),
    ]
    # Act: длинное сообщение не помещается, и всё, что старше него, тоже отбрасывается.
    kept = trim_history(history, current="текущий", limits=HistoryLimits(10, 100))
    # Assert
    assert kept == history[2:]


def test_trim_with_empty_history():
    # Arrange / Act
    kept = trim_history([], current="текущий", limits=HistoryLimits(10, 100))
    # Assert
    assert kept == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.0", 0.0), ("0.3", 0.3), (" 0,7 ", 0.7), ("1.00", 1.0), ("0.5", None), ("", None)],
)
def test_parse_temperature(raw, expected):
    # Arrange / Act
    result = parse_temperature(raw)
    # Assert
    assert result == expected


def test_llm_settings_are_loaded_and_key_hidden(tmp_path):
    # Arrange
    environ = {"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "db", **LLM_ENV, "LLM_PROJECT": "b1gfolder"}
    # Act
    config = Settings.load(tmp_path / ".env", environ=environ)
    # Assert
    assert config.llm_base_url == "https://llm.example/v1"
    assert config.llm_model_name == "yandexgpt-lite/latest"
    assert (config.llm_timeout, config.llm_max_tokens) == (30.0, 1500)
    assert (config.history_max_messages, config.history_max_chars) == (20, 12000)
    assert "llm-secret-key" not in repr(config)


@pytest.mark.parametrize("missing", ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"])
def test_missing_llm_setting_has_clear_error(tmp_path, missing):
    # Arrange
    environ = {"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "db", **LLM_ENV, missing: ""}
    # Act / Assert
    with pytest.raises(ConfigError, match=missing):
        Settings.load(tmp_path / ".env", environ=environ)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LLM_BASE_URL", "ftp://llm.example"),
        ("LLM_BASE_URL", "https://user:llm-secret-key@llm.example/v1"),
        ("LLM_TIMEOUT", "0"),
        ("LLM_MAX_TOKENS", "много"),
        ("HISTORY_MAX_MESSAGES", "1"),
        ("HISTORY_MAX_CHARS", "1000"),
    ],
)
def test_invalid_llm_setting_is_rejected_without_leaking(tmp_path, key, value):
    # Arrange
    environ = {"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "db", **LLM_ENV, key: value}
    # Act / Assert
    with pytest.raises(ConfigError, match=key) as exc:
        Settings.load(tmp_path / ".env", environ=environ)
    assert "llm-secret-key" not in str(exc.value)


@pytest.mark.parametrize(
    ("model", "name"),
    [("gpt://b1gfolder/yandexgpt/rc", "yandexgpt/rc"), ("qwen2.5:7b", "qwen2.5:7b")],
)
def test_model_name_hides_folder_id(model, name):
    # Arrange
    config = Settings(bot_token=TOKEN, postgres_password="db", llm_model=model)
    # Act / Assert
    assert config.llm_model_name == name
