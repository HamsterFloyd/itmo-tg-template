"""Обязательные сценарии лабораторной № 1 без настоящих Telegram и LLM API."""

import pytest
from conftest import BotHarness, FakeLLM

from app import texts
from app.context import HistoryLimits
from app.llm import ChatMessage, LLMError, LLMErrorReason, LLMResult
from app.prompts import MODES, QUIZ, STUDY, TRANSLATE
from app.storage import UserSettings

ALICE, BOB = 101, 202


# 1. Обычное сообщение передаётся модели с инструкцией активного режима.


async def test_message_goes_to_llm_with_active_mode_prompt(harness, llm):
    # Arrange
    llm.responses = ["Список изменяемый, кортеж — нет."]
    # Act
    await harness.send(ALICE, "Чем список отличается от кортежа?")
    # Assert
    call = llm.calls[0]
    assert call["messages"] == [
        ChatMessage("system", STUDY.system_prompt),
        ChatMessage("user", "Чем список отличается от кортежа?"),
    ]
    assert call["temperature"] == 0.3
    assert harness.sent_texts(ALICE) == ["Список изменяемый, кортеж — нет."]


async def test_translate_mode_sends_few_shot_examples_before_history(harness, llm, store):
    # Arrange
    await store.set_mode(ALICE, "translate")
    await harness.send(ALICE, "Добрый день")
    # Act
    await harness.send(ALICE, "на немецкий: Спасибо")
    # Assert
    messages = llm.calls[1]["messages"]
    examples = len(TRANSLATE.examples) * 2
    assert messages[0] == ChatMessage("system", TRANSLATE.system_prompt)
    assert messages[1] == ChatMessage("user", TRANSLATE.examples[0][0])
    assert messages[2] == ChatMessage("assistant", TRANSLATE.examples[0][1])
    assert messages[1 + examples :] == [
        ChatMessage("user", "Добрый день"),
        ChatMessage("assistant", "Ответ модели"),
        ChatMessage("user", "на немецкий: Спасибо"),
    ]


def test_modes_have_distinct_prompts_and_one_uses_few_shot():
    # Arrange
    prompts = [mode.system_prompt for mode in MODES.values()]
    # Act
    few_shot = [mode.command for mode in MODES.values() if mode.examples]
    # Assert
    assert set(MODES) == {"study", "translate", "quiz"}
    assert len(set(prompts)) == 3
    assert few_shot == ["translate"]


# 2. Истории двух пользователей не смешиваются.


async def test_histories_of_two_users_are_separate(harness, llm, store):
    # Arrange
    llm.responses = ["Ответ Алисе", "Ответ Бобу", "Второй ответ Алисе"]
    # Act
    await harness.send(ALICE, "Вопрос Алисы")
    await harness.send(BOB, "Вопрос Боба")
    await harness.send(ALICE, "Второй вопрос Алисы")
    # Assert
    assert llm.calls[1]["messages"][1:] == [ChatMessage("user", "Вопрос Боба")]
    assert llm.calls[2]["messages"][1:] == [
        ChatMessage("user", "Вопрос Алисы"),
        ChatMessage("assistant", "Ответ Алисе"),
        ChatMessage("user", "Второй вопрос Алисы"),
    ]
    assert store.history[BOB] == [
        ChatMessage("user", "Вопрос Боба"),
        ChatMessage("assistant", "Ответ Бобу"),
    ]
    assert harness.sent_texts(BOB) == ["Ответ Бобу"]


async def test_temperature_of_one_user_does_not_affect_another(harness, llm):
    # Arrange
    await harness.send(ALICE, "/settings 1.0")
    # Act
    await harness.send(BOB, "Вопрос")
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert [call["temperature"] for call in llm.calls] == [0.3, 1.0]


# 3. Сокращение истории без потери текущего запроса и нарушения порядка.


async def test_long_history_is_trimmed_from_oldest_keeping_current_message(store):
    # Arrange
    limits = HistoryLimits(max_messages=4, max_chars=4105)
    llm = FakeLLM()
    harness = BotHarness(llm, store, limits)
    store.history[ALICE] = [
        ChatMessage("user" if i % 2 == 0 else "assistant", f"сообщение {i}") for i in range(10)
    ]
    current = "в" * 4080
    # Act
    await harness.send(ALICE, current)
    # Assert
    sent = llm.calls[0]["messages"]
    assert sent[0].role == "system"
    assert sent[-1] == ChatMessage("user", current)
    # Бюджет 4105 − 4080 = 25 символов: помещается только последняя пара (22 символа).
    assert sent[1:-1] == [
        ChatMessage("user", "сообщение 8"),
        ChatMessage("assistant", "сообщение 9"),
    ]


async def test_history_is_limited_by_message_count_and_starts_with_user(store):
    # Arrange
    limits = HistoryLimits(max_messages=3, max_chars=12000)
    llm = FakeLLM()
    harness = BotHarness(llm, store, limits)
    store.history[ALICE] = [
        ChatMessage("user" if i % 2 == 0 else "assistant", f"сообщение {i}") for i in range(6)
    ]
    # Act
    await harness.send(ALICE, "новый вопрос")
    # Assert: из трёх последних отброшен ответ ассистента без своего вопроса.
    assert llm.calls[0]["messages"][1:] == [
        ChatMessage("user", "сообщение 4"),
        ChatMessage("assistant", "сообщение 5"),
        ChatMessage("user", "новый вопрос"),
    ]


# 4. Смена режима очищает историю и сохраняет новую настройку.


@pytest.mark.parametrize("mode", [TRANSLATE, QUIZ, STUDY])
async def test_mode_switch_clears_history_and_saves_mode(harness, llm, store, mode):
    # Arrange
    await harness.send(ALICE, "Вопрос")
    await harness.send(BOB, "Вопрос Боба")
    # Act
    await harness.send(ALICE, f"/{mode.command}")
    # Assert
    assert (await store.get_settings(ALICE)).mode == mode.command
    assert ALICE not in store.history
    assert len(store.history[BOB]) == 2
    assert len(llm.calls) == 2  # сама команда в модель не отправляется
    assert harness.sent_texts(ALICE)[-1] == (
        f"Режим «{mode.title}» включён. История предыдущего режима очищена.\n{mode.hint}"
    )


async def test_new_mode_prompt_is_used_after_switch(harness, llm):
    # Arrange
    await harness.send(ALICE, "/quiz")
    # Act
    await harness.send(ALICE, "словари в Python")
    # Assert
    assert llm.calls[0]["messages"] == [
        ChatMessage("system", QUIZ.system_prompt),
        ChatMessage("user", "словари в Python"),
    ]


# 5. /reset очищает только историю вызвавшего пользователя.


async def test_reset_clears_only_caller_history_and_keeps_settings(harness, store):
    # Arrange
    await harness.send(ALICE, "/translate")
    await harness.send(ALICE, "/settings 0.7")
    await harness.send(ALICE, "Привет")
    await harness.send(BOB, "Вопрос Боба")
    # Act
    await harness.send(ALICE, "/reset")
    # Assert
    assert ALICE not in store.history
    assert store.history[BOB] == [
        ChatMessage("user", "Вопрос Боба"),
        ChatMessage("assistant", "Ответ модели"),
    ]
    settings = await store.get_settings(ALICE)
    assert (settings.mode, settings.temperature) == ("translate", 0.7)
    assert (
        harness.sent_texts(ALICE)[-1] == "История диалога очищена. Режим и temperature сохранены."
    )


async def test_answer_arriving_after_reset_is_not_saved_to_new_history(store):
    # Arrange: /reset выполнен, пока модель формировала ответ.
    class ResetDuringCall(FakeLLM):
        async def complete(self, messages, *, temperature, request_id):
            await store.clear_history(ALICE)
            return await super().complete(messages, temperature=temperature, request_id=request_id)

    harness = BotHarness(ResetDuringCall("Поздний ответ"), store, HistoryLimits(20, 12000))
    # Act
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert harness.sent_texts(ALICE) == ["Поздний ответ"]
    assert ALICE not in store.history


# 6. /settings принимает разрешённое и отклоняет некорректное значение.


async def test_settings_shows_mode_model_and_temperature(harness):
    # Arrange
    await harness.send(ALICE, "/translate")
    # Act
    await harness.send(ALICE, "/settings")
    # Assert
    message = harness.sent()[-1]
    assert message.text == (
        "Текущие настройки:\n"
        "Режим: Перевод (/translate)\n"
        "Модель: yandexgpt-lite/latest\n"
        "Temperature: 0.3\n\n"
        "Выберите temperature кнопкой ниже или командой /settings 0.7. "
        "Доступно: 0.0, 0.3, 0.7, 1.0."
    )
    buttons = message.reply_markup.inline_keyboard[0]
    assert [b.callback_data for b in buttons] == [
        "temperature:0.0",
        "temperature:0.3",
        "temperature:0.7",
        "temperature:1.0",
    ]


@pytest.mark.parametrize(("raw", "expected"), [("0.7", 0.7), ("0,0", 0.0), ("1", 1.0)])
async def test_settings_accepts_allowed_value_and_applies_it(harness, llm, store, raw, expected):
    # Arrange / Act
    await harness.send(ALICE, f"/settings {raw}")
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert (await store.get_settings(ALICE)).temperature == expected
    assert llm.calls[0]["temperature"] == expected
    assert harness.sent_texts(ALICE)[0] == (
        f"Temperature {expected:.1f} сохранена и применится со следующего запроса."
    )


@pytest.mark.parametrize("raw", ["0.5", "2", "-0.3", "abc", "nan", "inf", "0.7.1"])
async def test_settings_rejects_invalid_value_before_api_call(harness, llm, store, raw):
    # Arrange / Act
    await harness.send(ALICE, f"/settings {raw}")
    # Assert
    assert (await store.get_settings(ALICE)).temperature == 0.3
    assert llm.calls == []
    assert harness.sent_texts(ALICE) == [
        "Недопустимое значение temperature. Выберите одно из значений: 0.0, 0.3, 0.7 или 1.0."
    ]


async def test_settings_button_saves_value_and_forged_button_is_rejected(harness, store):
    # Arrange / Act
    await harness.press(ALICE, "temperature:1.0")
    await harness.press(ALICE, "temperature:9.9")
    # Assert
    assert (await store.get_settings(ALICE)).temperature == 1.0
    assert harness.sent_texts(ALICE) == [
        "Temperature 1.0 сохранена и применится со следующего запроса.",
        texts.TEMPERATURE_INVALID,
    ]


# 7. Ошибка LLM преобразуется в безопасное пользовательское сообщение.


@pytest.mark.parametrize("reason", list(LLMErrorReason))
async def test_llm_error_becomes_safe_message_and_is_not_saved(harness, llm, store, reason):
    # Arrange
    llm.responses = [LLMError(reason, status=500)]
    # Act
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert harness.sent_texts(ALICE) == [texts.LLM_ERRORS[reason]]
    assert store.history.get(ALICE, []) == []


async def test_timeout_message_text_is_exact(harness, llm):
    # Arrange
    llm.responses = [LLMError(LLMErrorReason.TIMEOUT)]
    # Act
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert harness.sent_texts(ALICE) == [
        "Модель не ответила вовремя. Попробуйте повторить запрос чуть позже."
    ]


async def test_unexpected_failure_gives_generic_message(harness, store):
    # Arrange
    async def broken(*args, **kwargs):
        raise ConnectionError("password=secret-db")

    store.load_history = broken
    # Act
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert harness.sent_texts(ALICE) == [
        "Не удалось обработать сообщение. Попробуйте ещё раз позже."
    ]


async def test_truncated_answer_is_marked(harness, llm, store):
    # Arrange
    llm.responses = [LLMResult("Начало ответа", finish_reason="length")]
    # Act
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert harness.sent_texts(ALICE) == [
        "Начало ответа\n\n[Ответ сокращён: модель достигла лимита длины ответа.]"
    ]
    assert store.history[ALICE][-1] == ChatMessage("assistant", "Начало ответа")


# 8. Ответ длиннее 4096 символов разбивается и отправляется без потери текста.


async def test_long_answer_is_split_into_ordered_parts(harness, llm):
    # Arrange
    paragraphs = [f"Абзац {i}. " + "текст " * 150 for i in range(12)]
    answer = "\n\n".join(paragraphs)
    llm.responses = [answer]
    # Act
    await harness.send(ALICE, "Расскажи подробно")
    # Assert
    parts = harness.sent_texts(ALICE)
    assert len(answer) > 4096
    assert len(parts) > 1
    assert all(0 < len(part) <= 4096 for part in parts)
    assert "".join(parts) == answer
    assert all(m.parse_mode is None for m in harness.sent())


# Прочие сценарии обработки сообщений.


async def test_start_lists_commands_and_current_mode(harness, llm):
    # Arrange / Act
    await harness.send(ALICE, "/start")
    # Assert
    assert llm.calls == []
    assert harness.sent_texts(ALICE) == [
        "Привет! Я AI-ассистент студента: отвечаю с помощью языковой модели и помню "
        "контекст нашего диалога.\n\n"
        "Режимы:\n"
        "/study — объяснение учебных вопросов по программированию\n"
        "/translate — перевод текста на указанный язык\n"
        "/quiz — вопросы по выбранной теме и проверка ваших ответов\n\n"
        "Настройки:\n"
        "/settings — текущий режим, модель и temperature\n"
        "/reset — очистить историю диалога\n\n"
        f"Сейчас включён режим «Учёба». {STUDY.hint}"
    ]


async def test_unknown_command_is_not_sent_to_llm(harness, llm):
    # Arrange / Act
    await harness.send(ALICE, "/weather")
    # Assert
    assert llm.calls == []
    assert harness.sent_texts(ALICE) == ["Неизвестная команда. Список команд — /start."]


async def test_non_text_message_is_not_sent_to_llm(harness, llm):
    # Arrange / Act
    await harness.send(ALICE, None)
    # Assert
    assert llm.calls == []
    assert harness.sent_texts(ALICE) == ["Я обрабатываю только текстовые сообщения."]


async def test_group_chat_is_ignored(harness, llm):
    # Arrange / Act
    await harness.send(-100500, "Привет всем", chat_type="supergroup")
    # Assert
    assert llm.calls == []
    assert harness.sent() == []


async def test_unknown_mode_from_database_falls_back_to_study(harness, llm, store):
    # Arrange
    store.settings[ALICE] = UserSettings(mode="removed")
    # Act
    await harness.send(ALICE, "Вопрос")
    # Assert
    assert llm.calls[0]["messages"][0] == ChatMessage("system", STUDY.system_prompt)
