"""Тексты, которые бот показывает пользователю. Один тон, без технических деталей."""

from app.llm import LLMErrorReason
from app.prompts import MODES, Mode

TEMPERATURE_CHOICES = (0.0, 0.3, 0.7, 1.0)


def temperature_text(value: float) -> str:
    return f"{value:.1f}"


def start(mode: Mode) -> str:
    modes = "\n".join(f"/{m.command} — {m.description}" for m in MODES.values())
    return (
        "Привет! Я AI-ассистент студента: отвечаю с помощью языковой модели и помню "
        "контекст нашего диалога.\n\n"
        f"Режимы:\n{modes}\n\n"
        "Настройки:\n"
        "/settings — текущий режим, модель и temperature\n"
        "/reset — очистить историю диалога\n\n"
        f"Сейчас включён режим «{mode.title}». {mode.hint}"
    )


def mode_switched(mode: Mode) -> str:
    return f"Режим «{mode.title}» включён. История предыдущего режима очищена.\n{mode.hint}"


def settings(mode: Mode, model: str, temperature: float) -> str:
    choices = ", ".join(temperature_text(t) for t in TEMPERATURE_CHOICES)
    return (
        "Текущие настройки:\n"
        f"Режим: {mode.title} (/{mode.command})\n"
        f"Модель: {model}\n"
        f"Temperature: {temperature_text(temperature)}\n\n"
        f"Выберите temperature кнопкой ниже или командой /settings 0.7. Доступно: {choices}."
    )


def temperature_saved(value: float) -> str:
    return f"Temperature {temperature_text(value)} сохранена и применится со следующего запроса."


TEMPERATURE_INVALID = (
    "Недопустимое значение temperature. Выберите одно из значений: 0.0, 0.3, 0.7 или 1.0."
)
RESET_DONE = "История диалога очищена. Режим и temperature сохранены."
UNKNOWN_COMMAND = "Неизвестная команда. Список команд — /start."
TEXT_ONLY = "Я обрабатываю только текстовые сообщения."
INTERNAL_ERROR = "Не удалось обработать сообщение. Попробуйте ещё раз позже."
TRUNCATED_NOTICE = "\n\n[Ответ сокращён: модель достигла лимита длины ответа.]"

LLM_ERRORS = {
    LLMErrorReason.TIMEOUT: "Модель не ответила вовремя. Попробуйте повторить запрос чуть позже.",
    LLMErrorReason.RATE_LIMIT: (
        "Сейчас слишком много запросов к модели. Подождите немного и повторите."
    ),
    LLMErrorReason.UNAVAILABLE: "Сервис языковой модели временно недоступен. Попробуйте позже.",
    LLMErrorReason.AUTH: (
        "Бот не смог обратиться к языковой модели из-за ошибки настройки. "
        "Сообщите администратору бота."
    ),
    LLMErrorReason.BAD_REQUEST: (
        "Модель не смогла обработать запрос. Сократите сообщение или очистите историю "
        "командой /reset."
    ),
    LLMErrorReason.BAD_RESPONSE: "Модель вернула некорректный ответ. Попробуйте ещё раз.",
    LLMErrorReason.EMPTY: "Модель вернула пустой ответ. Попробуйте переформулировать запрос.",
}
