"""Разбиение длинного ответа на сообщения Telegram."""

TELEGRAM_LIMIT = 4096


def _utf16_length(text: str) -> int:
    # Telegram считает длину в кодовых единицах UTF-16: эмодзи может занимать две.
    return len(text.encode("utf-16-le")) // 2


def _fitting_prefix(text: str, limit: int) -> int:
    """Наибольшее число символов text, укладывающееся в limit единиц UTF-16."""
    if _utf16_length(text) <= limit:
        return len(text)
    low, high = 0, min(len(text), limit)
    while low < high:
        middle = (low + high + 1) // 2
        if _utf16_length(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return low


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Делит текст на части не длиннее limit без потери символов и с сохранением порядка.

    Граница выбирается по абзацу, затем по строке, затем по пробелу; разделитель
    остаётся в конце предыдущей части. Части только из пробелов не отправляются,
    потому что Telegram не принимает пустые сообщения.
    """
    parts: list[str] = []
    rest = text
    while rest:
        # Хотя бы один символ за шаг: защита от зацикливания при крошечном limit.
        size = max(1, _fitting_prefix(rest, limit))
        if size < len(rest):
            window = rest[:size]
            for separator in ("\n\n", "\n", " "):
                position = window.rfind(separator)
                if position > 0:
                    size = position + len(separator)
                    break
        parts.append(rest[:size])
        rest = rest[size:]
    return [part for part in parts if part.strip()]
