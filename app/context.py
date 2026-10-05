"""Сборка запроса к модели: инструкция режима, сокращённая история и текущее сообщение."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.llm import ChatMessage
from app.prompts import Mode


@dataclass(frozen=True)
class HistoryLimits:
    max_messages: int
    # Суммарная длина истории и текущего сообщения в символах. Инструкция и
    # few-shot-примеры режима постоянны и в этот бюджет не входят.
    max_chars: int


def trim_history(
    history: Sequence[ChatMessage], *, current: str, limits: HistoryLimits
) -> list[ChatMessage]:
    """Оставляет самый новый непрерывный участок истории, который помещается в лимиты.

    Сообщения отбрасываются только с начала, поэтому порядок не меняется и в
    середине не появляется пропусков. Если после сокращения история начинается
    с ответа ассистента, он тоже отбрасывается: каждая реплика модели остаётся
    рядом с вопросом, на который она отвечала.
    """
    budget = limits.max_chars - len(current)
    kept: list[ChatMessage] = []
    used = 0
    for message in reversed(history):
        if len(kept) >= limits.max_messages or used + len(message.content) > budget:
            break
        kept.append(message)
        used += len(message.content)
    kept.reverse()
    while kept and kept[0].role != "user":
        kept.pop(0)
    return kept


def build_messages(
    mode: Mode, history: Sequence[ChatMessage], current: str, limits: HistoryLimits
) -> list[ChatMessage]:
    return [
        *mode.prompt_messages(),
        *trim_history(history, current=current, limits=limits),
        ChatMessage("user", current),
    ]
