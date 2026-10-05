"""Клиент модели против локального HTTP-сервера, имитирующего OpenAI-совместимый API."""

import asyncio
import logging

import pytest
from aiohttp import web

from app.llm import ChatMessage, LLMError, LLMErrorReason, OpenAICompatibleClient

MESSAGES = [ChatMessage("system", "Секретная инструкция"), ChatMessage("user", "Личный вопрос")]


def completion(content, finish_reason="stop"):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


@pytest.fixture
async def api(unused_tcp_port):
    """Сервер отвечает по очереди заданными обработчиками и запоминает запросы."""
    responses = []
    requests = []

    async def handler(request: web.Request) -> web.StreamResponse:
        requests.append({"headers": dict(request.headers), "json": await request.json()})
        return await responses.pop(0)(request)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    clients = []

    def client(**kwargs):
        options = {
            "base_url": f"http://127.0.0.1:{unused_tcp_port}/v1/",
            "api_key": "llm-secret-key",
            "model": "gpt://b1gfolder/yandexgpt-lite/latest",
            "model_name": "yandexgpt-lite/latest",
            "project": "b1gfolder",
            "timeout": 2,
            "max_tokens": 100,
            "retry_delay": 0,
            **kwargs,
        }
        created = OpenAICompatibleClient(**options)
        clients.append(created)
        return created

    try:
        yield client, responses, requests
    finally:
        for created in clients:
            await created.close()
        await runner.cleanup()


def json_reply(body, status=200):
    async def respond(_request):
        return web.json_response(body, status=status)

    return respond


async def test_success_sends_roles_parameters_and_headers(api):
    # Arrange
    client, responses, requests = api
    responses.append(json_reply(completion("Готово")))
    # Act
    result = await client().complete(MESSAGES, temperature=0.7, request_id="req1")
    # Assert
    assert (result.text, result.finish_reason, result.total_tokens) == ("Готово", "stop", 17)
    assert requests[0]["json"] == {
        "model": "gpt://b1gfolder/yandexgpt-lite/latest",
        "messages": [
            {"role": "system", "content": "Секретная инструкция"},
            {"role": "user", "content": "Личный вопрос"},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    assert requests[0]["headers"]["Authorization"] == "Bearer llm-secret-key"
    assert requests[0]["headers"]["OpenAI-Project"] == "b1gfolder"


async def test_temporary_server_error_is_retried_once(api):
    # Arrange
    client, responses, requests = api
    responses += [json_reply({"error": "busy"}, status=503), json_reply(completion("Со второй"))]
    # Act
    result = await client().complete(MESSAGES, temperature=0.3, request_id="req2")
    # Assert
    assert result.text == "Со второй"
    assert len(requests) == 2


async def test_retries_are_limited(api):
    # Arrange
    client, responses, requests = api
    responses += [json_reply({}, status=500), json_reply({}, status=502)]
    # Act / Assert
    with pytest.raises(LLMError) as exc:
        await client(max_retries=1).complete(MESSAGES, temperature=0.3, request_id="req3")
    assert exc.value.reason is LLMErrorReason.UNAVAILABLE
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (401, LLMErrorReason.AUTH),
        (403, LLMErrorReason.AUTH),
        (400, LLMErrorReason.BAD_REQUEST),
        (404, LLMErrorReason.BAD_REQUEST),
    ],
)
async def test_client_errors_are_not_retried(api, status, reason):
    # Arrange
    client, responses, requests = api
    responses.append(json_reply({"error": {"message": "denied"}}, status=status))
    # Act / Assert
    with pytest.raises(LLMError) as exc:
        await client().complete(MESSAGES, temperature=0.3, request_id="req4")
    assert exc.value.reason is reason
    assert len(requests) == 1


async def test_rate_limit_after_retry_is_reported(api):
    # Arrange
    client, responses, _ = api
    responses += [json_reply({}, status=429), json_reply({}, status=429)]
    # Act / Assert
    with pytest.raises(LLMError) as exc:
        await client().complete(MESSAGES, temperature=0.3, request_id="req5")
    assert exc.value.reason is LLMErrorReason.RATE_LIMIT


async def test_timeout_is_reported(api):
    # Arrange
    client, responses, requests = api

    async def slow(_request):
        await asyncio.sleep(3)
        return web.json_response(completion("Поздно"))

    responses += [slow, slow]
    # Act / Assert
    with pytest.raises(LLMError) as exc:
        await client(timeout=0.2).complete(MESSAGES, temperature=0.3, request_id="req6")
    assert exc.value.reason is LLMErrorReason.TIMEOUT
    assert len(requests) == 2


async def test_unreachable_service_is_reported(unused_tcp_port):
    # Arrange
    client = OpenAICompatibleClient(
        base_url=f"http://127.0.0.1:{unused_tcp_port}/v1",
        api_key="key",
        model="model",
        model_name="model",
        max_retries=0,
    )
    try:
        # Act / Assert
        with pytest.raises(LLMError) as exc:
            await client.complete(MESSAGES, temperature=0.3, request_id="req7")
        assert exc.value.reason is LLMErrorReason.UNAVAILABLE
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (completion(""), LLMErrorReason.EMPTY),
        (completion("   \n"), LLMErrorReason.EMPTY),
        (completion(None), LLMErrorReason.EMPTY),
        ({"choices": []}, LLMErrorReason.BAD_RESPONSE),
        ({"result": "other format"}, LLMErrorReason.BAD_RESPONSE),
        (completion(["part"]), LLMErrorReason.BAD_RESPONSE),
    ],
)
async def test_empty_or_malformed_answer_is_rejected(api, body, reason):
    # Arrange
    client, responses, _ = api
    responses.append(json_reply(body))
    # Act / Assert
    with pytest.raises(LLMError) as exc:
        await client().complete(MESSAGES, temperature=0.3, request_id="req8")
    assert exc.value.reason is reason


async def test_non_json_answer_is_rejected(api):
    # Arrange
    client, responses, _ = api

    async def html(_request):
        return web.Response(text="<html>gateway</html>", content_type="text/html")

    responses.append(html)
    # Act / Assert
    with pytest.raises(LLMError) as exc:
        await client().complete(MESSAGES, temperature=0.3, request_id="req9")
    assert exc.value.reason is LLMErrorReason.BAD_RESPONSE


async def test_length_finish_reason_marks_result_truncated(api):
    # Arrange
    client, responses, _ = api
    responses.append(json_reply(completion("Обрезанный", finish_reason="length")))
    # Act
    result = await client().complete(MESSAGES, temperature=0.3, request_id="req10")
    # Assert
    assert result.truncated is True


async def test_logs_contain_request_id_and_model_but_not_dialog_or_key(api, caplog):
    # Arrange
    client, responses, _ = api
    responses += [json_reply({}, status=500), json_reply(completion("Ответ с личными данными"))]
    caplog.set_level(logging.INFO, logger="app.llm")
    # Act
    await client().complete(MESSAGES, temperature=0.3, request_id="req11")
    # Assert
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "req11" in logs and "yandexgpt-lite/latest" in logs and "токены" in logs
    for private in ["Секретная инструкция", "Личный вопрос", "личными данными", "llm-secret-key"]:
        assert private not in logs
    assert "b1gfolder" not in logs


async def test_error_text_does_not_contain_key_or_url(api):
    # Arrange
    client, responses, _ = api
    responses.append(json_reply({"error": "llm-secret-key"}, status=401))
    # Act / Assert
    with pytest.raises(LLMError) as exc:
        await client().complete(MESSAGES, temperature=0.3, request_id="req12")
    assert str(exc.value) == "auth (HTTP 401)"
