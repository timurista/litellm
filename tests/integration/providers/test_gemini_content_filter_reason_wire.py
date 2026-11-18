import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from integration._support.anthropic_sse import SseEvent, delta_text, event_type, event_types, parse_sse
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
)
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.routing.test_payment_required_cooldown import (
    NO_DEPLOYMENTS,
    cooldown_config,
    cooled,
    router_settings,
)
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

import litellm
from litellm.constants import DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS

_BACKEND: Final = "gemini-3.8-flash"
_OPENAI_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-gemini-key"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_ANSWER: Final = "Waves fold silver light into the shore."
_PARTIAL: Final = "Waves fold "
_OPENAI_ANSWER: Final = "Tides keep their own time."
_OPENAI_PROMPT: Final = "<scenario openai-plain>"
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_STREAM_SECONDS: Final = 60.0
_JSON_LIST: Final = TypeAdapter(list[JsonValue])

Provider: TypeAlias = Literal["gemini", "vertex_ai"]
_Chunks: TypeAlias = tuple[Mapping[str, JsonValue], ...]
_PROVIDERS: Final[tuple[Provider, ...]] = ("gemini", "vertex_ai")


@dataclass(frozen=True, slots=True)
class _Upstream:
    tag: str
    outcome: Mapping[str, JsonValue]
    usage: Mapping[str, int]
    frames: tuple[Mapping[str, JsonValue], ...]
    text: str
    finish_reason: str
    reason_fields: Mapping[str, str]
    note: str | None = None

    @property
    def prompt(self) -> str:
        return f"<scenario {self.tag}>"

    def body(self) -> bytes:
        return json.dumps({**self.outcome, "usageMetadata": dict(self.usage)}).encode()

    def events(self) -> tuple[bytes, ...]:
        finished: Final = {**self.frames[-1], "usageMetadata": dict(self.usage)}
        return tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in (*self.frames[:-1], finished))

    def expected_usage(self) -> Mapping[str, int]:
        return {
            "prompt_tokens": self.usage["promptTokenCount"],
            "completion_tokens": self.usage.get("candidatesTokenCount", 0),
            "total_tokens": self.usage["totalTokenCount"],
        }


def _text_frame(text: str) -> Mapping[str, JsonValue]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}]}


def _noted(key: str, note: str | None) -> Mapping[str, JsonValue]:
    return {} if note is None else {key: note}


def _with_note_tag(tag: str, note: str | None) -> str:
    return tag if note is None else f"{tag}-with-note"


def _flagged(reason: str, note: str | None = None) -> _Upstream:
    candidate: Final[Mapping[str, JsonValue]] = {"finishReason": reason, "index": 0, **_noted("finishMessage", note)}
    outcome: Final[Mapping[str, JsonValue]] = {"candidates": [dict(candidate)]}
    return _Upstream(
        tag=_with_note_tag(f"flagged-{reason}", note),
        outcome=outcome,
        usage={"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16},
        frames=(_text_frame(_PARTIAL), outcome),
        text=_PARTIAL,
        finish_reason="content_filter",
        reason_fields={"native_finish_reason": reason},
        note=note,
    )


def _blocked(reason: str, note: str | None = None) -> _Upstream:
    feedback: Final[Mapping[str, JsonValue]] = {"blockReason": reason, **_noted("blockReasonMessage", note)}
    outcome: Final[Mapping[str, JsonValue]] = {"promptFeedback": dict(feedback)}
    return _Upstream(
        tag=_with_note_tag(f"blocked-{reason}", note),
        outcome=outcome,
        usage={"promptTokenCount": 9, "totalTokenCount": 9},
        frames=(outcome,),
        text="",
        finish_reason="content_filter",
        reason_fields={"block_reason": reason},
        note=note,
    )


def _finished_after_text(native: str, finish_reason: str, note: str | None = None) -> _Upstream:
    finish: Final[Mapping[str, JsonValue]] = {"finishReason": native, "index": 0, **_noted("finishMessage", note)}
    return _Upstream(
        tag=_with_note_tag(f"answered-{native}", note),
        outcome={"candidates": [{"content": {"role": "model", "parts": [{"text": _ANSWER}]}, **finish}]},
        usage={"promptTokenCount": 7, "candidatesTokenCount": 8, "totalTokenCount": 15},
        frames=(
            _text_frame(_PARTIAL),
            _text_frame(_ANSWER.removeprefix(_PARTIAL)),
            {"candidates": [dict(finish)]},
        ),
        text=_ANSWER,
        finish_reason=finish_reason,
        reason_fields={"native_finish_reason": native},
        note=note,
    )


_FLAGGED_REASONS: Final = ("RECITATION", "SAFETY", "IMAGE_SAFETY")
_FLAGGED: Final = MappingProxyType({reason: _flagged(reason) for reason in _FLAGGED_REASONS})
_FLAGGED_WITH_NOTE: Final = _flagged("SAFETY", "The candidate was blocked for safety.")
_BLOCKED: Final = _blocked("SAFETY")
_BLOCKED_WITH_NOTE: Final = _blocked("PROHIBITED_CONTENT", "The prompt was blocked.")
_ANSWERED: Final = _finished_after_text("STOP", "stop")
_MALFORMED: Final = _finished_after_text("MALFORMED_FUNCTION_CALL", "stop")
_MALFORMED_WITH_NOTE: Final = _finished_after_text(
    "MALFORMED_FUNCTION_CALL", "stop", "Malformed function call: print(default_api.lookup(city='Paris'"
)
_UPSTREAMS: Final = (
    *_FLAGGED.values(),
    _FLAGGED_WITH_NOTE,
    _BLOCKED,
    _BLOCKED_WITH_NOTE,
    _ANSWERED,
    _MALFORMED,
    _MALFORMED_WITH_NOTE,
)
_FILTERED: Final = tuple(
    pytest.param(upstream, id=upstream.tag)
    for upstream in (*_FLAGGED.values(), _FLAGGED_WITH_NOTE, _BLOCKED, _BLOCKED_WITH_NOTE)
)
_WITH_NOTE: Final = tuple(
    pytest.param(upstream, id=upstream.tag)
    for upstream in (_FLAGGED_WITH_NOTE, _BLOCKED_WITH_NOTE, _MALFORMED_WITH_NOTE)
)
_FILTERED_ON_BRIDGES: Final = tuple(
    pytest.param(upstream, id=upstream.tag) for upstream in (_FLAGGED["RECITATION"], _BLOCKED)
)
_STREAMED: Final = tuple(
    pytest.param(upstream, id=upstream.tag)
    for upstream in (
        _FLAGGED["RECITATION"],
        _FLAGGED_WITH_NOTE,
        _BLOCKED,
        _BLOCKED_WITH_NOTE,
        _ANSWERED,
        _MALFORMED,
        _MALFORMED_WITH_NOTE,
    )
)


def _openai_events() -> tuple[bytes, ...]:
    def chunk(delta: Mapping[str, JsonValue], finish_reason: str | None) -> bytes:
        frame: Final = {
            "id": "chatcmpl-scripted",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": _OPENAI_BACKEND,
            "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(frame)}\n\n".encode()

    return (
        chunk({"role": "assistant", "content": "Tides keep "}, None),
        chunk({"content": "their own time."}, None),
        chunk({}, "stop"),
        b"data: [DONE]\n\n",
    )


def _respond(request: Request) -> Reply:
    sent: Final = request.body.decode()
    if request.target.endswith("/chat/completions") and _OPENAI_PROMPT in sent:
        return Reply(content_type="text/event-stream", chunks=_openai_events())
    matched: Final = tuple(upstream for upstream in _UPSTREAMS if upstream.prompt in sent)
    if len(matched) != 1:
        return Reply(status=404, body=json.dumps({"error": f"no single scenario in {sent}"}).encode())
    if f"/models/{_BACKEND}:streamGenerateContent" in request.target:
        return Reply(content_type="text/event-stream", chunks=matched[0].events())
    if f"/models/{_BACKEND}:generateContent" in request.target:
        return Reply(body=matched[0].body())
    return Reply(status=404, body=json.dumps({"error": f"unexpected target {request.target}"}).encode())


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    models: Mapping[str, str]

    @property
    def model(self) -> str:
        return self.models["gemini"]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway) -> Iterator[_Rig]:
    with wire_server(_respond) as wire, session_gateway.scenario() as scenario:
        models: Final = MappingProxyType(
            {
                "gemini": scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY),
                "vertex_ai": scenario.model(
                    model=f"vertex_ai/{_BACKEND}",
                    api_base=f"{wire.url}{_MODEL_PATH}",
                    api_key=None,
                    vertex_project=_PROJECT,
                    vertex_location=_LOCATION,
                    vertex_credentials=service_account_json(_PROJECT, session_gateway.upstream_url.rstrip("/")),
                ),
                "openai": scenario.model(
                    model=f"openai/{_OPENAI_BACKEND}", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
                ),
            }
        )
        yield _Rig(session_gateway, models)


def _post(rig: _Rig, path: str, body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    response: Final = rig.gateway.request("POST", path, body)
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


def _chat(rig: _Rig, upstream: _Upstream, provider: Provider = "gemini") -> Mapping[str, JsonValue]:
    return _post(
        rig,
        "/v1/chat/completions",
        {"model": rig.models[provider], "messages": [{"role": "user", "content": upstream.prompt}], **_CACHE_BUST},
    )


def _stream(rig: _Rig, model: str, prompt: str) -> _Chunks:
    with rig.gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            **_CACHE_BUST,
        },
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
        timeout=_STREAM_SECONDS,
    ) as response:
        assert response.status_code == 200, response.read()
        events: Final = tuple(
            line.removeprefix("data: ") for line in response.iter_lines() if line.startswith("data: ")
        )
    assert events and events[-1] == "[DONE]", events
    return tuple(JSON_OBJECT.validate_json(event) for event in events[:-1])


def _streamed_choices(chunks: _Chunks) -> _Chunks:
    nested: Final = (_JSON_LIST.validate_python(chunk.get("choices", [])) for chunk in chunks)
    return tuple(JSON_OBJECT.validate_python(choice) for choice in chain.from_iterable(nested))


def _finish_choice(chunks: _Chunks) -> Mapping[str, JsonValue]:
    finished: Final = tuple(choice for choice in _streamed_choices(chunks) if choice.get("finish_reason"))
    assert len(finished) == 1, chunks
    return finished[0]


def _streamed_text(chunks: _Chunks) -> str:
    deltas: Final = (JSON_OBJECT.validate_python(choice.get("delta") or {}) for choice in _streamed_choices(chunks))
    return "".join(str(delta.get("content") or "") for delta in deltas)


def _assembled_choice_fields(chunks: _Chunks) -> object:
    assembled: Final = litellm.stream_chunk_builder(chunks=list(chunks))
    assert assembled is not None
    return getattr(assembled.choices[0], "provider_specific_fields", None)


def _first_choice(payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    choices: Final = payload["choices"]
    assert isinstance(choices, list) and len(choices) == 1, payload
    return JSON_OBJECT.validate_python(choices[0])


@pytest.mark.parametrize("reason", _FLAGGED_REASONS)
def test_regression_flagged_finish_reason_reaches_caller_as_native_finish_reason(rig: _Rig, reason: str) -> None:
    payload: Final = _chat(rig, _FLAGGED[reason])
    choice: Final = _first_choice(payload)
    assert choice.get("provider_specific_fields") == {"native_finish_reason": reason}, payload


def test_regression_blocked_prompt_reaches_caller_as_block_reason(rig: _Rig) -> None:
    payload: Final = _chat(rig, _BLOCKED)
    choice: Final = _first_choice(payload)
    assert choice.get("provider_specific_fields") == {"block_reason": "SAFETY"}, payload


@pytest.mark.parametrize("upstream", _FILTERED)
def test_guard_filtered_answer_keeps_content_filter_null_content_and_usage(rig: _Rig, upstream: _Upstream) -> None:
    payload: Final = _chat(rig, upstream)
    choice: Final = _first_choice(payload)
    assert choice["finish_reason"] == "content_filter", payload
    assert JSON_OBJECT.validate_python(choice["message"]).get("content") is None, payload
    usage: Final = JSON_OBJECT.validate_python(payload["usage"])
    assert {key: usage[key] for key in upstream.expected_usage()} == upstream.expected_usage(), payload


def test_guard_stop_answer_keeps_native_finish_reason_and_text(rig: _Rig) -> None:
    payload: Final = _chat(rig, _ANSWERED)
    choice: Final = _first_choice(payload)
    assert choice["finish_reason"] == "stop", payload
    assert JSON_OBJECT.validate_python(choice["message"])["content"] == _ANSWER, payload
    assert choice.get("provider_specific_fields") == {"native_finish_reason": "STOP"}, payload


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _STREAMED)
def test_non_streamed_choice_carries_only_google_reason_code(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    payload: Final = _chat(rig, upstream, provider)
    choice: Final = _first_choice(payload)
    assert choice["finish_reason"] == upstream.finish_reason, payload
    assert choice.get("provider_specific_fields") == upstream.reason_fields, payload


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _STREAMED)
def test_regression_streamed_finish_chunk_and_assembled_choice_carry_google_reason(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    chunks: Final = _stream(rig, rig.models[provider], upstream.prompt)
    assert _finish_choice(chunks).get("provider_specific_fields") == upstream.reason_fields, chunks
    assert _assembled_choice_fields(chunks) == upstream.reason_fields, chunks


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _STREAMED)
def test_regression_streamed_and_non_streamed_google_reason_agree(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    non_streamed: Final = _first_choice(_chat(rig, upstream, provider)).get("provider_specific_fields")
    streamed: Final = _finish_choice(_stream(rig, rig.models[provider], upstream.prompt)).get(
        "provider_specific_fields"
    )
    assert non_streamed is not None
    assert streamed == non_streamed


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _STREAMED)
def test_guard_streamed_text_finish_reason_and_usage_unchanged(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    chunks: Final = _stream(rig, rig.models[provider], upstream.prompt)
    assert _streamed_text(chunks) == upstream.text, chunks
    assert _finish_choice(chunks)["finish_reason"] == upstream.finish_reason, chunks
    usages: Final = tuple(JSON_OBJECT.validate_python(chunk["usage"]) for chunk in chunks if chunk.get("usage"))
    assert usages, chunks
    assert {key: usages[-1][key] for key in upstream.expected_usage()} == upstream.expected_usage(), chunks


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _WITH_NOTE)
def test_guard_google_free_text_note_never_reaches_the_non_streamed_response(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    assert upstream.note is not None
    payload: Final = _chat(rig, upstream, provider)
    assert upstream.note not in json.dumps(payload), payload


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("upstream", _WITH_NOTE)
def test_guard_google_free_text_note_never_reaches_streamed_chunks_or_the_assembled_response(
    rig: _Rig, upstream: _Upstream, provider: Provider
) -> None:
    assert upstream.note is not None
    chunks: Final = _stream(rig, rig.models[provider], upstream.prompt)
    assert [chunk for chunk in chunks if upstream.note in json.dumps(chunk)] == [], chunks
    assembled: Final = litellm.stream_chunk_builder(chunks=list(chunks))
    assert assembled is not None
    assert upstream.note not in assembled.model_dump_json(), chunks


def test_guard_openai_stream_without_choice_fields_gains_none(rig: _Rig) -> None:
    chunks: Final = _stream(rig, rig.models["openai"], _OPENAI_PROMPT)
    assert _streamed_text(chunks) == _OPENAI_ANSWER, chunks
    assert _finish_choice(chunks)["finish_reason"] == "stop", chunks
    assert all(choice.get("provider_specific_fields") is None for choice in _streamed_choices(chunks)), chunks
    assert _assembled_choice_fields(chunks) is None, chunks


@pytest.mark.parametrize("upstream", _FILTERED_ON_BRIDGES)
def test_guard_messages_endpoint_still_reports_refusal(rig: _Rig, upstream: _Upstream) -> None:
    payload: Final = _post(
        rig,
        "/v1/messages",
        {"model": rig.model, "max_tokens": 64, "messages": [{"role": "user", "content": upstream.prompt}]},
    )
    assert payload["stop_reason"] == "refusal", payload


@pytest.mark.parametrize("upstream", _FILTERED_ON_BRIDGES)
def test_guard_responses_endpoint_still_reports_content_filter_incomplete(rig: _Rig, upstream: _Upstream) -> None:
    payload: Final = _post(rig, "/v1/responses", {"model": rig.model, "input": upstream.prompt})
    assert payload["status"] == "incomplete", payload
    assert JSON_OBJECT.validate_python(payload["incomplete_details"])["reason"] == "content_filter", payload


Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
_Events: TypeAlias = tuple[SseEvent, ...]
_ToolCalls: TypeAlias = tuple[tuple[str, Mapping[str, JsonValue]], ...]

_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_ERROR_EVENT_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("messages", "responses")
_PATHS: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
)
_ERROR_ENDING: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "error then done", "messages": "error", "responses": "response.failed"}
)
_NORMAL_ENDING: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "done", "messages": "message_stop", "responses": "response.completed"}
)
_STOP_FINISH: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "stop", "messages": "end_turn", "responses": "completed"}
)
_TOOL_FINISH: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "tool_calls", "messages": "tool_use", "responses": "completed"}
)
_NO_RETRY_NO_CACHE: Final[Mapping[str, JsonValue]] = MappingProxyType({**_CACHE_BUST, "num_retries": 0})
_TOOL_NAME: Final = "lookup_city"
_TOOL_ARGUMENTS: Final[Mapping[str, JsonValue]] = MappingProxyType({"city": "Paris"})
_TOOL_PARAMETERS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
)
_TOOL_CALL_CANDIDATE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "content": {"role": "model", "parts": [{"functionCall": {"name": _TOOL_NAME, "args": dict(_TOOL_ARGUMENTS)}}]},
        "index": 0,
    }
)
_SECOND_ANSWER: Final = "Gulls trace the tide line."
_HEALTHY_ANSWER: Final = "Harbor lights answer the tide."
_PLAIN_PROMPT: Final = "<scenario no-tools-declared>"
_NOTE_MARKER: Final = "finish-message-marker-for-the-model-only"
_NOTE: Final = f"Malformed function call: {_NOTE_MARKER}"
_FAILED_TOOL_CALL_REASONS: Final = ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL")
_NORMAL_FINISH_REASONS: Final = ("STOP", "MALFORMED_RESPONSE", "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE")
_INPUT_PRICE: Final = 0.001
_OUTPUT_PRICE: Final = 0.002
_ROW_SECONDS: Final = 70
_COOLDOWN_THRESHOLD: Final = DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS
_MAX_COOLDOWN_REQUESTS: Final = 60
_REQUESTS_AFTER_COOLDOWN: Final = 5
_OWNED_PROXY: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)
_UPSTREAM_500_FRAME: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"error": {"code": 500, "message": "scripted upstream failure", "status": "INTERNAL"}}
)
_STATUS_ROWS: Final = 'SELECT status FROM "LiteLLM_SpendLogs" WHERE model_group=%s'
_SPEND_ROWS: Final = (
    'SELECT status, spend, prompt_tokens, completion_tokens, total_tokens FROM "LiteLLM_SpendLogs" WHERE model_group=%s'
)


class _SpendRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _answer_candidate(text: str, index: int = 0) -> Mapping[str, JsonValue]:
    return {"content": {"role": "model", "parts": [{"text": text}]}, "index": index}


def _finish_candidate(native: str, index: int = 0) -> Mapping[str, JsonValue]:
    return {"finishReason": native, "index": index}


def _frame(*candidates: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {"candidates": [dict(candidate) for candidate in candidates]}


def _text_then_finish(native: str, note: str | None = None) -> _Upstream:
    return replace(_finished_after_text(native, "stop", note), tag=_with_note_tag(f"text-then-finish-{native}", note))


def _text_inside_final_chunk(native: str) -> _Upstream:
    final: Final = {**_answer_candidate(_ANSWER.removeprefix(_PARTIAL)), **_finish_candidate(native)}
    return replace(
        _finished_after_text(native, "stop"),
        tag=f"text-inside-final-chunk-{native}",
        frames=(_text_frame(_PARTIAL), _frame(final)),
    )


def _whole_answer_in_one_chunk(native: str) -> _Upstream:
    answered: Final = _finished_after_text(native, "stop")
    return replace(answered, tag=f"whole-answer-in-one-chunk-{native}", frames=(answered.outcome,))


def _finish_without_text(native: str, note: str | None = None) -> _Upstream:
    answered: Final = _finished_after_text(native, "stop", note)
    return replace(
        answered,
        tag=_with_note_tag(f"no-text-{native}", note),
        outcome=answered.frames[-1],
        frames=answered.frames[-1:],
        text="",
    )


def _tool_call_then_finish(native: str) -> _Upstream:
    return replace(
        _finished_after_text(native, "stop"),
        tag=f"tool-call-then-finish-{native}",
        outcome=_frame({**_TOOL_CALL_CANDIDATE, **_finish_candidate(native)}),
        frames=(_frame(_TOOL_CALL_CANDIDATE), _frame(_finish_candidate(native))),
        text="",
    )


def _tool_call_inside_final_chunk(native: str) -> _Upstream:
    return replace(
        _finished_after_text(native, "stop"),
        tag=f"tool-call-inside-final-chunk-{native}",
        outcome=_frame({**_TOOL_CALL_CANDIDATE, **_finish_candidate(native)}),
        frames=(_frame({**_TOOL_CALL_CANDIDATE, **_finish_candidate(native)}),),
        text="",
    )


def _two_candidates(failing_index: int) -> _Upstream:
    finishes: Final = tuple(
        _finish_candidate("MALFORMED_FUNCTION_CALL" if index == failing_index else "STOP", index) for index in (0, 1)
    )
    return replace(
        _finished_after_text("STOP", "stop"),
        tag=f"two-candidates-index-{failing_index}-fails",
        outcome=_frame(
            {**_answer_candidate(_ANSWER, 0), **finishes[0]},
            {**_answer_candidate(_SECOND_ANSWER, 1), **finishes[1]},
        ),
        frames=(_frame(_answer_candidate(_ANSWER, 0), _answer_candidate(_SECOND_ANSWER, 1)), _frame(*finishes)),
    )


def _params(scripts: tuple[_Upstream, ...]) -> tuple[object, ...]:
    return tuple(pytest.param(script, id=script.tag) for script in scripts)


_TEXT_BEFORE_FAILURE: Final = (
    *map(_text_then_finish, _FAILED_TOOL_CALL_REASONS),
    *(_text_then_finish(native, _NOTE) for native in _FAILED_TOOL_CALL_REASONS),
    *map(_text_inside_final_chunk, _FAILED_TOOL_CALL_REASONS),
    *map(_whole_answer_in_one_chunk, _FAILED_TOOL_CALL_REASONS),
)
_NO_TEXT_FAILURE: Final = (
    *map(_finish_without_text, _FAILED_TOOL_CALL_REASONS),
    *(_finish_without_text(native, _NOTE) for native in _FAILED_TOOL_CALL_REASONS),
)
_REAL_TOOL_CALLS: Final = (
    _tool_call_then_finish("MALFORMED_FUNCTION_CALL"),
    _tool_call_inside_final_chunk("MALFORMED_FUNCTION_CALL"),
)
_NORMAL_FINISHES: Final = tuple(map(_text_then_finish, _NORMAL_FINISH_REASONS))
_WITH_FINISH_MESSAGE: Final = tuple(
    script for script in (*_TEXT_BEFORE_FAILURE, *_NO_TEXT_FAILURE) if script.note is not None
)
_TWO_CANDIDATES: Final = (_two_candidates(0), _two_candidates(1))
_NON_STREAMED_FAILURES: Final = (
    *map(_text_then_finish, _FAILED_TOOL_CALL_REASONS),
    *(_text_then_finish(native, _NOTE) for native in _FAILED_TOOL_CALL_REASONS),
    *_NO_TEXT_FAILURE,
)
_FAILURE_WITH_AND_WITHOUT_TEXT: Final = (
    _text_then_finish("MALFORMED_FUNCTION_CALL"),
    _finish_without_text("MALFORMED_FUNCTION_CALL"),
)
_HEALTHY_SCRIPT: Final = replace(
    _finished_after_text("STOP", "stop"),
    tag="healthy-answer",
    outcome=_frame({**_answer_candidate(_HEALTHY_ANSWER), **_finish_candidate("STOP")}),
    frames=(_text_frame(_HEALTHY_ANSWER), _frame(_finish_candidate("STOP"))),
    text=_HEALTHY_ANSWER,
)
_UPSTREAM_500_MID_STREAM: Final = replace(
    _finished_after_text("STOP", "stop"),
    tag="upstream-500-mid-stream",
    frames=(_text_frame(_PARTIAL), _UPSTREAM_500_FRAME),
    text=_PARTIAL,
)
_SCRIPTS: Final = (
    *_TEXT_BEFORE_FAILURE,
    *_NO_TEXT_FAILURE,
    *_REAL_TOOL_CALLS,
    *_NORMAL_FINISHES,
    *_TWO_CANDIDATES,
)


def _scripted_reply(script: _Upstream, target: str) -> Reply:
    if f"/models/{_BACKEND}:streamGenerateContent" in target:
        return Reply(content_type="text/event-stream", chunks=script.events())
    if f"/models/{_BACKEND}:generateContent" in target:
        return Reply(body=script.body())
    return Reply(status=404, body=json.dumps({"error": f"unexpected target {target}"}).encode())


def _respond_with_script(request: Request) -> Reply:
    sent: Final = request.body.decode()
    matched: Final = tuple(script for script in _SCRIPTS if script.prompt in sent)
    if len(matched) != 1:
        return Reply(status=404, body=json.dumps({"error": f"no single script in {sent}"}).encode())
    return _scripted_reply(matched[0], request.target)


def _respond_with_healthy_answer(request: Request) -> Reply:
    return _scripted_reply(_HEALTHY_SCRIPT, request.target)


def _respond_with_upstream_500(request: Request) -> Reply:
    return _scripted_reply(_UPSTREAM_500_MID_STREAM, request.target)


def _failing_only_requests_with_tools(script: _Upstream) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        declares_tools: Final = "tools" in JSON_OBJECT.validate_json(request.body)
        return _scripted_reply(script if declares_tools else _HEALTHY_SCRIPT, request.target)

    return respond


def _add_deployment(
    scenario: Scenario, provider: Provider, wire: Wire, *, group: str | None = None, identity: str | None = None
) -> str:
    model_info: Final[Mapping[str, JsonValue] | None] = None if identity is None else {"id": identity}
    match provider:
        case "gemini":
            return scenario.model(
                name=group,
                model_info=model_info,
                model=f"gemini/{_BACKEND}",
                api_base=wire.url,
                api_key=_API_KEY,
                input_cost_per_token=_INPUT_PRICE,
                output_cost_per_token=_OUTPUT_PRICE,
            )
        case "vertex_ai":
            return scenario.model(
                name=group,
                model_info=model_info,
                model=f"vertex_ai/{_BACKEND}",
                api_base=f"{wire.url}{_MODEL_PATH}",
                api_key=None,
                vertex_project=_PROJECT,
                vertex_location=_LOCATION,
                vertex_credentials=service_account_json(_PROJECT, scenario.gateway.upstream_url.rstrip("/")),
                input_cost_per_token=_INPUT_PRICE,
                output_cost_per_token=_OUTPUT_PRICE,
            )


@dataclass(frozen=True, slots=True)
class _ScriptedGemini:
    gateway: Gateway
    models: Mapping[Provider, str]
    wire: Wire


@pytest.fixture(scope="module")
def scripted_gemini(session_gateway: Gateway) -> Iterator[_ScriptedGemini]:
    with wire_server(_respond_with_script) as wire, session_gateway.scenario() as scenario:
        models: Final[Mapping[Provider, str]] = MappingProxyType(
            {provider: _add_deployment(scenario, provider, wire) for provider in _PROVIDERS}
        )
        yield _ScriptedGemini(session_gateway, models, wire)


def _tool_declaration(endpoint: Endpoint) -> JsonValue:
    parameters: Final = dict(_TOOL_PARAMETERS)
    match endpoint:
        case "chat":
            return {"type": "function", "function": {"name": _TOOL_NAME, "parameters": parameters}}
        case "messages":
            return {"name": _TOOL_NAME, "input_schema": parameters}
        case "responses":
            return {"type": "function", "name": _TOOL_NAME, "parameters": parameters}


def _stream_body(
    endpoint: Endpoint,
    model: str,
    prompt: str,
    *,
    fallback: str | None = None,
    candidate_count: int | None = None,
    declares_tools: bool = True,
) -> Mapping[str, JsonValue]:
    shared: Final[Mapping[str, JsonValue]] = {
        "model": model,
        "stream": True,
        **_NO_RETRY_NO_CACHE,
        **({} if fallback is None else {"fallbacks": [fallback]}),
        **({} if candidate_count is None else {"n": candidate_count}),
        **({"tools": [_tool_declaration(endpoint)]} if declares_tools else {}),
    }
    messages: Final[JsonValue] = [{"role": "user", "content": prompt}]
    match endpoint:
        case "chat":
            return {**shared, "messages": messages}
        case "messages":
            return {**shared, "max_tokens": 64, "messages": messages}
        case "responses":
            return {**shared, "input": prompt}


def _completion_body(
    endpoint: Endpoint,
    model: str,
    prompt: str,
    *,
    fallback: str | None = None,
    candidate_count: int | None = None,
    declares_tools: bool = True,
) -> Mapping[str, JsonValue]:
    streaming: Final = _stream_body(
        endpoint,
        model,
        prompt,
        fallback=fallback,
        candidate_count=candidate_count,
        declares_tools=declares_tools,
    )
    return {key: value for key, value in streaming.items() if key != "stream"}


def _send(
    gateway: Gateway, endpoint: Endpoint, body: Mapping[str, JsonValue], *, key: str | None = None
) -> httpx.Response:
    return gateway.client.post(
        _PATHS[endpoint],
        json=dict(body),
        headers={"Authorization": f"Bearer {gateway.key if key is None else key}"},
        timeout=_STREAM_SECONDS,
    )


def _stream_script(
    scripted: _ScriptedGemini, endpoint: Endpoint, provider: Provider, script: _Upstream
) -> httpx.Response:
    return _send(scripted.gateway, endpoint, _stream_body(endpoint, scripted.models[provider], script.prompt))


def _payloads(events: _Events) -> _Chunks:
    return tuple(event.data for event in events)


def _chat_ending(body: str, events: _Events) -> str:
    finished_with_done: Final = body.rstrip().endswith("data: [DONE]")
    last_is_error: Final = bool(events) and "error" in events[-1].data
    match (last_is_error, finished_with_done):
        case (True, True):
            return "error then done"
        case (False, True):
            return "done"
        case _:
            return "cut off"


def _ending(endpoint: Endpoint, body: str, events: _Events) -> str:
    match endpoint:
        case "chat":
            return _chat_ending(body, events)
        case "messages" | "responses":
            return event_types(events)[-1] if events else "empty"


def _error_frames(endpoint: Endpoint, events: _Events) -> _Events:
    match endpoint:
        case "chat":
            return tuple(event for event in events if "error" in event.data)
        case "messages":
            return tuple(event for event in events if event_type(event) == "error")
        case "responses":
            return tuple(event for event in events if event_type(event) in ("error", "response.failed"))


def _finishes(endpoint: Endpoint, events: _Events) -> tuple[str, ...]:
    match endpoint:
        case "chat":
            choices: Final = _streamed_choices(_payloads(events))
            return tuple(str(choice["finish_reason"]) for choice in choices if choice.get("finish_reason"))
        case "messages":
            deltas: Final = (object_value(event.data["delta"]) for event in events if event.event == "message_delta")
            return tuple(str(delta.get("stop_reason")) for delta in deltas)
        case "responses":
            ended: Final = tuple(
                event for event in events if event_type(event) in ("response.completed", "response.incomplete")
            )
            return tuple(str(object_value(event.data["response"])["status"]) for event in ended)


def _delivered_text(endpoint: Endpoint, events: _Events) -> str:
    match endpoint:
        case "chat":
            return _streamed_text(_payloads(events))
        case "messages":
            return delta_text(events)
        case "responses":
            deltas: Final = (event for event in events if event_type(event) == "response.output_text.delta")
            return "".join(str(event.data["delta"]) for event in deltas)


def _chat_tool_calls(events: _Events) -> _ToolCalls:
    deltas: Final = (object_value(choice.get("delta") or {}) for choice in _streamed_choices(_payloads(events)))
    calls: Final = chain.from_iterable(_JSON_LIST.validate_python(delta.get("tool_calls") or []) for delta in deltas)
    functions: Final = tuple(object_value(object_value(call)["function"]) for call in calls)
    name: Final = "".join(str(function.get("name") or "") for function in functions)
    arguments: Final = "".join(str(function.get("arguments") or "") for function in functions)
    return ((name, JSON_OBJECT.validate_json(arguments)),)


def _messages_tool_calls(events: _Events) -> _ToolCalls:
    starts: Final = (event for event in events if event.event == "content_block_start")
    blocks: Final = tuple(object_value(event.data["content_block"]) for event in starts)
    name: Final = "".join(str(block["name"]) for block in blocks if block["type"] == "tool_use")
    deltas: Final = (object_value(event.data["delta"]) for event in events if event.event == "content_block_delta")
    arguments: Final = "".join(str(delta.get("partial_json", "")) for delta in deltas)
    return ((name, JSON_OBJECT.validate_json(arguments)),)


def _responses_tool_calls(events: _Events) -> _ToolCalls:
    completed: Final = tuple(event for event in events if event_type(event) == "response.completed")
    assert len(completed) == 1, events
    output: Final = tuple(map(object_value, list_value(object_value(completed[0].data["response"])["output"])))
    return tuple(
        (str(item["name"]), JSON_OBJECT.validate_json(str(item["arguments"])))
        for item in output
        if item["type"] == "function_call"
    )


def _delivered_tool_calls(endpoint: Endpoint, events: _Events) -> _ToolCalls:
    match endpoint:
        case "chat":
            return _chat_tool_calls(events)
        case "messages":
            return _messages_tool_calls(events)
        case "responses":
            return _responses_tool_calls(events)


def _native_reason(script: _Upstream) -> str:
    return script.reason_fields["native_finish_reason"]


def _terminal_error_message(endpoint: Endpoint, events: _Events) -> str:
    terminal: Final = events[-1].data
    match endpoint:
        case "chat" | "messages":
            return str(object_value(terminal["error"])["message"])
        case "responses":
            return str(object_value(object_value(terminal["response"])["error"])["message"])


def _assert_error_ending(endpoint: Endpoint, response: httpx.Response, script: _Upstream) -> None:
    events: Final = parse_sse(response.text)
    observed: Final = (
        response.status_code,
        _ending(endpoint, response.text, events),
        _finishes(endpoint, events),
        _delivered_text(endpoint, events),
    )
    assert observed == (200, _ERROR_ENDING[endpoint], (), script.text), response.text
    assert _native_reason(script) in _terminal_error_message(endpoint, events), response.text
    assert _NOTE_MARKER not in response.text, response.text


def _assert_normal_ending(endpoint: Endpoint, response: httpx.Response, text: str, finish: str) -> None:
    events: Final = parse_sse(response.text)
    observed: Final = (
        response.status_code,
        _ending(endpoint, response.text, events),
        _error_frames(endpoint, events),
        _finishes(endpoint, events),
        _delivered_text(endpoint, events),
    )
    assert observed == (200, _NORMAL_ENDING[endpoint], (), (finish,), text), response.text


def _assert_500_naming_the_finish_reason(response: httpx.Response, script: _Upstream) -> None:
    assert response.status_code == 500, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert _native_reason(script) in str(error["message"]), response.text
    assert _NOTE_MARKER not in response.text, response.text


def _completed_answer(endpoint: Endpoint, payload: Mapping[str, JsonValue]) -> tuple[str, str]:
    match endpoint:
        case "chat":
            choice: Final = _first_choice(payload)
            return str(object_value(choice["message"]).get("content") or ""), str(choice["finish_reason"])
        case "messages":
            blocks: Final = tuple(map(object_value, list_value(payload["content"])))
            text: Final = "".join(str(block["text"]) for block in blocks if block["type"] == "text")
            return text, str(payload["stop_reason"])
        case "responses":
            items: Final = tuple(map(object_value, list_value(payload["output"])))
            messages: Final = (item for item in items if item["type"] == "message")
            parts: Final = chain.from_iterable(list_value(item["content"]) for item in messages)
            return "".join(str(object_value(part)["text"]) for part in parts), str(payload["status"])


def _completed_tool_calls(endpoint: Endpoint, payload: Mapping[str, JsonValue]) -> _ToolCalls:
    match endpoint:
        case "chat":
            calls: Final = list_value(object_value(_first_choice(payload)["message"])["tool_calls"])
            functions: Final = tuple(object_value(object_value(call)["function"]) for call in calls)
            return tuple(
                (str(function["name"]), JSON_OBJECT.validate_json(str(function["arguments"]))) for function in functions
            )
        case "messages":
            blocks: Final = tuple(map(object_value, list_value(payload["content"])))
            return tuple(
                (str(block["name"]), object_value(block["input"])) for block in blocks if block["type"] == "tool_use"
            )
        case "responses":
            items: Final = tuple(map(object_value, list_value(payload["output"])))
            return tuple(
                (str(item["name"]), JSON_OBJECT.validate_json(str(item["arguments"])))
                for item in items
                if item["type"] == "function_call"
            )


def _assert_completed(endpoint: Endpoint, response: httpx.Response, text: str, finish: str) -> None:
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload.get("error") is None, response.text
    assert _completed_answer(endpoint, payload) == (text, finish), response.text


def _caller_outcome(endpoint: Endpoint, response: httpx.Response) -> tuple[int, str]:
    if response.status_code != 200:
        return response.status_code, "error body"
    return 200, _ending(endpoint, response.text, parse_sse(response.text))


def _outcomes_before_or_after_the_failure_is_surfaced(
    endpoint: Endpoint, script: _Upstream
) -> frozenset[tuple[int, str]]:
    streamed: Final = frozenset(((200, _NORMAL_ENDING[endpoint]), (200, _ERROR_ENDING[endpoint])))
    answered_before_streaming: Final = endpoint == "chat" and script.text == ""
    return streamed | frozenset(((500, "error body"),)) if answered_before_streaming else streamed


def _generate_calls(wire: Wire) -> tuple[str, ...]:
    return tuple(request.target.split("?")[0].rsplit(":", 1)[-1] for request in wire.drain())


def _recorded_row(gateway: Gateway, model: str) -> _SpendRow:
    rows: Final = eventually(
        lambda: gateway.rows(_SPEND_ROWS, (model,)), lambda found: len(found) >= 1, seconds=_ROW_SECONDS
    )
    assert len(rows) == 1, rows
    return _SpendRow.model_validate(rows[0])


def _priced_stream_row(
    scripted: _ScriptedGemini, endpoint: Endpoint, provider: Provider, script: _Upstream
) -> tuple[httpx.Response, _SpendRow]:
    with scripted.gateway.scenario() as scenario:
        model: Final = _add_deployment(scenario, provider, scripted.wire)
        response: Final = _send(scripted.gateway, endpoint, _stream_body(endpoint, model, script.prompt))
        return response, _recorded_row(scripted.gateway, model)


def _assert_row_keeps_gemini_usage_and_cost(row: _SpendRow, script: _Upstream) -> None:
    prompt: Final = script.usage["promptTokenCount"]
    completion: Final = script.usage["candidatesTokenCount"]
    tokens: Final = (row.prompt_tokens, row.completion_tokens, row.total_tokens)
    assert tokens == (prompt, completion, script.usage["totalTokenCount"]), row
    assert row.spend == pytest.approx(prompt * _INPUT_PRICE + completion * _OUTPUT_PRICE), row


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_TEXT_BEFORE_FAILURE))
def test_regression_failed_tool_call_after_text_delivers_the_text_and_ends_in_an_error(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    _assert_error_ending(endpoint, response, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("script", _params(_NO_TEXT_FAILURE))
def test_regression_chat_failed_tool_call_without_text_answers_500_naming_the_finish_reason(
    scripted_gemini: _ScriptedGemini, script: _Upstream, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, "chat", provider, script)
    _assert_500_naming_the_finish_reason(response, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ERROR_EVENT_ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NO_TEXT_FAILURE))
def test_regression_failed_tool_call_without_text_ends_in_the_error_event(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    _assert_error_ending(endpoint, response, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_regression_failed_tool_call_is_recorded_as_a_failure_with_gemini_usage_and_cost(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    script: Final = _text_then_finish("MALFORMED_FUNCTION_CALL")
    response, row = _priced_stream_row(scripted_gemini, endpoint, provider, script)
    assert response.status_code == 200, response.text
    assert row.status == "failure", row
    _assert_row_keeps_gemini_usage_and_cost(row, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_regression_failed_tool_call_without_text_is_answered_by_the_fallback_deployment(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    script: Final = _finish_without_text("MALFORMED_FUNCTION_CALL")
    with (
        wire_server(_respond_with_script) as failing,
        wire_server(_respond_with_healthy_answer) as healthy,
        gateway.scenario() as scenario,
    ):
        primary: Final = _add_deployment(scenario, provider, failing)
        fallback: Final = _add_deployment(scenario, provider, healthy)
        response: Final = _send(gateway, endpoint, _stream_body(endpoint, primary, script.prompt, fallback=fallback))
        _assert_normal_ending(endpoint, response, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        assert (_generate_calls(failing), _generate_calls(healthy)) == (
            ("streamGenerateContent",),
            ("streamGenerateContent",),
        ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_REAL_TOOL_CALLS))
def test_guard_real_tool_call_before_a_malformed_finish_still_ends_as_a_tool_call(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    _assert_normal_ending(endpoint, response, "", _TOOL_FINISH[endpoint])
    assert _delivered_tool_calls(endpoint, parse_sse(response.text)) == ((_TOOL_NAME, dict(_TOOL_ARGUMENTS)),), (
        response.text
    )


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NORMAL_FINISHES))
def test_guard_finish_reasons_other_than_a_failed_tool_call_stay_a_normal_finish(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    _assert_normal_ending(endpoint, response, script.text, _STOP_FINISH[endpoint])


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_guard_clean_stop_is_recorded_as_a_success_with_the_same_usage_and_cost(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    script: Final = _text_then_finish("STOP")
    response, row = _priced_stream_row(scripted_gemini, endpoint, provider, script)
    assert response.status_code == 200, response.text
    assert row.status == "success", row
    _assert_row_keeps_gemini_usage_and_cost(row, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_WITH_FINISH_MESSAGE))
def test_guard_gemini_finish_message_never_reaches_the_caller(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _stream_script(scripted_gemini, endpoint, provider, script)
    allowed: Final = _outcomes_before_or_after_the_failure_is_surfaced(endpoint, script)
    assert _caller_outcome(endpoint, response) in allowed, (response.status_code, response.text)
    assert _delivered_text(endpoint, parse_sse(response.text)) == script.text, response.text
    assert _NOTE_MARKER not in response.text, response.text


def _complete_script(
    scripted: _ScriptedGemini, endpoint: Endpoint, provider: Provider, script: _Upstream
) -> httpx.Response:
    return _send(scripted.gateway, endpoint, _completion_body(endpoint, scripted.models[provider], script.prompt))


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NON_STREAMED_FAILURES))
def test_regression_non_streamed_failed_tool_call_answers_500_naming_the_finish_reason(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _complete_script(scripted_gemini, endpoint, provider, script)
    _assert_500_naming_the_finish_reason(response, script)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_regression_non_streamed_failed_tool_call_is_answered_by_the_fallback_deployment(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    with (
        wire_server(_respond_with_script) as failing,
        wire_server(_respond_with_healthy_answer) as healthy,
        gateway.scenario() as scenario,
    ):
        primary: Final = _add_deployment(scenario, provider, failing)
        fallback: Final = _add_deployment(scenario, provider, healthy)
        body: Final = _completion_body(endpoint, primary, script.prompt, fallback=fallback)
        response: Final = _send(gateway, endpoint, body)
        _assert_completed(endpoint, response, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        assert (_generate_calls(failing), _generate_calls(healthy)) == (
            ("generateContent",),
            ("generateContent",),
        ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_regression_non_streamed_failed_tool_call_is_recorded_as_a_failure(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = scripted_gemini.gateway
    with gateway.scenario() as scenario:
        model: Final = _add_deployment(scenario, provider, scripted_gemini.wire)
        response: Final = _send(gateway, endpoint, _completion_body(endpoint, model, script.prompt))
        rows: Final = eventually(
            lambda: gateway.rows(_STATUS_ROWS, (model,)), lambda found: len(found) >= 1, seconds=_ROW_SECONDS
        )
        assert [row["status"] for row in rows] == ["failure"], (rows, response.text)


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_guard_non_streamed_real_tool_call_with_a_malformed_finish_stays_a_tool_call(
    scripted_gemini: _ScriptedGemini, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _complete_script(scripted_gemini, endpoint, provider, _REAL_TOOL_CALLS[0])
    _assert_completed(endpoint, response, "", _TOOL_FINISH[endpoint])
    assert _completed_tool_calls(endpoint, JSON_OBJECT.validate_json(response.content)) == (
        (_TOOL_NAME, dict(_TOOL_ARGUMENTS)),
    ), response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_NORMAL_FINISHES))
def test_guard_non_streamed_finish_reasons_other_than_a_failed_tool_call_stay_a_normal_answer(
    scripted_gemini: _ScriptedGemini, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    response: Final = _complete_script(scripted_gemini, endpoint, provider, script)
    _assert_completed(endpoint, response, script.text, _STOP_FINISH[endpoint])


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("script", _params(_TWO_CANDIDATES))
def test_guard_non_streamed_one_of_two_candidates_ending_malformed_stays_200(
    scripted_gemini: _ScriptedGemini, script: _Upstream, provider: Provider
) -> None:
    body: Final = _completion_body("chat", scripted_gemini.models[provider], script.prompt, candidate_count=2)
    response: Final = _send(scripted_gemini.gateway, "chat", body)
    assert response.status_code == 200, response.text
    choices: Final = tuple(map(object_value, list_value(JSON_OBJECT.validate_json(response.content)["choices"])))
    answers: Final = tuple(
        (choice["index"], object_value(choice["message"])["content"], choice["finish_reason"]) for choice in choices
    )
    assert sorted(answers, key=str) == [(0, _ANSWER, "stop"), (1, _SECOND_ANSWER, "stop")], response.text


@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("script", _params(_TWO_CANDIDATES))
def test_guard_one_of_two_candidates_ending_malformed_does_not_end_the_stream_in_an_error(
    scripted_gemini: _ScriptedGemini, script: _Upstream, provider: Provider
) -> None:
    response: Final = _send(
        scripted_gemini.gateway,
        "chat",
        _stream_body("chat", scripted_gemini.models[provider], script.prompt, candidate_count=2),
    )
    events: Final = parse_sse(response.text)
    choices: Final = _streamed_choices(_payloads(events))

    def text_of(index: int) -> str:
        chosen: Final = (choice for choice in choices if choice.get("index") == index)
        return "".join(str(object_value(choice.get("delta") or {}).get("content") or "") for choice in chosen)

    observed: Final = (
        response.status_code,
        _ending("chat", response.text, events),
        _error_frames("chat", events),
        frozenset(_finishes("chat", events)),
        text_of(0),
        text_of(1),
    )
    assert observed == (200, "done", (), frozenset(("stop",)), _ANSWER, _SECOND_ANSWER), response.text


@dataclass(frozen=True, slots=True)
class _CooldownRig:
    gateway: Gateway
    cache: OwnedRedis


@pytest.fixture(scope="module")
def cooldown_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_CooldownRig]:
    directory: Final = tmp_path_factory.mktemp("gemini-failed-tool-call-cooldown")
    with ExitStack() as stack:
        upstream: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(directory))
        owned: Final = stack.enter_context(
            owned_proxy_process(
                upstream,
                directory,
                {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2"},
                config=cooldown_config(directory, router_settings(cache), prometheus=False),
            )
        )
        yield _CooldownRig(owned.gateway, cache)


def _new_group() -> str:
    return f"integration-gemini-cooldown-{uuid.uuid4().hex[:12]}"


def _requests_until_both_deployments_passed_the_threshold(
    gateway: Gateway,
    endpoint: Endpoint,
    body: Mapping[str, JsonValue],
    wires: tuple[Wire, Wire],
    *,
    key: str | None = None,
) -> tuple[httpx.Response, ...]:
    def send() -> Iterator[httpx.Response]:
        for response in (_send(gateway, endpoint, body, key=key) for _ in range(_MAX_COOLDOWN_REQUESTS)):
            yield response
            if NO_DEPLOYMENTS in response.text or _both_passed_the_threshold(wires):
                return

    return tuple(send())


def _both_passed_the_threshold(wires: tuple[Wire, Wire]) -> bool:
    return all(wire.received.qsize() > _COOLDOWN_THRESHOLD for wire in wires)


def _calls_and_cooldown_until_cooled(
    rig: _CooldownRig, endpoint: Endpoint, body: Mapping[str, JsonValue], wire: Wire, identity: str
) -> tuple[tuple[int, bool], ...]:
    def observe() -> tuple[int, bool]:
        _send(rig.gateway, endpoint, body)
        return wire.received.qsize(), cooled(rig.cache, identity)

    def send() -> Iterator[tuple[int, bool]]:
        for observed in (observe() for _ in range(_MAX_COOLDOWN_REQUESTS)):
            yield observed
            if observed[1]:
                return

    return tuple(send())


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_guard_failed_tool_calls_do_not_cool_the_group_for_a_request_without_tools(
    cooldown_rig: _CooldownRig, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = cooldown_rig.gateway
    group: Final = _new_group()
    respond: Final = _failing_only_requests_with_tools(script)
    with wire_server(respond) as first, wire_server(respond) as second, gateway.scenario() as scenario:
        _add_deployment(scenario, provider, first, group=group)
        _add_deployment(scenario, provider, second, group=group)
        burst: Final = _requests_until_both_deployments_passed_the_threshold(
            gateway, endpoint, _stream_body(endpoint, group, script.prompt), (first, second)
        )
        assert [response.text for response in burst if NO_DEPLOYMENTS in response.text] == []
        assert _both_passed_the_threshold((first, second)), (first.received.qsize(), second.received.qsize())
        answer: Final = _send(gateway, endpoint, _stream_body(endpoint, group, _PLAIN_PROMPT, declares_tools=False))
        _assert_normal_ending(endpoint, answer, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("script", _params(_FAILURE_WITH_AND_WITHOUT_TEXT))
def test_guard_non_streamed_failed_tool_calls_do_not_cool_the_group_for_a_second_client(
    cooldown_rig: _CooldownRig, script: _Upstream, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = cooldown_rig.gateway
    group: Final = _new_group()
    respond: Final = _failing_only_requests_with_tools(script)
    with wire_server(respond) as first, wire_server(respond) as second, gateway.scenario() as scenario:
        _add_deployment(scenario, provider, first, group=group)
        _add_deployment(scenario, provider, second, group=group)
        failing_client: Final = scenario.key(models=[group])
        second_client: Final = scenario.key(models=[group])
        burst: Final = _requests_until_both_deployments_passed_the_threshold(
            gateway,
            endpoint,
            _completion_body(endpoint, group, script.prompt),
            (first, second),
            key=failing_client,
        )
        assert [response.text for response in burst if NO_DEPLOYMENTS in response.text] == []
        assert _both_passed_the_threshold((first, second)), (first.received.qsize(), second.received.qsize())
        body: Final = _completion_body(endpoint, group, _PLAIN_PROMPT, declares_tools=False)
        answer: Final = _send(gateway, endpoint, body, key=second_client)
        _assert_completed(endpoint, answer, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])


@_OWNED_PROXY
@pytest.mark.parametrize("provider", _PROVIDERS)
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_guard_upstream_500_mid_stream_still_cools_the_deployment_at_the_failure_threshold(
    cooldown_rig: _CooldownRig, endpoint: Endpoint, provider: Provider
) -> None:
    gateway: Final = cooldown_rig.gateway
    group: Final = _new_group()
    identity: Final = f"failing-{uuid.uuid4().hex}"
    body: Final = _stream_body(endpoint, group, _PLAIN_PROMPT, declares_tools=False)
    with (
        wire_server(_respond_with_upstream_500) as failing,
        wire_server(_respond_with_healthy_answer) as healthy,
        gateway.scenario() as scenario,
    ):
        _add_deployment(scenario, provider, failing, group=group, identity=identity)
        _add_deployment(scenario, provider, healthy, group=group)
        history: Final = _calls_and_cooldown_until_cooled(cooldown_rig, endpoint, body, failing, identity)
        assert [calls for calls, is_cooled in history if is_cooled and calls < _COOLDOWN_THRESHOLD] == [], history
        assert history[-1][1], history
        calls_when_cooled: Final = failing.received.qsize()
        after: Final = tuple(_send(gateway, endpoint, body) for _ in range(_REQUESTS_AFTER_COOLDOWN))
        for response in after:
            _assert_normal_ending(endpoint, response, _HEALTHY_ANSWER, _STOP_FINISH[endpoint])
        assert failing.received.qsize() == calls_when_cooled, history
