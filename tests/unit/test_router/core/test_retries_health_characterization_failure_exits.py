"""
Characterization of the failures a router core does not hold against a deployment, on every path a
failure is recorded from: the early exits of ``deployment_callback_on_failure`` (nothing counted, no
cooldown), the per-caller exits of ``_should_run_cooldown_logic`` (counted, no cooldown), and what else
decides a cooldown: the ``retry-after`` header, ``litellm.allowed_fails`` and a team deployment's siblings.
"""

import asyncio
import email.utils
from datetime import datetime, timedelta, timezone
from typing import Final

import httpx
import pytest

import litellm
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.unusable_tool_call_failure import mark_unusable_tool_call_failure
from litellm.router_core import RouterCore
from litellm.router_utils.cooldown_handlers import mark_deployment_selection_failure
from litellm.types.router import AllowedFailsPolicy
from litellm.types.utils import BACKGROUND_RESPONSE_STATUS_POLL_CALL_ORIGIN
from tests.unit.test_router.core.retries_health_helpers import (
    ErrorBuilder,
    MakeCore,
    attempt,
    attempt_sync,
    cooled_down,
    fresh,
    not_found,
)
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    auth_error,
    deployment,
    rate_limit,
    server_error,
)
from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts, collect_texts_sync

MESSAGES: Final = [{"role": "user", "content": "hi"}]
TOKEN_EXCHANGE: Final = "http://127.0.0.1:9/token"
PER_USER_COPILOT: Final = {
    "model": "github_copilot/gpt-4o",
    "litellm_credential_name": "ghc",
    "github_copilot_auth_type": "per_user_oauth",
}


def group(
    *ids: str, **params: object
) -> list[dict[str, object]]:  # kwargs-ok: litellm_params  # mutable-ok: a model_list
    return [deployment("g", deployment_id, **params) for deployment_id in ids]


def selection_failure() -> Exception:
    error: Final = rate_limit()
    mark_deployment_selection_failure(error)
    return error


def unusable_tool_call_failure() -> litellm.InternalServerError:
    error: Final = server_error()
    mark_unusable_tool_call_failure(error)
    return error


def wrapped_unusable_tool_call_failure() -> MidStreamFallbackError:
    return MidStreamFallbackError(
        message="cut", model="gpt-4o", llm_provider="openai", original_exception=unusable_tool_call_failure()
    )


def server_error_worded_like_a_tool_call_failure() -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message="Gemini ended the stream with finish reason MALFORMED_FUNCTION_CALL instead of a usable tool call",
        llm_provider="vertex_ai",
        model="gemini-3.8-flash",
    )


def caller_auth_error() -> Exception:
    return litellm.CallerCredentialAuthenticationError(
        message="not connected", llm_provider="github_copilot", model="gpt-4o"
    )


def caller_rate_limit() -> Exception:
    return litellm.CallerCredentialRateLimitError(
        message="token exchange limited", llm_provider="github_copilot", model="gpt-4o"
    )


def rate_limited_with(headers: dict[str, str]) -> litellm.RateLimitError:  # mutable-ok: httpx's header mapping
    return litellm.RateLimitError(
        message="slow down", llm_provider="openai", model="gpt-4o", response=httpx.Response(429, headers=headers)
    )


async def settle() -> None:
    """Let the logging and recording tasks a finished stream scheduled run."""
    for _ in range(20):
        await asyncio.sleep(0)


# --- U1: deployment_callback_on_failure's early exits -----------------------------------------------


@pytest.mark.parametrize(
    ("builder", "params"),
    [
        (selection_failure, {}),
        (auth_error, {"token_exchange_endpoint": TOKEN_EXCHANGE}),
        (unusable_tool_call_failure, {}),
    ],
    ids=["selection-failure", "caller-scoped-oauth", "unusable-tool-call"],
)
@pytest.mark.asyncio
async def test_an_attempt_failure_outside_the_deployment_is_neither_counted_nor_cooled(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder, params: dict[str, object]
) -> None:
    core: Final = make_core(model_list=group("a", "b", **params), num_retries=0, allowed_fails=0, cooldown_time=60)
    host.answer("*", fresh(builder))

    await attempt(core)

    assert cooled_down(core, ["a", "b"]) == {}
    assert core.health().failures_this_minute("a") + core.health().failures_this_minute("b") == 0


@pytest.mark.parametrize(
    ("builder", "params"),
    [
        (selection_failure, {}),
        (auth_error, {"token_exchange_endpoint": TOKEN_EXCHANGE}),
        (unusable_tool_call_failure, {}),
    ],
    ids=["selection-failure", "caller-scoped-oauth", "unusable-tool-call"],
)
def test_a_sync_attempt_failure_outside_the_deployment_is_neither_counted_nor_cooled(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder, params: dict[str, object]
) -> None:
    core: Final = make_core(model_list=group("a", "b", **params), num_retries=0, allowed_fails=0, cooldown_time=60)
    host.answer("*", fresh(builder))

    attempt_sync(core)

    assert cooled_down(core, ["a", "b"]) == {}
    assert core.health().failures_this_minute("a") + core.health().failures_this_minute("b") == 0


@pytest.mark.asyncio
async def test_a_background_poll_404_on_an_attempt_is_neither_counted_nor_cooled(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, allowed_fails=0)
    host.answer("*", fresh(not_found))

    await attempt(core, metadata={INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_STATUS_POLL_CALL_ORIGIN})

    assert cooled_down(core, ["a", "b"]) == {}
    assert core.health().failures_this_minute("a") + core.health().failures_this_minute("b") == 0


@pytest.mark.asyncio
async def test_a_stream_failure_outside_the_deployment_still_counts_its_request(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("partial", error=selection_failure(), logged=True))
    core: Final = make_core(
        model_list=[deployment("g", "a", rpm=100, order=1), deployment("g", "b", order=2)],
        num_retries=0,
        allowed_fails=0,
    )

    stream: Final = await core.acompletion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
    with pytest.raises(litellm.RateLimitError):
        await collect_texts(stream)
    await settle()

    assert cooled_down(core, ["a", "b"]) == {}
    assert core.health().failures_this_minute("a") == 0
    # `async_deployment_callback_on_failure` makes none of the failure callback's exits.
    assert await core.health().usage("g") == (0, 2)


def test_a_sync_stream_failure_outside_the_deployment_is_neither_counted_nor_cooled(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer("openai/a", chat_stream("partial", error=selection_failure(), logged=True, sync=True))
    core: Final = make_core(
        model_list=[deployment("g", "a", order=1), deployment("g", "b", order=2)], num_retries=0, allowed_fails=0
    )

    stream: Final = core.completion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves completion
    with pytest.raises(litellm.RateLimitError):
        collect_texts_sync(stream)

    assert cooled_down(core, ["a", "b"]) == {}
    assert core.health().failures_this_minute("a") == 0


@pytest.mark.parametrize(
    "builder",
    [selection_failure, unusable_tool_call_failure, wrapped_unusable_tool_call_failure],
    ids=["selection-failure", "unusable-tool-call", "wrapped-unusable-tool-call"],
)
@pytest.mark.asyncio
async def test_the_failure_hook_counts_a_failure_outside_the_deployment_as_nothing(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, allowed_fails=0)
    logged: Final = {
        "exception": builder(),
        "litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": "a"}},
    }

    assert core.deployment_callback_on_failure(logged, None, None, None) is False  # pyright: ignore[reportAttributeAccessIssue]  # every core serves it
    assert core.health().failures_this_minute("a") == 0


@pytest.mark.parametrize(
    ("builder", "failures", "cooled"),
    [(unusable_tool_call_failure, 0, set()), (server_error_worded_like_a_tool_call_failure, 1, {"a"})],
    ids=["unusable-tool-call", "ordinary-500"],
)
@pytest.mark.asyncio
async def test_only_a_marked_tool_call_failure_keeps_a_stream_off_the_deployment(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder, failures: int, cooled: set[str]
) -> None:
    host.answer("openai/a", chat_stream("partial", error=builder(), logged=True))
    core: Final = make_core(
        model_list=[deployment("g", "a", order=1), deployment("g", "b", order=2)], num_retries=0, allowed_fails=0
    )

    stream: Final = await core.acompletion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
    with pytest.raises(litellm.InternalServerError):
        await collect_texts(stream)
    await settle()

    assert core.health().failures_this_minute("a") == failures
    assert set(cooled_down(core, ["a", "b"])) == cooled


@pytest.mark.parametrize(
    ("builder", "failures", "cooled"),
    [(unusable_tool_call_failure, 0, set()), (server_error_worded_like_a_tool_call_failure, 1, {"a"})],
    ids=["unusable-tool-call", "ordinary-500"],
)
def test_only_a_marked_tool_call_failure_keeps_a_sync_stream_off_the_deployment(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder, failures: int, cooled: set[str]
) -> None:
    host.answer("openai/a", chat_stream("partial", error=builder(), logged=True, sync=True))
    core: Final = make_core(
        model_list=[deployment("g", "a", order=1), deployment("g", "b", order=2)], num_retries=0, allowed_fails=0
    )

    stream: Final = core.completion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves completion
    with pytest.raises(litellm.InternalServerError):
        collect_texts_sync(stream)

    assert core.health().failures_this_minute("a") == failures
    assert set(cooled_down(core, ["a", "b"])) == cooled


@pytest.mark.asyncio
async def test_a_500_worded_like_a_tool_call_failure_still_counts_against_the_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a", order=1), deployment("g", "b", order=2)], num_retries=0, allowed_fails=0
    )
    host.answer("*", fresh(server_error_worded_like_a_tool_call_failure))

    await attempt(core)

    assert core.health().failures_this_minute("a") == 1
    assert "a" in cooled_down(core, ["a", "b"])


@pytest.mark.asyncio
async def test_an_unusable_tool_call_failure_is_still_retried_and_falls_back(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("f", "z")], num_retries=1, fallbacks=[{"g": ["f"]}]
    )
    host.answer("openai/a", fresh(unusable_tool_call_failure))

    await attempt(core)

    assert host.deployments_called("acompletion") == ["a", "a", "z"]


@pytest.mark.parametrize(
    ("builder", "cooled"),
    [(unusable_tool_call_failure, set()), (server_error_worded_like_a_tool_call_failure, {"a"})],
    ids=["unusable-tool-call", "ordinary-500"],
)
def test_a_fallback_hop_blocked_from_logging_cools_down_only_for_an_ordinary_failure(
    make_core: MakeCore, builder: ErrorBuilder, cooled: set[str]
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, allowed_fails=0)
    logging_obj: Final = Logging(
        model="gpt-4o",
        messages=MESSAGES,
        stream=False,
        call_type="acompletion",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.model_call_details["current_attempt_api_call_start_time"] = datetime(2026, 1, 1)
    logging_obj.mark_logging_complete("async_failure")
    logging_obj.model_call_details["current_attempt_api_call_start_time"] = datetime(2026, 1, 3)
    error: Final = builder()
    error.failed_deployment_id = "a"

    core.trigger_fallback_cooldown({"litellm_logging_obj": logging_obj}, error, generic=False, own_loop=True)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves it

    assert set(cooled_down(core, ["a", "b"])) == cooled


# --- N13: _should_run_cooldown_logic's per-caller exits ---------------------------------------------


@pytest.mark.parametrize("builder", [caller_auth_error, caller_rate_limit], ids=["authentication", "rate-limit"])
@pytest.mark.asyncio
async def test_a_caller_credential_error_is_counted_but_never_cools_a_shared_deployment(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, allowed_fails=0)
    host.answer("*", fresh(builder))

    await attempt(core)

    assert cooled_down(core, ["a", "b"]) == {}
    assert core.health().failures_this_minute("a") + core.health().failures_this_minute("b") == 1


@pytest.mark.parametrize("builder", [auth_error, server_error], ids=["401", "500"])
@pytest.mark.asyncio
async def test_a_per_user_copilot_deployment_is_counted_but_never_cools_down(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "gh", weight=1, **PER_USER_COPILOT), deployment("g", "other", weight=0)],
        num_retries=0,
        allowed_fails=0,
    )
    host.answer("*", fresh(builder))

    await attempt(core)

    assert cooled_down(core, ["gh", "other"]) == {}
    assert core.health().failures_this_minute("gh") == 1


# --- S3: the retry-after header ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected_seconds"),
    [
        ({"retry-after-ms": "3000", "retry-after": "8"}, 8),
        ({"retry-after-ms": "7000"}, 5),
        ({"retry-after": "2.5"}, 5),
        ({"retry-after": "soon"}, 5),
        ({"retry-after": "-3"}, 5),
        ({"retry-after": " 7 "}, 7),
    ],
    ids=["seconds-over-ms", "ms-ignored", "float-ignored", "unreadable", "negative", "padded-int"],
)
@pytest.mark.asyncio
async def test_a_cooldown_takes_its_time_from_retry_after_read_as_an_int_only(
    make_core: MakeCore, host: ScriptedHost, headers: dict[str, str], expected_seconds: int
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, cooldown_time=5)
    host.answer("*", lambda **_: rate_limited_with(headers))

    await attempt(core)

    (entry,) = cooled_down(core, ["a", "b"]).values()
    assert entry["cooldown_time"] == expected_seconds


@pytest.mark.asyncio
async def test_a_cooldown_reads_a_retry_after_date(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, cooldown_time=5)
    until: Final = email.utils.format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)
    host.answer("*", lambda **_: rate_limited_with({"retry-after": until}))

    await attempt(core)

    (entry,) = cooled_down(core, ["a", "b"]).values()
    cooldown_time: Final = entry["cooldown_time"]
    assert isinstance(cooldown_time, int | float) and 100 <= cooldown_time <= 120


# --- U5: litellm.allowed_fails -----------------------------------------------------------------------


def weighted() -> list[dict[str, object]]:  # mutable-ok: a model_list
    return [deployment("g", "a", weight=1), deployment("g", "b", weight=0)]


@pytest.mark.asyncio
async def test_a_write_to_the_process_allowed_fails_starts_the_count_at_the_copied_value(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    core: Final = make_core(model_list=weighted(), num_retries=0)
    monkeypatch.setattr(litellm, "allowed_fails", 0)
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    assert core.settings().allowed_fails == 3
    assert cooled_down(core, ["a", "b"]) == {}


@pytest.mark.asyncio
async def test_a_process_allowed_fails_put_back_ends_the_count(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    core: Final = make_core(model_list=weighted(), num_retries=0)
    host.answer("*", fresh(rate_limit))
    monkeypatch.setattr(litellm, "allowed_fails", 0)
    await attempt(core)
    assert cooled_down(core, ["a", "b"]) == {}

    monkeypatch.setattr(litellm, "allowed_fails", 3)
    await attempt(core)

    assert list(cooled_down(core, ["a", "b"])) == ["a"]


@pytest.mark.asyncio
async def test_the_allowed_fails_count_is_the_process_value_copied_at_init(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "allowed_fails", 1)
    host.set_dynamic(allowed_fails=1)
    core: Final = make_core(
        model_list=weighted(), num_retries=0, allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=0)
    )
    host.answer("*", fresh(rate_limit))

    await attempt(core)
    assert cooled_down(core, ["a", "b"]) == {}

    await attempt(core)
    assert list(cooled_down(core, ["a", "b"])) == ["a"]


# --- U10: a team deployment's siblings --------------------------------------------------------------


def team_deployment(
    name: str, deployment_id: str, **info: object
) -> dict[str, object]:  # kwargs-ok: model_info  # mutable-ok: a model_list entry
    return deployment(
        name,
        deployment_id,
        weight=1 if deployment_id == "ta1" else 0,
        model_info={"team_id": "T1", "team_public_model_name": "tp", **info},
    )


async def call_team_model(core: RouterCore) -> object:
    return await attempt(core, "tp", metadata={"user_api_key_team_id": "T1"})


@pytest.mark.asyncio
async def test_a_team_deployment_with_a_sibling_under_its_public_name_cools_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[team_deployment("t1a", "ta1"), team_deployment("t1b", "ta2")], num_retries=0)
    host.answer("*", fresh(rate_limit))

    await call_team_model(core)

    assert list(cooled_down(core, ["ta1", "ta2"])) == ["ta1"]


@pytest.mark.parametrize("others", [(), (("t1b", "ta2", {"blocked": True}),)], ids=["alone", "blocked-sibling"])
@pytest.mark.asyncio
async def test_a_team_deployment_with_no_routable_sibling_stays_a_lone_deployment(
    make_core: MakeCore, host: ScriptedHost, others: tuple[tuple[str, str, dict[str, object]], ...]
) -> None:
    core: Final = make_core(
        model_list=[
            team_deployment("t1a", "ta1"),
            *(team_deployment(name, deployment_id, **info) for name, deployment_id, info in others),
        ],
        num_retries=0,
    )
    host.answer("*", fresh(rate_limit))

    await call_team_model(core)

    assert cooled_down(core, ["ta1", "ta2"]) == {}
