from typing import Final

from litellm.exceptions import MidStreamFallbackError

_MARK_ATTRIBUTE: Final = "_litellm_unusable_tool_call_failure"


def mark_unusable_tool_call_failure(error: BaseException) -> None:
    """Tag a stream that ended on a tool call the model could not produce: the request and the model's
    output caused it, so the deployment that served it stays healthy."""
    setattr(error, _MARK_ATTRIBUTE, True)


def is_unusable_tool_call_failure(error: BaseException | None) -> bool:
    if isinstance(error, MidStreamFallbackError):
        return is_unusable_tool_call_failure(error.original_exception)
    return getattr(error, _MARK_ATTRIBUTE, False) is True


def carry_unusable_tool_call_mark(source: BaseException, mapped: BaseException) -> None:
    if is_unusable_tool_call_failure(source):
        mark_unusable_tool_call_failure(mapped)
