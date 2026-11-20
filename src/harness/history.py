from dataclasses import replace

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart

OMITTED_TOOL_OUTPUT = "[Older tool output omitted to limit context.]"


def compact_history(
    messages: list[ModelMessage],
    *,
    max_tool_output_chars: int = 4_000,
    keep_recent_tool_returns: int = 2,
) -> list[ModelMessage]:
    """Mask oversized older tool results without changing stored conversation history."""
    tool_return_locations = [
        (message_index, part_index)
        for message_index, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        for part_index, part in enumerate(message.parts)
        if isinstance(part, ToolReturnPart)
    ]
    retained = set(
        tool_return_locations[-keep_recent_tool_returns:] if keep_recent_tool_returns else ()
    )

    compacted: list[ModelMessage] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            compacted.append(message)
            continue

        parts = []
        for part_index, part in enumerate(message.parts):
            location = (message_index, part_index)
            if (
                isinstance(part, ToolReturnPart)
                and location not in retained
                and isinstance(part.content, str)
                and len(part.content) > max_tool_output_chars
            ):
                part = replace(part, content=OMITTED_TOOL_OUTPUT)
            parts.append(part)
        compacted.append(replace(message, parts=parts))
    return compacted
