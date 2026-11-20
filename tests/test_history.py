from pydantic_ai.messages import ModelRequest, ToolReturnPart

from harness.history import OMITTED_TOOL_OUTPUT, compact_history


def test_compacts_old_large_tool_output_but_keeps_recent_output() -> None:
    history = [
        ModelRequest(parts=[ToolReturnPart(tool_name="lookup", content="x" * 4_001, tool_call_id="1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="lookup", content="y" * 4_001, tool_call_id="2")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="lookup", content="z" * 4_001, tool_call_id="3")]),
    ]

    compacted = compact_history(history, keep_recent_tool_returns=2)

    assert compacted[0].parts[0].content == OMITTED_TOOL_OUTPUT
    assert compacted[1].parts[0].content == "y" * 4_001
    assert compacted[2].parts[0].content == "z" * 4_001
    assert history[0].parts[0].content == "x" * 4_001


def test_preserves_short_tool_results_and_regular_messages() -> None:
    history = [
        ModelRequest(parts=[ToolReturnPart(tool_name="lookup", content="small", tool_call_id="1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="lookup", content="x" * 4_001, tool_call_id="2")]),
    ]

    compacted = compact_history(history, max_tool_output_chars=5, keep_recent_tool_returns=0)

    assert compacted[0].parts[0].content == "small"
    assert compacted[1].parts[0].content == OMITTED_TOOL_OUTPUT
    assert compacted[0] is not history[0]
