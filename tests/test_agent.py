import json
from types import SimpleNamespace

import openai

from ctagentopenai.agent import (
    Agent,
    AgentExecutionHalt,
    ConversationState,
    ConversationStrategy,
    DEFAULT_MODEL_CONFIG,
    SUMMARY_SYSTEM_PROMPT,
    TurnRecord,
    UsageSnapshot,
    build_openai_tools,
    build_system_prompt,
    extract_openai_tool_calls,
    get_model_config,
    get_tool_by_name,
    incomplete_response_notice,
    lowest_reasoning_effort,
)
from ctagentopenai.tool import (
    CalculatorTool,
    FavoriteColorTool,
    GetTimeTool,
    ListFilesTool,
    ReadFileTool,
    ToolCall,
)


class FakeResponses:
    def __init__(self, queued_responses):
        self.queued_responses = list(queued_responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.queued_responses.pop(0)


class FakeClient:
    def __init__(self, queued_responses):
        self.responses = FakeResponses(queued_responses)


def make_response(
    output=None,
    output_text="",
    response_id="resp_1",
    usage=None,
    incomplete_details=None,
):
    return SimpleNamespace(
        output=output or [],
        output_text=output_text,
        id=response_id,
        error=None,
        incomplete_details=incomplete_details,
        usage=usage
        or SimpleNamespace(input_tokens=123, output_tokens=12, total_tokens=135),
        model_dump_json=lambda indent=2: "{}",
    )


def test_favorite_color_tool_schema():
    tool = build_openai_tools([FavoriteColorTool()])[0]

    assert tool["type"] == "function"
    assert tool["name"] == "favorite_color"
    assert tool["parameters"]["additionalProperties"] is False
    assert tool["parameters"]["required"] == []


def test_list_files_tool_schema_requires_path():
    tool = build_openai_tools([ListFilesTool()])[0]

    assert tool["parameters"]["required"] == ["path"]


def test_calculator_tool_returns_bc_output(monkeypatch):
    monkeypatch.setattr(
        "ctagentopenai.tool.subprocess.check_output",
        lambda args, input: b"4\n",
    )

    result = CalculatorTool().invoke(
        ToolCall(tool_name="calculator", call_id="call_1", arguments={"calculation": "2 + 2"})
    )

    assert result.output == "4"
    assert result.is_error is False


def test_inference_wraps_string_input(monkeypatch):
    captured = {}

    class StubOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return make_response()

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent("test-key")
    response = agent.inference("hello")

    assert response.output_text == ""
    assert captured["api_key"] == "test-key"
    assert captured["kwargs"]["input"][0]["role"] == "system"
    assert captured["kwargs"]["input"][1] == {"role": "user", "content": "hello"}
    assert "Estimated current context window usage: 0/400000 tokens (0.00%)." in (
        captured["kwargs"]["input"][0]["content"]
    )
    assert captured["kwargs"]["max_output_tokens"] == DEFAULT_MODEL_CONFIG["default_max_output_tokens"]


def test_inference_logs_max_output_token_truncation(monkeypatch, caplog):
    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            return make_response(
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent("test-key")

    with caplog.at_level("WARNING"):
        response = agent.inference("hello")

    assert response.output_text == ""
    assert "The response was cut off because it exceeded the maximum output tokens." in (
        caplog.text
    )


def test_inference_raises_execution_halt_on_rate_limit(monkeypatch):
    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            response = SimpleNamespace(status_code=429, request=None, headers={})
            raise openai.RateLimitError(
                "too many tokens requested",
                response=response,
                body={"error": {"message": "too many tokens requested"}},
            )

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent("test-key")

    try:
        agent.inference("hello")
    except AgentExecutionHalt as exc:
        assert exc.user_message == (
            "[request rejected: exceeded API rate limit or token budget; stopping further turns]"
        )
    else:
        raise AssertionError("expected AgentExecutionHalt")


def test_incomplete_response_notice_formats_known_reasons():
    assert (
        incomplete_response_notice(SimpleNamespace(incomplete_details=SimpleNamespace(reason="max_output_tokens")))
        == "[response truncated: reached max_output_tokens]"
    )
    assert (
        incomplete_response_notice(SimpleNamespace(incomplete_details=SimpleNamespace(reason="content_filter")))
        == "[response truncated: content filtered]"
    )


def test_get_tool_by_name_returns_tool():
    tool = get_tool_by_name([FavoriteColorTool()], "favorite_color")

    assert tool.spec.name == "favorite_color"


def test_extract_openai_tool_calls_builds_internal_calls():
    response = make_response(
        output=[
            SimpleNamespace(
                type="function_call",
                name="favorite_color",
                arguments=json.dumps({}),
                call_id="call_1",
            )
        ]
    )

    tool_calls = extract_openai_tool_calls(response)

    assert tool_calls == [ToolCall(tool_name="favorite_color", call_id="call_1", arguments={})]


def test_get_time_tool_returns_timestamp():
    result = GetTimeTool().invoke(ToolCall(tool_name="get_time", call_id="call_1"))

    assert result.is_error is False
    assert result.output


def test_list_files_tool_lists_directory_contents(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()

    result = ListFilesTool().invoke(
        ToolCall(tool_name="list_files", call_id="call_1", arguments={"path": "."})
    )

    assert result.is_error is False
    assert result.output.splitlines() == ["nested/", "alpha.txt"]


def test_list_files_tool_rejects_paths_outside_project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    result = ListFilesTool().invoke(
        ToolCall(tool_name="list_files", call_id="call_1", arguments={"path": "../"})
    )

    assert result.is_error is True
    assert "escapes the current project" in result.output


def test_read_file_tool_reads_utf8_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    readme = tmp_path / "notes.txt"
    readme.write_text("hello\nworld\n", encoding="utf-8")

    result = ReadFileTool().invoke(
        ToolCall(tool_name="read_file", call_id="call_1", arguments={"path": "notes.txt"})
    )

    assert result.is_error is False
    assert result.output == "hello\nworld\n"


def test_read_file_tool_rejects_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()

    result = ReadFileTool().invoke(
        ToolCall(tool_name="read_file", call_id="call_1", arguments={"path": "docs"})
    )

    assert result.is_error is True
    assert "not a file" in result.output


def test_read_file_tool_rejects_paths_outside_project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    result = ReadFileTool().invoke(
        ToolCall(tool_name="read_file", call_id="call_1", arguments={"path": "../secret.txt"})
    )

    assert result.is_error is True
    assert "escapes the current project" in result.output


def test_inference_with_tools_executes_function_calls(monkeypatch):
    first_response = make_response(
        output=[
            SimpleNamespace(
                type="function_call",
                name="favorite_color",
                arguments=json.dumps({}),
                call_id="call_1",
            )
        ],
        response_id="resp_tool",
    )
    second_response = make_response(output=[], output_text="Your favorite color is blue.")

    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses([first_response, second_response])

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent("test-key")
    output_text, response_id = agent.inference_with_tools("What is my favorite color?")

    calls = agent.client.responses.calls
    assert output_text == "Your favorite color is blue."
    assert response_id == "resp_1"
    assert len(calls) == 2
    assert calls[1]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "blue",
    }


def test_build_system_prompt_includes_usage_and_strategy():
    prompt = build_system_prompt(
        "You are direct.",
        ConversationStrategy.LOCAL_LAST_N,
        UsageSnapshot(input_tokens=2000),
    )

    assert "Conversation mode: local-last-n." in prompt
    assert "2000/400000 tokens (0.50%)" in prompt


def test_usage_percentage_returns_fraction_of_context_window():
    from ctagentopenai.agent import usage_percentage

    assert usage_percentage(UsageSnapshot(input_tokens=2000)) == 0.5


def test_lowest_reasoning_effort_picks_first_supported_value():
    assert lowest_reasoning_effort(DEFAULT_MODEL_CONFIG) == "minimal"
    assert lowest_reasoning_effort({}) is None


def test_get_model_config_raises_for_unknown_model():
    try:
        get_model_config("unknown-model")
    except ValueError as exc:
        assert "Unsupported model" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_local_all_replays_full_transcript(monkeypatch):
    first_response = make_response(output_text="First answer", response_id="resp_1")
    second_response = make_response(output_text="Second answer", response_id="resp_2")

    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses([first_response, second_response])

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent("test-key", conversation_strategy=ConversationStrategy.LOCAL_LAST_N, last_n_turns=99)
    agent.inference_with_tools("First question")
    agent.inference_with_tools("Second question")

    second_call = agent.client.responses.calls[1]
    assert second_call["previous_response_id"] is None
    assert second_call["input"][1] == {"role": "user", "content": "First question"}
    assert second_call["input"][2] == {"role": "assistant", "content": "First answer"}
    assert second_call["input"][3] == {"role": "user", "content": "Second question"}


def test_local_last_n_limits_visible_turns(monkeypatch):
    responses = [
        make_response(output_text="Answer one", response_id="resp_1"),
        make_response(output_text="Answer two", response_id="resp_2"),
        make_response(output_text="Answer three", response_id="resp_3"),
    ]

    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses(responses)

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent(
        "test-key",
        conversation_strategy=ConversationStrategy.LOCAL_LAST_N,
        last_n_turns=1,
    )
    agent.inference_with_tools("Question one")
    agent.inference_with_tools("Question two")
    agent.inference_with_tools("Question three")

    third_call = agent.client.responses.calls[2]
    assert third_call["input"][1] == {"role": "user", "content": "Question two"}
    assert third_call["input"][2] == {"role": "assistant", "content": "Answer two"}
    assert third_call["input"][3] == {"role": "user", "content": "Question three"}
    assert {"role": "user", "content": "Question one"} not in third_call["input"]


def test_summary_compaction_updates_summary_and_reuses_it(monkeypatch):
    responses = [
        make_response(output_text="Answer one", response_id="resp_1"),
        make_response(output_text="Answer two", response_id="resp_2"),
        make_response(output_text="Answer three", response_id="resp_3"),
        make_response(output_text="Summary of older turns", response_id="resp_summary"),
        make_response(output_text="Answer four", response_id="resp_4"),
        make_response(output_text="Updated summary of older turns", response_id="resp_summary_2"),
    ]

    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses(responses)

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent(
        "test-key",
        conversation_strategy=ConversationStrategy.LOCAL_LAST_N,
        last_n_turns=2,
        summary_trigger_turns=2,
        summary_keep_recent_turns=1,
    )
    agent.inference_with_tools("Question one")
    agent.inference_with_tools("Question two")
    agent.inference_with_tools("Question three")
    agent.inference_with_tools("Question four")

    calls = agent.client.responses.calls
    summary_call = calls[3]
    fourth_turn_call = calls[4]
    second_summary_call = calls[5]
    assert "tools" not in summary_call
    assert summary_call["input"][0]["content"] == SUMMARY_SYSTEM_PROMPT
    assert summary_call["model"] == "gpt-5-nano"
    assert summary_call["reasoning"] == {"effort": "minimal"}
    assert summary_call["store"] is False
    assert "Question one" in summary_call["input"][1]["content"]
    assert "Answer one" in summary_call["input"][1]["content"]
    assert fourth_turn_call["input"][1] == {
        "role": "system",
        "content": "Summary of earlier conversation:\nSummary of older turns",
    }
    assert fourth_turn_call["input"][2] == {"role": "user", "content": "Question three"}
    assert fourth_turn_call["input"][3] == {"role": "assistant", "content": "Answer three"}
    assert fourth_turn_call["input"][4] == {"role": "user", "content": "Question four"}
    assert "Existing summary:\nSummary of older turns" in second_summary_call["input"][1]["content"]
    assert "Question three" in second_summary_call["input"][1]["content"]
    assert agent.conversation_state.summary_text == "Updated summary of older turns"
    assert agent.conversation_state.summarized_turn_count == 3


def test_summary_compaction_halts_on_incomplete_summary_response(monkeypatch):
    responses = [
        make_response(output_text="Answer one", response_id="resp_1"),
        make_response(output_text="Answer two", response_id="resp_2"),
        make_response(output_text="Answer three", response_id="resp_3"),
        make_response(
            output_text="",
            response_id="resp_summary",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        ),
    ]

    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses(responses)

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent(
        "test-key",
        conversation_strategy=ConversationStrategy.LOCAL_LAST_N,
        last_n_turns=2,
        summary_trigger_turns=2,
        summary_keep_recent_turns=1,
    )
    agent.inference_with_tools("Question one")
    agent.inference_with_tools("Question two")

    try:
        agent.inference_with_tools("Question three")
    except AgentExecutionHalt as exc:
        assert exc.user_message == (
            "[summary generation failed: response was truncated before summary text was produced]"
        )
    else:
        raise AssertionError("expected AgentExecutionHalt")


def test_summary_compaction_halts_on_empty_summary_text(monkeypatch):
    responses = [
        make_response(output_text="Answer one", response_id="resp_1"),
        make_response(output_text="Answer two", response_id="resp_2"),
        make_response(output_text="Answer three", response_id="resp_3"),
        make_response(output_text="   ", response_id="resp_summary"),
    ]

    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses(responses)

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent(
        "test-key",
        conversation_strategy=ConversationStrategy.LOCAL_LAST_N,
        last_n_turns=2,
        summary_trigger_turns=2,
        summary_keep_recent_turns=1,
    )
    agent.inference_with_tools("Question one")
    agent.inference_with_tools("Question two")

    try:
        agent.inference_with_tools("Question three")
    except AgentExecutionHalt as exc:
        assert exc.user_message == (
            "[summary generation failed: response contained no summary text]"
        )
    else:
        raise AssertionError("expected AgentExecutionHalt")


def test_agent_rejects_unsupported_summary_reasoning_effort():
    try:
        Agent(
            "test-key",
            summary_model="gpt-5-nano",
            summary_reasoning_effort="none",
        )
    except ValueError as exc:
        assert "Unsupported summary reasoning effort" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_visible_turns_excludes_summarized_turns():
    state = ConversationState(
        strategy=ConversationStrategy.LOCAL_LAST_N,
        last_n_turns=3,
        summary_trigger_turns=4,
        summary_keep_recent_turns=2,
    )
    state.turns = [
        TurnRecord("Q1", "A1", UsageSnapshot()),
        TurnRecord("Q2", "A2", UsageSnapshot()),
        TurnRecord("Q3", "A3", UsageSnapshot()),
        TurnRecord("Q4", "A4", UsageSnapshot()),
    ]
    state.summarized_turn_count = 2

    visible = state.visible_turns()

    assert [turn.user_message for turn in visible] == ["Q3", "Q4"]


def test_visible_turns_include_all_unsummarized_turns_when_summary_enabled():
    state = ConversationState(
        strategy=ConversationStrategy.LOCAL_LAST_N,
        last_n_turns=2,
        summary_trigger_turns=3,
        summary_keep_recent_turns=2,
    )
    state.turns = [
        TurnRecord("Q1", "A1", UsageSnapshot()),
        TurnRecord("Q2", "A2", UsageSnapshot()),
        TurnRecord("Q3", "A3", UsageSnapshot()),
        TurnRecord("Q4", "A4", UsageSnapshot()),
    ]
    state.summarized_turn_count = 1

    visible = state.visible_turns()

    assert [turn.user_message for turn in visible] == ["Q2", "Q3", "Q4"]
    assert state.unsummarized_turn_count() == 3


def test_server_managed_uses_previous_response_id(monkeypatch):
    first_response = make_response(output_text="First answer", response_id="resp_1")
    second_response = make_response(output_text="Second answer", response_id="resp_2")

    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses([first_response, second_response])

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent("test-key", conversation_strategy=ConversationStrategy.SERVER_MANAGED)
    agent.inference_with_tools("First question")
    agent.inference_with_tools("Second question")

    second_call = agent.client.responses.calls[1]
    assert second_call["previous_response_id"] == "resp_1"
    assert second_call["input"][1] == {"role": "user", "content": "Second question"}
    assert len(second_call["input"]) == 2


def test_multi_turn_exercise_plan_preserves_prior_constraints(monkeypatch):
    responses = [
        make_response(output_text="Initial plan", response_id="resp_1"),
        make_response(output_text="Back-safe update", response_id="resp_2"),
        make_response(output_text="Busy-schedule update", response_id="resp_3"),
        make_response(output_text="Beach-season update", response_id="resp_4"),
    ]

    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses(responses)

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent("test-key", conversation_strategy=ConversationStrategy.LOCAL_LAST_N, last_n_turns=99)
    turns = [
        "Create an exercise plan for a 48 year old male.",
        "He has a back injury preventing high-impact exercise.",
        "He also has a busy schedule with a teenage son.",
        "His goal is to stay fit and he wants spring and summer activities on the beach.",
    ]

    for turn in turns:
        agent.inference_with_tools(turn)

    fourth_call = agent.client.responses.calls[3]
    input_messages = fourth_call["input"]
    assert input_messages[1] == {"role": "user", "content": turns[0]}
    assert input_messages[2] == {"role": "assistant", "content": "Initial plan"}
    assert input_messages[3] == {"role": "user", "content": turns[1]}
    assert input_messages[4] == {"role": "assistant", "content": "Back-safe update"}
    assert input_messages[5] == {"role": "user", "content": turns[2]}
    assert input_messages[6] == {"role": "assistant", "content": "Busy-schedule update"}
    assert input_messages[7] == {"role": "user", "content": turns[3]}
