import json
from types import SimpleNamespace

import pytest

from ctagentopenai.agent import (
    Agent,
    build_openai_tools,
    extract_openai_tool_calls,
    get_tool_by_name,
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


def make_response(output=None, output_text="", response_id="resp_1"):
    return SimpleNamespace(
        output=output or [],
        output_text=output_text,
        id=response_id,
        error=None,
        incomplete_details=None,
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


def test_run_agent_loop_exits_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    class StubOpenAI:
        def __init__(self, api_key):
            self.responses = FakeResponses([])

    monkeypatch.setattr("ctagentopenai.agent.OpenAI", StubOpenAI)

    agent = Agent("test-key")
    monkeypatch.setattr(agent, "non_empty_input", lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt))

    agent.run_agent_loop()

    captured = capsys.readouterr()
    assert "Exiting..." in captured.out
