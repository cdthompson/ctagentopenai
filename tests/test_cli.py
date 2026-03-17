from ctagentopenai import cli, memory_lab, memory_lab_suite, runner
from ctagentopenai.agent import AgentExecutionHalt, ConversationStrategy


class StubConversationState:
    def __init__(
        self,
        usage,
        strategy=ConversationStrategy.SERVER_MANAGED,
        last_n_turns=3,
        turns=None,
        previous_response_id=None,
    ):
        self.latest_usage = usage
        self.strategy = strategy
        self.last_n_turns = last_n_turns
        self.turns = turns or []
        self.previous_response_id = previous_response_id

    def visible_turns(self):
        if self.strategy == ConversationStrategy.LOCAL_LAST_N:
            return self.turns[-self.last_n_turns :]
        return []

    def transcript_character_count(self):
        return sum(
            len(turn.user_message) + len(turn.assistant_message)
            for turn in self.visible_turns()
        )


class StubUsage:
    def __init__(self, input_tokens, output_tokens=0, total_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class StubIncompleteDetails:
    def __init__(self, reason):
        self.reason = reason


def test_cli_one_shot_mode_reads_key_and_prints_response(monkeypatch, capsys, tmp_path):
    key_file = tmp_path / "openai.key"
    key_file.write_text("test-key\n")
    init_args = {}

    class StubAgent:
        def __init__(self, api_key, conversation_strategy, last_n_turns):
            self.api_key = api_key
            init_args["conversation_strategy"] = conversation_strategy
            init_args["last_n_turns"] = last_n_turns
            self.conversation_state = StubConversationState(
                StubUsage(input_tokens=1000, output_tokens=100, total_tokens=1100),
                strategy=conversation_strategy,
                last_n_turns=last_n_turns,
            )
            self.last_response = type("StubResponse", (), {"incomplete_details": None})()

        def inference_with_tools(self, user_input):
            assert self.api_key == "test-key"
            assert user_input == "hello"
            return "hi there", "resp_1"

    monkeypatch.setattr(cli, "Agent", StubAgent)

    cli.main(
        [
            "--api-key",
            str(key_file),
            "--input",
            "hello",
            "--history-mode",
            "local-last-n",
            "--last-n-turns",
            "4",
        ]
    )

    output = capsys.readouterr()
    assert "Welcome to CTAgentOpenAI!" in output.out
    assert "[startup model=gpt-5-nano, history_mode=local-last-n, last_n_turns=4, context_window=400000, default_max_output_tokens=8192]" in output.out
    assert "hi there" in output.out
    assert "[context 0.25% | 1000/400000 input tokens | 100 output tokens | 1100 total tokens]" in output.out
    assert str(init_args["conversation_strategy"]) == "ConversationStrategy.LOCAL_LAST_N"
    assert init_args["last_n_turns"] == 4


def test_cli_input_file_runs_each_line_as_a_turn(monkeypatch, capsys, tmp_path):
    key_file = tmp_path / "openai.key"
    key_file.write_text("test-key\n")
    turns_file = tmp_path / "turns.txt"
    turns_file.write_text("first turn\nsecond turn\n", encoding="utf-8")

    captured = {}

    class StubAgent:
        def __init__(self, api_key, conversation_strategy, last_n_turns):
            self.api_key = api_key
            self.conversation_state = StubConversationState(
                StubUsage(input_tokens=0),
                strategy=conversation_strategy,
                last_n_turns=last_n_turns,
            )
            self.last_response = type("StubResponse", (), {"incomplete_details": None})()

        def inference_with_tools(self, user_input):
            captured.setdefault("turns", []).append(user_input)
            if user_input == "first turn":
                self.conversation_state.latest_usage = StubUsage(
                    input_tokens=1000,
                    output_tokens=100,
                    total_tokens=1100,
                )
                self.last_response = type("StubResponse", (), {"incomplete_details": None})()
                return "first reply", "resp_1"
            self.conversation_state.latest_usage = StubUsage(
                input_tokens=2000,
                output_tokens=120,
                total_tokens=2120,
            )
            self.last_response = type("StubResponse", (), {"incomplete_details": None})()
            return "second reply", "resp_2"

    monkeypatch.setattr(cli, "Agent", StubAgent)

    cli.main(["--api-key", str(key_file), "--input-file", str(turns_file)])

    output = capsys.readouterr()
    assert "Welcome to CTAgentOpenAI!" in output.out
    assert "[startup model=gpt-5-nano, history_mode=server-managed, context_window=400000, default_max_output_tokens=8192]" in output.out
    assert "[0.00%] > first turn" in output.out
    assert "first reply" in output.out
    assert "[context 0.25% | 1000/400000 input tokens | 100 output tokens | 1100 total tokens]" in output.out
    assert "[0.25%] > second turn" in output.out
    assert "second reply" in output.out
    assert "[context 0.50% | 2000/400000 input tokens | 120 output tokens | 2120 total tokens]" in output.out
    assert captured["turns"] == ["first turn", "second turn"]


def test_cli_prompt_and_usage_helpers_format_current_usage():
    agent = type(
        "StubAgent",
        (),
        {
            "conversation_state": StubConversationState(
                StubUsage(input_tokens=4000, output_tokens=50, total_tokens=4050)
            ),
            "last_response": type("StubResponse", (), {"incomplete_details": None})(),
        },
    )()

    assert runner.prompt_text(agent) == "[1.00%] > "
    assert runner.usage_text(agent) == (
        "[context 1.00% | 4000/400000 input tokens | 50 output tokens | 4050 total tokens]"
    )


def test_startup_summary_formats_history_mode_and_limits():
    agent = type(
        "StubAgent",
        (),
        {
            "conversation_state": type(
                "StubConversationStateWithMode",
                (),
                {"strategy": ConversationStrategy.LOCAL_LAST_N, "last_n_turns": 2},
            )(),
        },
    )()

    assert runner.startup_summary(agent) == (
        "[startup model=gpt-5-nano, history_mode=local-last-n, last_n_turns=2, context_window=400000, default_max_output_tokens=8192]"
    )


def test_print_agent_response_shows_incomplete_notice(capsys):
    agent = type(
        "StubAgent",
        (),
        {
            "conversation_state": StubConversationState(
                StubUsage(input_tokens=4000, output_tokens=50, total_tokens=4050)
            ),
            "last_response": type(
                "StubResponse",
                (),
                {"incomplete_details": StubIncompleteDetails("max_output_tokens")},
            )(),
        },
    )()

    runner.print_agent_response(agent, "partial reply")

    output = capsys.readouterr().out
    assert "partial reply" in output
    assert "[response truncated: reached max_output_tokens]" in output
    assert "[context 1.00% | 4000/400000 input tokens | 50 output tokens | 4050 total tokens]" in output


def test_run_agent_loop_exits_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    agent = type(
        "StubAgent",
        (),
        {
            "conversation_state": StubConversationState(StubUsage(0)),
            "last_response": type("StubResponse", (), {"incomplete_details": None})(),
        },
    )()
    runner.run_agent_loop(
        agent,
        input_fn=lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    output = capsys.readouterr()
    assert "Exiting..." in output.out


def test_run_turns_stops_on_agent_execution_halt(capsys):
    captured = []

    class StubAgent:
        def __init__(self):
            self.conversation_state = StubConversationState(StubUsage(0))
            self.last_response = type("StubResponse", (), {"incomplete_details": None})()

        def inference_with_tools(self, user_input):
            captured.append(user_input)
            if user_input == "second turn":
                raise AgentExecutionHalt(
                    "[request rejected: exceeded API rate limit or token budget; stopping further turns]"
                )
            self.conversation_state.latest_usage = StubUsage(
                input_tokens=1000,
                output_tokens=100,
                total_tokens=1100,
            )
            return "first reply", "resp_1"

    runner.run_turns(StubAgent(), ["first turn", "second turn", "third turn"])

    output = capsys.readouterr().out
    assert "first reply" in output
    assert "[request rejected: exceeded API rate limit or token budget; stopping further turns]" in output
    assert "third turn" not in output
    assert captured == ["first turn", "second turn"]


def test_cli_rejects_both_input_flags(tmp_path):
    key_file = tmp_path / "openai.key"
    key_file.write_text("test-key\n")
    turns_file = tmp_path / "turns.txt"
    turns_file.write_text("first turn\n", encoding="utf-8")

    try:
        cli.main(
            [
                "--api-key",
                str(key_file),
                "--input",
                "hello",
                "--input-file",
                str(turns_file),
            ]
        )
    except SystemExit as exc:
        assert str(exc) == "--input and --input-file cannot be used together"
    else:
        raise AssertionError("expected SystemExit")


def test_memory_lab_reports_memory_snapshots(monkeypatch, capsys, tmp_path):
    key_file = tmp_path / "openai.key"
    key_file.write_text("test-key\n")
    turns_file = tmp_path / "turns.txt"
    turns_file.write_text("first turn\n", encoding="utf-8")

    class StubTurn:
        def __init__(self, user_message, assistant_message):
            self.user_message = user_message
            self.assistant_message = assistant_message

    class StubAgent:
        def __init__(self, api_key, conversation_strategy, last_n_turns):
            self.api_key = api_key
            self.conversation_state = StubConversationState(
                StubUsage(input_tokens=0),
                strategy=conversation_strategy,
                last_n_turns=last_n_turns,
            )
            self.last_response = type("StubResponse", (), {"incomplete_details": None})()

        def inference_with_tools(self, user_input):
            self.conversation_state.turns.append(StubTurn(user_input, "first reply"))
            self.conversation_state.latest_usage = StubUsage(
                input_tokens=1000,
                output_tokens=100,
                total_tokens=1100,
            )
            self.conversation_state.previous_response_id = "resp_1"
            return "first reply", "resp_1"

    monkeypatch.setattr(memory_lab, "Agent", StubAgent)

    memory_lab.main(
        [
            "--api-key",
            str(key_file),
            "--input-file",
            str(turns_file),
            "--history-mode",
            "local-last-n",
            "--last-n-turns",
            "2",
        ]
    )

    output = capsys.readouterr().out
    assert "Welcome to CTAgentOpenAI memory lab!" in output
    assert "[memory strategy=local-last-n | stored_turns=0 | visible_turns=0 | transcript_chars=0 | last_input_tokens=0 | last_output_tokens=0 | previous_response_id=-]" in output
    assert "[memory strategy=local-last-n | stored_turns=1 | visible_turns=1 | transcript_chars=21 | last_input_tokens=1000 | last_output_tokens=100 | previous_response_id=resp_1]" in output


def test_memory_lab_suite_lists_named_cases(capsys):
    memory_lab_suite.main(["--list"])

    output = capsys.readouterr().out
    assert "Memory lab suite cases:" in output
    assert "- exercise-server:" in output
    assert "- forgetting-last-n-2:" in output
    assert "- overflow-last-n-99:" in output
