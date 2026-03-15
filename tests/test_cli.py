from ctagentopenai import cli


def test_cli_one_shot_mode_reads_key_and_prints_response(monkeypatch, capsys, tmp_path):
    key_file = tmp_path / "openai.key"
    key_file.write_text("test-key\n")

    class StubAgent:
        def __init__(self, api_key):
            self.api_key = api_key

        def inference_with_tools(self, user_input, api_key):
            assert self.api_key == "test-key"
            assert api_key == "test-key"
            assert user_input == "hello"
            return "hi there", "resp_1"

    monkeypatch.setattr(cli, "Agent", StubAgent)

    cli.main(["--api-key", str(key_file), "--input", "hello"])

    captured = capsys.readouterr()
    assert "Welcome to CTAgentOpenAI!" in captured.out
    assert "hi there" in captured.out
