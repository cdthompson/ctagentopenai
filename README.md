# CTAgentOpenAI

This project is for experimenting with the OpenAI Python SDK while building up
agent behavior from first principles: inference, conversation loops, local
tools, and related workflow patterns.

## Requirements

- Python 3.9+
- `uv`
- an OpenAI API key saved to a local file such as `.openai.key`

## Setup

Install project and development dependencies:

```bash
uv sync --dev
```

## Test

Run the test suite:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=ctagentopenai
```

## Build

Build a source distribution and wheel:

```bash
uv build
```

Build artifacts are written to `dist/`.

## Features

Current agent capabilities:

- conversational CLI with interactive, one-shot, and file-driven multi-turn input
- local tool calling inside a model/tool/model loop
- explicit conversation-history control for comparing server-managed and app-managed memory
- per-turn context usage display as both a percentage and token counts
- visible handling of truncated responses when the model hits `max_output_tokens`

Conversation history behavior:

- `server-managed`: the app sends `previous_response_id` and lets the API manage prior turns
- `local-last-n`: the app keeps only the last `N` turns in memory and resends those, which makes forgetting behavior easy to observe

Flags that control this behavior:

- `--history-mode {server-managed,local-last-n}` selects the memory strategy
- `--last-n-turns N` sets how many prior turns are retained for `local-last-n`
- `--input-file PATH` replays one user turn per line from a file
- `--info` prints per-turn context usage through the logger in addition to the human-facing transcript
- `--debug` enables deeper request/response diagnostics

Checked-in example inputs:

- `inputs/exercise-plan-turns.txt` is a general multi-turn example
- `inputs/last-n-forgetting-turns.txt` is tuned to show forgetting with short intermediate acknowledgements
- `inputs/keep-all-overflow-turns.txt` is intentionally oversized to stress a very large `local-last-n` setting

## Limits And Failure Modes

The current CLI makes several different limits visible, and they mean different
things:

- `max_output_tokens` limit: the model may stop mid-answer if it uses up the
  allowed output budget for a single response. In the transcript this is shown
  as `[response truncated: reached max_output_tokens]`.
- organization TPM rate limit: a request can be rejected before inference if
  the combined token load is too large for the account's tokens-per-minute
  limit. In the transcript this is shown as
  `[request rejected: exceeded API rate limit or token budget; stopping further turns]`.
- context window pressure: the app shows estimated input-token usage per turn as
  a percentage of the configured model context window. This helps show when
  very large local replay windows or long conversations are becoming unrealistic even before a hard
  failure.

Practical implications:

- `local-last-n` is the simplest controlled memory policy and is useful for
  demonstrating forgetting.
- a very large `local-last-n` value is intentionally naive. It is useful for
  learning, but it will eventually run into either account limits or model-context limits.
- true long-run memory management belongs in the next milestone, where older
  turns can be summarized or selectively retained instead of replayed forever.

## CLI Usage

Run the CLI in one-shot mode:

```bash
uv run ctagentopenai --api-key .openai.key --input "What is 2 + 2?"
```

Run the interactive loop:

```bash
uv run ctagentopenai --api-key .openai.key
```

Enable debug logging:

```bash
uv run ctagentopenai --api-key .openai.key --debug
```

Run the memory lab against a scripted set of turns:

```bash
uv run ctagentopenai-memory-lab --api-key .openai.key --input-file scenarios/example.txt --history-mode local-last-n --last-n-turns 3
```

List or run the named memory-lab suite:

```bash
uv run ctagentopenai-memory-lab-suite --list
uv run ctagentopenai-memory-lab-suite --api-key .openai.key --case forgetting-last-n-2
```

Enable info logging for per-turn context usage:

```bash
uv run ctagentopenai --api-key .openai.key --info
```

Run with local conversation history and keep only the last 2 turns:

```bash
uv run ctagentopenai --api-key .openai.key --history-mode local-last-n --last-n-turns 2
```

Run a multi-turn session from a file where each line is one turn:

```bash
uv run ctagentopenai --api-key .openai.key --input-file inputs/exercise-plan-turns.txt
```

The checked-in `inputs/last-n-forgetting-turns.txt` file is designed to keep
intermediate responses short so you can observe `local-last-n` memory loss more
clearly on the final turn.

Available conversation history modes:

- `server-managed`: use `previous_response_id` and let the API manage history
- `local-last-n`: keep only the last `N` turns in memory and resend those

## Project Notes

- The package source lives under `src/ctagentopenai/`
- Tests live under `tests/`
- Roadmap and study notes live under `docs/`
