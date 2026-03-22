# CTAgentOpenAI

This project is for experimenting with the OpenAI Python SDK while building up
agent behavior from first principles: inference, conversation loops, local
tools, and related workflow patterns. It also allowed me to try out Codex, as a contrast
to Claude and Kiro which I had more familiarity with.

I had a particular method of going about this, which was to see behavior at the limits
without needing to push up token usage and costs. So the codebase has some flags
that allow for exercising the boundaries of operation which one many never encounter
in a production agent, such as: intentionally forgetting recent context, using a 
poor performing model when summarizing.

While using Codex and VSCode, there was plenty of conversation to
explore what production agents might do (e.g. robust persistent memory) vs. what a scaled down
version would be for this toy agent (simple in-memory summarization only).

I could continue expansion, such as adding support for other cloud and local models, but I believe
the patterns will be similar at this level of abstraction and I'd rather move up the stack.
For example, [Claude Agent SDK](https://nader.substack.com/p/the-complete-guide-to-building-agents)
takes care of the agent loop and tooling, as does [pi-mono](https://github.com/badlogic/pi-mono/tree/main)
and [Strands](https://strandsagents.com/). Even no-code or little-code approaches such as Claude desktop
app with it's built-in skills means business outcomes can be achieved without having to use a coded
harness at all - just bring custom data and connectors to the table. My takeaway, even if i never touch
agentic-loop code again, is a hightened understanding of the lower-level so that I have more intuition
when working on business outcomes.


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

The repo currently explores three main agent topics.

### 1. Tool Calling

The base agent supports a conversational CLI with interactive, one-shot, and
file-driven multi-turn input. Inside a turn, the model can call local tools in
a model/tool/model loop.

Current local tools include:

- `favorite_color`
- `calculator`
- `get_time`
- `list_files`
- `read_file`
- `query_label` when a local FDA label corpus is enabled

Useful controls:

- `--input-file PATH` replays one user turn per line from a file
- `--info` prints per-turn context usage through the logger
- `--debug` enables deeper request/response diagnostics

### 2. Memory And Context Management

The repo compares server-managed conversation state with app-managed history and
makes context growth visible with per-turn token reporting.

Supported history modes:

- `server-managed`: the app sends `previous_response_id` and lets the API manage prior turns
- `local-last-n`: the app keeps only the last `N` turns in memory and resends those

When rolling summary is enabled, unsummarized turns stay visible until they are
compacted. After compaction, the summary replaces the older turns and the most
recent turns stay verbatim.

Memory controls:

- `--history-mode {server-managed,local-last-n}` selects the memory strategy
- `--last-n-turns N` sets how many prior turns are retained for `local-last-n`
- `--summary-trigger-turns N` enables rolling-summary compaction once stored turns exceed `N`
- `--summary-keep-recent-turns N` keeps the newest `N` turns raw while summarizing older ones
- `--summary-model MODEL` selects a model specifically for the summarization step
- `--summary-reasoning-effort LEVEL` selects the reasoning effort for the summarization step

Checked-in memory examples:

- `inputs/exercise-plan-turns.txt` is a general multi-turn example
- `inputs/last-n-forgetting-turns.txt` is tuned to show forgetting with short intermediate acknowledgements
- `inputs/summary-compaction-turns.txt` is tuned to force rolling-summary compaction quickly
- `inputs/keep-all-overflow-turns.txt` is intentionally oversized to stress a very large `local-last-n` setting

### 3. Retrieval And Knowledge Access

The current retrieval milestone adds an FDA drug-label corpus that can be built
locally, indexed into SQLite, and queried through a tool-facing interface.

Current retrieval capabilities:

- section-aware chunking of FDA-style label records
- alias resolution for generic and brand names
- `grep`-style free-text retrieval over the chunk corpus
- BM25 scoring over the same chunk set for comparison
- `query_label` tool integration for single-drug label questions

Retrieval controls:

- `--label-db PATH` enables the `query_label` tool against a local FDA label corpus
- `--label-retrieval-method {bm25,grep}` selects the retrieval backend for `query_label`

Checked-in retrieval examples:

- `inputs/fda-label-sample.json` is a small FDA-style label corpus for local retrieval demos
- `inputs/fda-label-eval.json` is a small eval set for comparing `grep` and BM25 retrieval

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
- when rolling summary is enabled, the memory stays contiguous: unsummarized
  turns remain in the prompt until they are replaced by summary text.
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

Run the rolling-summary demo with aggressive compaction:

```bash
uv run ctagentopenai-memory-lab --api-key .openai.key --input-file inputs/summary-compaction-turns.txt --history-mode local-last-n --last-n-turns 2 --summary-trigger-turns 3 --summary-keep-recent-turns 2
```

Run the same summary demo with a stronger summary model and reasoning setting:

```bash
uv run ctagentopenai-memory-lab --api-key .openai.key --input-file inputs/summary-compaction-turns.txt --history-mode local-last-n --last-n-turns 2 --summary-trigger-turns 3 --summary-keep-recent-turns 2 --summary-model gpt-5-mini --summary-reasoning-effort low
```

List or run the named memory-lab suite:

```bash
uv run ctagentopenai-memory-lab-suite --list
uv run ctagentopenai-memory-lab-suite --api-key .openai.key --case forgetting-last-n-2
uv run ctagentopenai-memory-lab-suite --api-key .openai.key --case summary-last-n-2-strong
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

Build a local FDA label corpus from the checked-in sample data:

```bash
uv run ctagentopenai-label-corpus build --db .artifacts/fda-labels.sqlite --source inputs/fda-label-sample.json
```

Query the local corpus directly with BM25:

```bash
uv run ctagentopenai-label-corpus query --db .artifacts/fda-labels.sqlite --drug ibuprofen --question "What does the label say about pregnancy?"
```

Compare `grep` and BM25 on the bundled eval set:

```bash
uv run ctagentopenai-label-corpus eval --db .artifacts/fda-labels.sqlite --eval-file inputs/fda-label-eval.json
```

Run the agent with the FDA label retrieval tool enabled:

```bash
uv run ctagentopenai --api-key .openai.key --label-db .artifacts/fda-labels.sqlite --input "What does the label say about pregnancy for ibuprofen?"
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
