# CTAgentOpenAI

This project is for experimenting with the OpenAI Python SDK while building up
agent behavior from first principles: inference, conversation loops, local
tools, memory policies, retrieval, and related workflow patterns.

A large part of the exercise was to inspect behavior at the edges without
driving up token usage or cost. That led to features for intentionally
constrained memory, visible context pressure, naive versus improved retrieval,
and other controls that are more useful for learning than for production.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Test](#test)
- [Build](#build)
- [License](#license)
- [Features](#features)
- [Notes](#notes)
- [Edge-Case Demos](#edge-case-demos)
- [Limits And Failure Modes](#limits-and-failure-modes)
- [CLI Usage](#cli-usage)
- [My thoughts on production-ready architecture for agents](#my-thoughts-on-production-ready-architecture-for-agents)

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

## License

This repository's code is released under the MIT License. See [`LICENSE`](LICENSE).

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

- `inputs/fda-label-sample.json` is a small synthetic FDA-style label corpus for local retrieval demos
- `inputs/fda-label-eval.json` is a small eval set for comparing `grep` and BM25 retrieval

Data notes:

- the checked-in `inputs/fda-label-sample.json` file is a handcrafted demo corpus in the style of FDA/openFDA label data and is not an official FDA export
- the intended real-world source for this milestone is FDA/openFDA drug labeling data
- FDA/openFDA data is generally public domain / CC0; requested attribution is: `Data provided by the U.S. Food and Drug Administration (open.fda.gov)`

## Notes

I built this project with substantial help from Codex, both as a coding
assistant and as part of the learning goal. I already had experience working
with other AI-assisted development tools and model stacks; the new element here
was gaining direct hands-on experience with Codex and GPT-based coding
workflows.

I also built this at a relatively low level on purpose. The goal was to deepen
intuition for tool calling, memory, retrieval, context pressure, and evaluation
by working through the mechanics directly. In future production work, I would
usually prefer higher-level agent frameworks or managed tooling when they fit
the problem, rather than rebuilding the loop from scratch. Frameworks such as
[Claude Agent SDK](https://nader.substack.com/p/the-complete-guide-to-building-agents),
[Strands](https://strandsagents.com/), and similar higher-level platforms are
often the better choice once the underlying patterns are understood.

## Edge-Case Demos

Several examples in the repo are intentionally designed to exercise failure
modes or boundary behavior for learning purposes:

- `inputs/last-n-forgetting-turns.txt` shows how a small `local-last-n` window can cause the agent to forget earlier constraints
- `inputs/summary-compaction-turns.txt` forces rolling-summary compaction quickly so the summary boundary is easy to inspect
- `inputs/keep-all-overflow-turns.txt` is sized to stress unrealistic replay windows and make context pressure visible
- the `Limits And Failure Modes` section below explains how truncated outputs, rate limits, and context-window pressure show up in the CLI
- the FDA label retrieval demo and `inputs/fda-label-eval.json` make it easy to compare naive free-text retrieval with BM25 ranking over the same chunk corpus

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

The FDA label retrieval demo is for software experimentation and does not
provide medical advice.

The checked-in `inputs/last-n-forgetting-turns.txt` file is designed to keep
intermediate responses short so you can observe `local-last-n` memory loss more
clearly on the final turn.

Available conversation history modes:

- `server-managed`: use `previous_response_id` and let the API manage history
- `local-last-n`: keep only the last `N` turns in memory and resend those

## My thoughts on production-ready architecture for agents

If one were to be considering how a low-level agent built like this one would be promoted to a
more production-like experience, such as a web UI, there are a few immediate limitations to address:

1. **In-memory context**: Sessions would need to be persisted, as the compute layers could be intentionally
transient (Lambda) or unintentionally crash/die/age out and need replacement (EC2, Fargate).
I would be looking to cache layers such as Redis/ElastiCache, possibly backed by longer-term persistence in DynamoDB.
DynamoDB is fast enough that it could serve as both the primary session store and persistence layer, if the keys and indexing are designed for it.

2. **Retrieval**: Retrieval would obviously not be a local file such as sqlite. Again looking for persistence
that would outlast compute layer recycling. The storage would heavily depend upon the size of the data, update
frequency, and retrieval patterns. SQL-based storage, compatible with the sqlite queries in this repo, include
PostgreSQL/MySQL on RDS, preferably Aurora. OpenSearch would provide full-text search out of the box.

In any live service needing fast retrieval times, the raw data will need transformed from the origin into a format
ideal for querying, sometimes needing multiple indexes and denormalization. Small, well indexed data like the FDA 
label drug-facts we used, is easy to gather all relevant data into local working memory and run the BM25 search on
that subset but this is a rather unique case. 

3. **UX**: The CLI would need to be replaced by a web front-end if we're doing turn-based chat. Web assets could be served from
S3, `cli.py` replaced by an `api.py` implementing a web server such as flask on long-lived compute or even a single handler
function served up by Lambda.

4. **Authentication and Authorization**: Exposing anything as a service would require some auth of some sort. There are too many
to list here, but cannot be skipped. Okta, Cognito, standard OAuth...

5. **Logging and Monitoring**: Alarms on service metrics are table stakes, but some business metrics could be:
hit/miss rate on retrieval tool, memory compaction frequency and summary size, tool usage frequency, and all
could use model choice as a dimension to compare behavior between them. A good illustration is that, when
using the lowest reasoning level, the summary step produced a one-word summary that dropped any useful knowledge.

That covers the immediate needs to make it functional, but this still leaves several areas outside a full production design:
- Data pipeline for ingestion of updated retrieval data (e.g. new drugs approved by FDA)
- CI/CD to roll out updated service code and prompts
- Separate agent for response quality validation gating the answers sent to clients
