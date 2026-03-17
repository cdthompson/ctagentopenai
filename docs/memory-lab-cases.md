# Memory Lab Cases

This document collects the manual scenario suite for the memory lab. It is
separate from the README so the cases can evolve as the memory system evolves.

## Purpose

The memory lab is meant to make conversation-state behavior visible while
comparing different memory strategies. These cases are not white-box tests.
They are replayable demonstrations that let you inspect:

- how many turns are stored versus visible
- transcript growth over time
- context-token growth across strategies
- whether important facts are retained or forgotten
- how server-managed and local replay differ under the same input

## Running The Suite

List the available cases:

```bash
uv run ctagentopenai-memory-lab-suite --list
```

Run one named case:

```bash
uv run ctagentopenai-memory-lab-suite --api-key .openai.key --case forgetting-last-n-2
```

Run the full suite:

```bash
uv run ctagentopenai-memory-lab-suite --api-key .openai.key --all
```

Run a raw memory-lab command directly when you want to experiment outside the
named suite:

```bash
uv run ctagentopenai-memory-lab --api-key .openai.key --input-file inputs/last-n-forgetting-turns.txt --history-mode local-last-n --last-n-turns 2
```

## Suite Cases

### `exercise-server`

Input: `inputs/exercise-plan-turns.txt`

Why it exists:

- baseline multi-turn conversation with service-managed memory
- useful as a control when comparing the same conversation under local replay

What to watch:

- `previous_response_id`
- token growth across turns
- whether the final output incorporates earlier user constraints naturally

### `exercise-last-n-99`

Input: `inputs/exercise-plan-turns.txt`

Why it exists:

- approximates replaying the full local transcript without needing a separate history mode

What to watch:

- stored turns and visible turns should grow together
- transcript character count should increase every turn
- token usage should reflect resending the full conversation each time

### `exercise-last-n-2`

Input: `inputs/exercise-plan-turns.txt`

Why it exists:

- compares the same conversation with an intentionally narrow local replay window

What to watch:

- stored turns keep growing while visible turns cap at 2
- transcript character count should plateau instead of growing without bound
- final answer quality may degrade if older constraints matter

### `forgetting-server`

Input: `inputs/last-n-forgetting-turns.txt`

Why it exists:

- control case for the forgetting scenario
- shows the strongest expected memory behavior before pinned facts or summaries exist

What to watch:

- whether all recorded dietary constraints survive to the final summary turn
- token usage growth without visible forgetting

### `forgetting-last-n-2`

Input: `inputs/last-n-forgetting-turns.txt`

Why it exists:

- intentionally demonstrates forgetting when only the last two turns are replayed

What to watch:

- visible turns remain capped at 2
- transcript chars stay relatively small
- the final answer should tend to miss older constraints

### `forgetting-last-n-3`

Input: `inputs/last-n-forgetting-turns.txt`

Why it exists:

- compares a slightly wider local replay window against `forgetting-last-n-2`

What to watch:

- whether one additional remembered turn changes the final summary materially
- how much transcript and token growth you pay for a wider window

### `overflow-last-n-99`

Input: `inputs/keep-all-overflow-turns.txt`

Why it exists:

- stress test for naive local replay using an intentionally oversized last-N window

What to watch:

- transcript characters and input tokens should grow aggressively
- this is the case most likely to reveal context pressure or rate-limit pressure

### `overflow-server`

Input: `inputs/keep-all-overflow-turns.txt`

Why it exists:

- stress comparison against `overflow-last-n-99`

What to watch:

- whether service-managed history behaves differently from local replay
- whether your local memory instrumentation diverges from what the service appears to retain

## Suggested Comparison Pairs

Use these pairs when you want the fastest signal:

- `exercise-server` vs `exercise-last-n-99`
- `exercise-last-n-99` vs `exercise-last-n-2`
- `forgetting-server` vs `forgetting-last-n-2`
- `forgetting-last-n-2` vs `forgetting-last-n-3`
- `overflow-server` vs `overflow-last-n-99`

## What To Record

As you run the suite, useful notes to capture are:

- where the final answer starts dropping earlier facts
- when input-token growth becomes obviously impractical
- whether the displayed memory snapshot aligns with model behavior
- which scenarios will be useful again after pinned facts and summaries are added

The same suite should stay useful for the next memory milestone. Once pinned
facts and summarization exist, add new cases here rather than scattering notes
through the README.
