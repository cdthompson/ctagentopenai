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

## Project Notes

- The package source lives under `src/ctagentopenai/`
- Tests live under `tests/`
- Roadmap and study notes live under `docs/`
