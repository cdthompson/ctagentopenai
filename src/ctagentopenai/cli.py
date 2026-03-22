"""
This project is for experimenting with OpenAI SDK in attempt to create an
agent which can chat and has use of some minimal tooling.

Items of interest I'm learning with this project (along with commands run):
- OpenAI SDK https://developers.openai.com/api/docs/quickstart/?language=python
    1. Sign into platform.openai.com
    2. Generate a new API key and save to .openai.key
- uv package manager https://docs.astral.sh/uv/getting-started/
    1. Install uv `curl -LsSf https://astral.sh/uv/install.sh | sh`
    2. Check installed python version `uv python list`
"""

import argparse
import logging
import sys

from .agent import Agent, ConversationStrategy
from .runner import print_agent_response, run_agent_loop, run_turns, startup_summary
from .tool import DEFAULT_TOOLS, QueryLabelTool


def main(argv=None):
    parser = argparse.ArgumentParser(description="CTAgentOpenAI")
    parser.add_argument("--api-key", help="OpenAI API key", required=True)
    parser.add_argument("--input", help="Input for the agent to process", required=False)
    parser.add_argument(
        "--input-file",
        help="Read one user turn per line from a file.",
        required=False,
    )
    parser.add_argument("--info", help="Enable info level logging", action="store_true")
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    parser.add_argument(
        "--history-mode",
        choices=[strategy.value for strategy in ConversationStrategy],
        default=ConversationStrategy.SERVER_MANAGED.value,
        help="Conversation history strategy to use.",
    )
    parser.add_argument(
        "--last-n-turns",
        type=int,
        default=3,
        help="Number of turns to keep when using local-last-n conversation history.",
    )
    parser.add_argument(
        "--summary-trigger-turns",
        type=int,
        default=None,
        help="When set, summarize older turns after the total stored turn count exceeds this value.",
    )
    parser.add_argument(
        "--summary-keep-recent-turns",
        type=int,
        default=2,
        help="Number of recent turns to keep verbatim after summary compaction.",
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help="Model to use for summary compaction. Defaults to the main conversation model.",
    )
    parser.add_argument(
        "--summary-reasoning-effort",
        default=None,
        help="Reasoning effort to use for summary compaction.",
    )
    parser.add_argument(
        "--label-db",
        default=None,
        help="Path to a local FDA label SQLite corpus. Enables the query_label tool when set.",
    )
    parser.add_argument(
        "--label-retrieval-method",
        choices=["bm25", "grep"],
        default="bm25",
        help="Retrieval method to use for the FDA label query tool.",
    )
    args = parser.parse_args(argv)

    api_key = open(args.api_key).read().strip()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.info:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    print("Welcome to CTAgentOpenAI!")
    print()

    tools = list(DEFAULT_TOOLS)
    if args.label_db:
        tools.append(QueryLabelTool(args.label_db, retrieval_method=args.label_retrieval_method))

    agent_kwargs = {
        "conversation_strategy": ConversationStrategy(args.history_mode),
        "last_n_turns": args.last_n_turns,
        "summary_trigger_turns": args.summary_trigger_turns,
        "summary_keep_recent_turns": args.summary_keep_recent_turns,
        "summary_model": args.summary_model,
        "summary_reasoning_effort": args.summary_reasoning_effort,
    }
    if args.label_db:
        agent_kwargs["tools"] = tools

    agent = Agent(api_key, **agent_kwargs)
    print(startup_summary(agent))
    print()

    if args.input and args.input_file:
        raise SystemExit("--input and --input-file cannot be used together")

    if args.input:
        print(f"> {args.input}")
        print()
        response_text, _ = agent.inference_with_tools(args.input)
        print_agent_response(agent, response_text)
        print()
    elif args.input_file:
        with open(args.input_file, encoding="utf-8") as input_file:
            run_turns(agent, input_file)
    else:
        run_agent_loop(agent)


if __name__ == "__main__":
    main(sys.argv[1:])
