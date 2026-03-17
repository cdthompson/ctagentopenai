from __future__ import annotations

import argparse
import logging
import sys

from .agent import Agent, ConversationStrategy
from .runner import run_turns, startup_summary


def memory_snapshot_text(agent: Agent) -> str:
    state = agent.conversation_state
    return (
        "[memory "
        f"strategy={state.strategy.value} | "
        f"stored_turns={len(state.turns)} | "
        f"visible_turns={len(state.visible_turns())} | "
        f"transcript_chars={state.transcript_character_count()} | "
        f"last_input_tokens={state.latest_usage.input_tokens} | "
        f"last_output_tokens={state.latest_usage.output_tokens} | "
        f"previous_response_id={state.previous_response_id or '-'}"
        "]"
    )


def print_memory_snapshot(agent: Agent, user_input: str, response_text: str) -> None:
    del user_input
    del response_text
    print(memory_snapshot_text(agent))


def main(argv=None):
    parser = argparse.ArgumentParser(description="CTAgentOpenAI memory lab")
    parser.add_argument("--api-key", help="OpenAI API key", required=True)
    parser.add_argument(
        "--input-file",
        help="Read one user turn per line from a file.",
        required=True,
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
    args = parser.parse_args(argv)

    api_key = open(args.api_key).read().strip()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.info:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    print("Welcome to CTAgentOpenAI memory lab!")
    print()

    agent = Agent(
        api_key,
        conversation_strategy=ConversationStrategy(args.history_mode),
        last_n_turns=args.last_n_turns,
    )
    print(startup_summary(agent))
    print(memory_snapshot_text(agent))
    print()

    with open(args.input_file, encoding="utf-8") as input_file:
        run_turns(agent, input_file, post_turn_hook=print_memory_snapshot)


if __name__ == "__main__":
    main(sys.argv[1:])
