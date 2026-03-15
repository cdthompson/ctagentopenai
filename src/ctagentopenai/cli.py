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

from .agent import (
    Agent,
    AgentExecutionHalt,
    ConversationStrategy,
    DEFAULT_MODEL_CONFIG,
    incomplete_response_notice,
    usage_percentage,
)


def prompt_text(agent: Agent) -> str:
    return f"[{usage_percentage(agent.conversation_state.latest_usage):.2f}%] > "


def usage_text(agent: Agent) -> str:
    usage = agent.conversation_state.latest_usage
    return (
        "[context "
        f"{usage_percentage(usage):.2f}% | "
        f"{usage.input_tokens}/{DEFAULT_MODEL_CONFIG['context_window']} input tokens | "
        f"{usage.output_tokens} output tokens | "
        f"{usage.total_tokens} total tokens]"
    )


def startup_summary(agent: Agent) -> str:
    last_n_suffix = ""
    if agent.conversation_state.strategy == ConversationStrategy.LOCAL_LAST_N:
        last_n_suffix = f", last_n_turns={agent.conversation_state.last_n_turns}"
    return (
        "[startup "
        f"model=gpt-5-nano, "
        f"history_mode={agent.conversation_state.strategy.value}"
        f"{last_n_suffix}, "
        f"context_window={DEFAULT_MODEL_CONFIG['context_window']}, "
        f"default_max_output_tokens={DEFAULT_MODEL_CONFIG['default_max_output_tokens']}"
        "]"
    )


def print_agent_response(agent: Agent, response_text: str) -> None:
    print(response_text)
    notice = incomplete_response_notice(agent.last_response)
    if notice:
        print(notice)
    print(usage_text(agent))


def non_empty_input(prompt: str) -> str:
    user_input = None
    while not user_input:
        user_input = input(prompt)
    return user_input


def run_turns(agent: Agent, turns) -> None:
    for raw_turn in turns:
        user_input = raw_turn.strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        print(prompt_text(agent) + user_input)
        print()
        try:
            response, _ = agent.inference_with_tools(user_input)
        except AgentExecutionHalt as exc:
            print(exc.user_message)
            print()
            break
        print_agent_response(agent, response)
        print()


def run_agent_loop(agent: Agent) -> None:
    try:
        user_input = non_empty_input(prompt_text(agent))
        while user_input.lower() not in ["exit", "quit"]:
            print()
            try:
                response, _ = agent.inference_with_tools(user_input)
            except AgentExecutionHalt as exc:
                print(exc.user_message)
                print()
                return
            print_agent_response(agent, response)
            print()
            user_input = non_empty_input(prompt_text(agent))
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        return


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

    agent = Agent(
        api_key,
        conversation_strategy=ConversationStrategy(args.history_mode),
        last_n_turns=args.last_n_turns,
    )
    print(startup_summary(agent))
    print()

    if args.input and args.input_file:
        raise SystemExit("--input and --input-file cannot be used together")

    if args.input:
        print(f"> {args.input}")
        print()
        try:
            response_text, _ = agent.inference_with_tools(args.input)
        except AgentExecutionHalt as exc:
            print(exc.user_message)
            print()
            return
        print_agent_response(agent, response_text)
        print()
    elif args.input_file:
        with open(args.input_file, encoding="utf-8") as input_file:
            run_turns(agent, input_file)
    else:
        run_agent_loop(agent)


if __name__ == "__main__":
    main(sys.argv[1:])
