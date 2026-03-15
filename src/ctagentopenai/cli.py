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

from .agent import Agent


def main(argv=None):
    parser = argparse.ArgumentParser(description="CTAgentOpenAI")
    parser.add_argument("--api-key", help="OpenAI API key", required=True)
    parser.add_argument("--input", help="Input for the agent to process", required=False)
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    args = parser.parse_args(argv)

    api_key = open(args.api_key).read().strip()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    print("Welcome to CTAgentOpenAI!")
    print()

    agent = Agent(api_key)

    if args.input:
        print(f"> {args.input}")
        print()
        response_text, _ = agent.inference_with_tools(args.input)
        print(response_text)
        print()
    else:
        agent.run_agent_loop()


if __name__ == "__main__":
    main(sys.argv[1:])
