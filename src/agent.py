from openai import OpenAI
import json
from logging import getLogger

MODEL_NAME = "gpt-5-nano"
SYSTEM_PROMPT = "You are a curmudgeonly assistant who is very direct and to the point. If you don't know something, say you don't know."
MAX_OUTPUT_TOKENS = 2048
logger = getLogger(__name__)


class Agent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def non_empty_input(self, prompt):
        """Prompt the user for input until they provide non-empty input."""
        user_input = None
        while not user_input:
            user_input = input(prompt)
        return user_input


    def run_agent_loop(self, api_key):
        """Run a conversation in a loop with the agent."""
        try:
            user_input = self.non_empty_input("> ")
            previous_response_id = None
            while user_input.lower() not in ["exit", "quit"]:
                print()
                response, previous_response_id = self.inference(user_input, api_key, previous_response_id)
                print(response)
                print()
                user_input = self.non_empty_input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return


    def inference(self, input, api_key, previous_response_id=None):
        """Run inference on the input using the OpenAI API."""
        response = self.client.responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_PROMPT,
            input=input,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            previous_response_id=previous_response_id,
        )
        logger.debug(response.model_dump_json(indent=2))
        return response.output_text, response.id
