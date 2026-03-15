import json
import subprocess
from logging import getLogger

from openai import OpenAI


MODEL_NAME = "gpt-5-nano"
MAX_OUTPUT_TOKENS = 4096
MAX_TOOL_CALLS = 5
SOUL_PROMPT = (
    "You are a curmudgeonly assistant who is very direct and to the point. "
    "If you don't know something, say you don't know."
)

logger = getLogger(__name__)


class BaseTool:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    def __call__(self, input):
        raise NotImplementedError("Tool subclasses must implement the __call__ method.")

    def to_openai_tool(self):
        """Convert the tool to a format compatible with the OpenAI API."""
        tool_dict = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }
        if self.parameters:
            tool_dict["parameters"] = self.parameters
        return tool_dict


class FavoriteColorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="FavoriteColorTool",
            description=(
                "This tool allows you to retrieve the user's favorite color. "
                "It takes no input and returns a string with their favorite color."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        )

    def __call__(self, input=None):
        return "blue"


class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="CalculatorTool",
            description=(
                "This tool allows you to perform basic arithmetic calculations. "
                "It takes a string input with the calculation you want to perform "
                "(e.g. '2 + 2') and returns a numeric value with the result."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "calculation": {
                        "type": "string",
                        "description": "The calculation to perform, as a string (e.g. '2 + 2')",
                    }
                },
                "required": ["calculation"],
                "additionalProperties": False,
            },
        )

    def __call__(self, calculation):
        """Use bc to perform basic arithmetic calculations."""
        try:
            result = subprocess.check_output(["bc"], input=calculation.encode())
            return result.decode().strip()
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error in calculation: {e}")
        except FileNotFoundError:
            logger.warning(
                "Error: 'bc' command not found. Please install 'bc' to use the Calculator Tool."
            )

        return ""


class Agent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.tools = [x.to_openai_tool() for x in [FavoriteColorTool(), CalculatorTool()]]
        self.system_prompt = SOUL_PROMPT

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
                response, previous_response_id = self.inference_with_tools(
                    user_input, api_key, previous_response_id
                )
                print(response)
                print()
                user_input = self.non_empty_input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return

    def inference(self, input, api_key, previous_response_id=None):
        """Run inference on the input using the OpenAI API."""
        if isinstance(input, str):
            input = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": input,
                },
            ]

        response = self.client.responses.create(
            model=MODEL_NAME,
            input=input,
            tools=self.tools,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            previous_response_id=previous_response_id,
        )
        if response.error:
            logger.error(f"Error in inference: {response.error}")
            return f"Error in inference: {response.error}", None
        if response.incomplete_details:
            logger.warning(f"Incomplete response: {response.incomplete_details}")
            if response.incomplete_details.type == "max_output_tokens":
                logger.warning(
                    "The response was cut off because it exceeded the maximum "
                    "output tokens. Consider increasing the max_output_tokens parameter."
                )

        logger.debug(response.model_dump_json(indent=2))
        return response

    def inference_with_tools(self, input, api_key, previous_response_id=None):
        """Run inference on the input using the OpenAI API, and handle tool calls."""
        input_list = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": input,
            },
        ]
        for _ in range(MAX_TOOL_CALLS):
            response = self.inference(input_list, api_key, previous_response_id)
            input_list.extend(response.output)

            tool_was_called = False
            for item in response.output:
                if getattr(item, "type", None) == "function_call":
                    logger.debug(f"Tool call detected: {item.name} with arguments {item.arguments}")
                    tool_class = globals()[item.name]
                    tool_args = json.loads(item.arguments)
                    result = tool_class()(**tool_args)
                    logger.debug(f"Tool call returned: {result}")
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": result,
                        }
                    )
                    tool_was_called = True

            if not tool_was_called:
                break

        return response.output_text, response.id
