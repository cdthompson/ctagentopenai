from __future__ import annotations

import json
from logging import getLogger
from typing import Any

from openai import OpenAI

from .tool import DEFAULT_TOOLS, Tool, ToolCall, ToolResult


MODEL_NAME = "gpt-5-nano"
MAX_OUTPUT_TOKENS = 4096
MAX_TOOL_CALLS = 5
SOUL_PROMPT = (
    "You are a curmudgeonly assistant who is very direct and to the point. "
    "If you don't know something, say you don't know."
)

logger = getLogger(__name__)


def get_tool_by_name(tools: list[Tool], name: str) -> Tool:
    for tool in tools:
        if tool.spec.name == name:
            return tool
    raise KeyError(f"Unknown tool: {name}")


def build_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    openai_tools = []
    for tool in tools:
        spec = tool.spec
        openai_tool = {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "strict": True,
        }
        if spec.input_schema is not None:
            openai_tool["parameters"] = spec.input_schema
        openai_tools.append(openai_tool)
    return openai_tools


def extract_openai_tool_calls(response) -> list[ToolCall]:
    calls = []
    for item in response.output:
        if getattr(item, "type", None) != "function_call":
            continue

        arguments = json.loads(item.arguments) if item.arguments else {}
        calls.append(
            ToolCall(
                tool_name=item.name,
                call_id=item.call_id,
                arguments=arguments,
            )
        )
    return calls


def build_openai_tool_outputs(results: list[ToolResult]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function_call_output",
            "call_id": result.call_id,
            "output": result.output,
        }
        for result in results
    ]


class Agent:
    def __init__(self, api_key: str, tools: list[Tool] | None = None):
        self.client = OpenAI(api_key=api_key)
        self.tools = tools or list(DEFAULT_TOOLS)
        self.openai_tools = build_openai_tools(self.tools)
        self.system_prompt = SOUL_PROMPT

    def non_empty_input(self, prompt):
        """Prompt the user for input until they provide non-empty input."""
        user_input = None
        while not user_input:
            user_input = input(prompt)
        return user_input

    def run_agent_loop(self):
        """Run a conversation in a loop with the agent."""
        try:
            user_input = self.non_empty_input("> ")
            previous_response_id = None
            while user_input.lower() not in ["exit", "quit"]:
                print()
                response, previous_response_id = self.inference_with_tools(
                    user_input, previous_response_id
                )
                print(response)
                print()
                user_input = self.non_empty_input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return

    def inference(self, input, previous_response_id=None):
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
            tools=self.openai_tools,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            previous_response_id=previous_response_id,
        )
        if response.error:
            logger.error("Error in inference: %s", response.error)
            return f"Error in inference: {response.error}", None
        if response.incomplete_details:
            logger.warning("Incomplete response: %s", response.incomplete_details)
            if response.incomplete_details.type == "max_output_tokens":
                logger.warning(
                    "The response was cut off because it exceeded the maximum "
                    "output tokens. Consider increasing the max_output_tokens parameter."
                )

        logger.debug(response.model_dump_json(indent=2))
        return response

    def inference_with_tools(self, input, previous_response_id=None):
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
            response = self.inference(input_list, previous_response_id)
            input_list.extend(response.output)

            tool_calls = extract_openai_tool_calls(response)
            if not tool_calls:
                break

            results = []
            for tool_call in tool_calls:
                logger.debug(
                    "Tool call detected: %s with arguments %s",
                    tool_call.tool_name,
                    tool_call.arguments,
                )
                result = get_tool_by_name(self.tools, tool_call.tool_name).invoke(tool_call)
                logger.debug("Tool call returned: %s", result.output)
                results.append(result)

            input_list.extend(build_openai_tool_outputs(results))

        return response.output_text, response.id
