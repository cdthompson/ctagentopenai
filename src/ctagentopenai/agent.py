from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import Any

from openai import OpenAI, RateLimitError

from .tool import DEFAULT_TOOLS, Tool, ToolCall, ToolResult


MODEL_CONFIG = {
    "gpt-5-nano": {
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "default_max_output_tokens": 8192,
    }
}
DEFAULT_MODEL_NAME = "gpt-5-nano"
DEFAULT_MODEL_CONFIG = MODEL_CONFIG[DEFAULT_MODEL_NAME]
MAX_TOOL_CALLS = 5
SOUL_PROMPT = (
    "You are a curmudgeonly assistant who is very direct and to the point. "
    "If you don't know something, say you don't know."
)

logger = getLogger(__name__)


class ConversationStrategy(str, Enum):
    SERVER_MANAGED = "server-managed"
    LOCAL_ALL = "local-all"
    LOCAL_LAST_N = "local-last-n"


class AgentExecutionHalt(RuntimeError):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class UsageSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class TurnRecord:
    user_message: str
    assistant_message: str
    usage: UsageSnapshot
    response_id: str | None = None


@dataclass
class ConversationState:
    strategy: ConversationStrategy = ConversationStrategy.SERVER_MANAGED
    last_n_turns: int = 3
    turns: list[TurnRecord] = field(default_factory=list)
    previous_response_id: str | None = None
    latest_usage: UsageSnapshot = field(default_factory=UsageSnapshot)

    def build_input(self, user_input: str, system_prompt: str) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(system_prompt, self.strategy, self.latest_usage),
            }
        ]
        for turn in self.visible_turns():
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})
        messages.append({"role": "user", "content": user_input})
        return messages

    def visible_turns(self) -> list[TurnRecord]:
        if self.strategy == ConversationStrategy.LOCAL_LAST_N:
            return self.turns[-self.last_n_turns :]
        if self.strategy == ConversationStrategy.LOCAL_ALL:
            return self.turns
        return []

    def previous_response_id_for_request(self) -> str | None:
        if self.strategy == ConversationStrategy.SERVER_MANAGED:
            return self.previous_response_id
        return None

    def record_turn(self, user_input: str, assistant_message: str, response) -> None:
        self.latest_usage = usage_snapshot_from_response(response)
        self.previous_response_id = getattr(response, "id", None)
        self.turns.append(
            TurnRecord(
                user_message=user_input,
                assistant_message=assistant_message,
                usage=self.latest_usage,
                response_id=self.previous_response_id,
            )
        )

    def transcript_character_count(self) -> int:
        return sum(
            len(turn.user_message) + len(turn.assistant_message)
            for turn in self.visible_turns()
        )


def usage_snapshot_from_response(response) -> UsageSnapshot:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageSnapshot()
    return UsageSnapshot(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def incomplete_response_notice(response) -> str | None:
    incomplete = getattr(response, "incomplete_details", None)
    if incomplete is None:
        return None

    reason = getattr(incomplete, "reason", None)
    if reason == "max_output_tokens":
        return "[response truncated: reached max_output_tokens]"
    if reason == "content_filter":
        return "[response truncated: content filtered]"
    return f"[response incomplete: {reason or 'unknown reason'}]"


def build_system_prompt(
    system_prompt: str,
    strategy: ConversationStrategy,
    usage: UsageSnapshot,
) -> str:
    usage_pct = usage_percentage(usage)
    return (
        f"{system_prompt}\n\n"
        f"Conversation mode: {strategy.value}.\n"
        f"Estimated current context window usage: "
        f"{usage.input_tokens}/{DEFAULT_MODEL_CONFIG['context_window']} tokens ({usage_pct:.2f}%)."
    )


def usage_percentage(usage: UsageSnapshot) -> float:
    context_window = DEFAULT_MODEL_CONFIG["context_window"]
    return (usage.input_tokens / context_window) * 100 if context_window else 0.0


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
    def __init__(
        self,
        api_key: str,
        tools: list[Tool] | None = None,
        conversation_strategy: ConversationStrategy = ConversationStrategy.SERVER_MANAGED,
        last_n_turns: int = 3,
    ):
        if last_n_turns < 1:
            raise ValueError("last_n_turns must be at least 1")
        self.client = OpenAI(api_key=api_key)
        self.tools = tools or list(DEFAULT_TOOLS)
        self.openai_tools = build_openai_tools(self.tools)
        self.system_prompt = SOUL_PROMPT
        self.last_response = None
        self.conversation_state = ConversationState(
            strategy=conversation_strategy,
            last_n_turns=last_n_turns,
        )

    def log_context_usage(self) -> None:
        logger.info(
            "Context usage: strategy=%s input_tokens=%s/%s (%.2f%%) output_tokens=%s total_tokens=%s",
            self.conversation_state.strategy.value,
            self.conversation_state.latest_usage.input_tokens,
            DEFAULT_MODEL_CONFIG["context_window"],
            usage_percentage(self.conversation_state.latest_usage),
            self.conversation_state.latest_usage.output_tokens,
            self.conversation_state.latest_usage.total_tokens,
        )

    def inference(self, input, previous_response_id=None):
        """Run inference on the input using the OpenAI API."""
        if isinstance(input, str):
            input = [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        self.system_prompt,
                        self.conversation_state.strategy,
                        self.conversation_state.latest_usage,
                    ),
                },
                {
                    "role": "user",
                    "content": input,
                },
            ]

        try:
            response = self.client.responses.create(
                model=DEFAULT_MODEL_NAME,
                input=input,
                tools=self.openai_tools,
                max_output_tokens=DEFAULT_MODEL_CONFIG["default_max_output_tokens"],
                max_tool_calls=MAX_TOOL_CALLS,
                previous_response_id=previous_response_id,
            )
        except RateLimitError as exc:
            logger.error("Rate limit error in inference: %s", exc)
            raise AgentExecutionHalt(
                "[request rejected: exceeded API rate limit or token budget; stopping further turns]"
            ) from exc
        if response.error:
            logger.error("Error in inference: %s", response.error)
            return f"Error in inference: {response.error}", None
        if response.incomplete_details:
            logger.warning("Incomplete response: %s", response.incomplete_details)
            if response.incomplete_details.reason == "max_output_tokens":
                logger.warning(
                    "The response was cut off because it exceeded the maximum "
                    "output tokens. Consider increasing the max_output_tokens parameter."
                )

        logger.debug(response.model_dump_json(indent=2))
        return response

    def inference_with_tools(self, input, previous_response_id=None):
        """Run inference on the input using the OpenAI API, and handle tool calls."""
        input_list = self.conversation_state.build_input(input, self.system_prompt)
        previous_response_id = self.conversation_state.previous_response_id_for_request()
        logger.debug(
            "Turn state: strategy=%s local_turns=%s visible_turns=%s "
            "estimated_input_tokens=%s transcript_chars=%s",
            self.conversation_state.strategy.value,
            len(self.conversation_state.turns),
            len(self.conversation_state.visible_turns()),
            self.conversation_state.latest_usage.input_tokens,
            self.conversation_state.transcript_character_count(),
        )
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

        self.last_response = response
        self.conversation_state.record_turn(input, response.output_text, response)
        self.log_context_usage()
        logger.debug("Response id: %s", self.conversation_state.previous_response_id)
        return response.output_text, response.id
