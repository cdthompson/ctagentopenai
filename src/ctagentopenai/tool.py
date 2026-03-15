from __future__ import annotations

import subprocess
from dataclasses import dataclass
from logging import getLogger
from typing import Any


logger = getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] | None = None
    react_format: str | None = None


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    call_id: str
    arguments: dict[str, Any] | None = None
    react_input: str | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    output: str
    is_error: bool = False


class Tool:
    spec: ToolSpec

    def invoke(self, call: ToolCall) -> ToolResult:
        raise NotImplementedError("Tool subclasses must implement invoke().")


class FavoriteColorTool(Tool):
    spec = ToolSpec(
        name="favorite_color",
        description="Retrieve the user's favorite color.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    def invoke(self, call: ToolCall) -> ToolResult:
        return ToolResult(call_id=call.call_id, tool_name=self.spec.name, output="blue")


class CalculatorTool(Tool):
    spec = ToolSpec(
        name="calculator",
        description="Perform a basic arithmetic calculation from a string expression.",
        input_schema={
            "type": "object",
            "properties": {
                "calculation": {
                    "type": "string",
                    "description": "The calculation to perform, for example '2 + 2'.",
                }
            },
            "required": ["calculation"],
            "additionalProperties": False,
        },
    )

    def invoke(self, call: ToolCall) -> ToolResult:
        arguments = call.arguments or {}
        calculation = arguments.get("calculation", "")

        try:
            result = subprocess.check_output(["bc"], input=calculation.encode())
            output = result.decode().strip()
            return ToolResult(call_id=call.call_id, tool_name=self.spec.name, output=output)
        except subprocess.CalledProcessError as exc:
            logger.warning("Error in calculation: %s", exc)
        except FileNotFoundError:
            logger.warning(
                "Error: 'bc' command not found. Please install 'bc' to use the calculator tool."
            )

        return ToolResult(
            call_id=call.call_id,
            tool_name=self.spec.name,
            output="",
            is_error=True,
        )


DEFAULT_TOOLS = [FavoriteColorTool(), CalculatorTool()]
