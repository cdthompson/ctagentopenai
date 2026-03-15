from __future__ import annotations

import subprocess
from datetime import datetime
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
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


class GetTimeTool(Tool):
    spec = ToolSpec(
        name="get_time",
        description="Get the current local date and time for this machine.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    def invoke(self, call: ToolCall) -> ToolResult:
        now = datetime.now().astimezone()
        output = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        return ToolResult(call_id=call.call_id, tool_name=self.spec.name, output=output)


class ListFilesTool(Tool):
    spec = ToolSpec(
        name="list_files",
        description="List files and directories under a path within the current project.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "A file or directory path relative to the current project root.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )

    def invoke(self, call: ToolCall) -> ToolResult:
        arguments = call.arguments or {}
        path_text = arguments.get("path", ".")
        target, error = _resolve_project_path(path_text)
        if error:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.spec.name,
                output=error,
                is_error=True,
            )

        if not target.exists():
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.spec.name,
                output=f"Path does not exist: {path_text}",
                is_error=True,
            )

        if target.is_file():
            output = _display_path(target)
        else:
            entries = sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower(), item.name))
            output = "\n".join(_display_path(entry) for entry in entries)

        return ToolResult(call_id=call.call_id, tool_name=self.spec.name, output=output)


class ReadFileTool(Tool):
    spec = ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file within the current project.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "A file path relative to the current project root.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )

    def invoke(self, call: ToolCall) -> ToolResult:
        arguments = call.arguments or {}
        path_text = arguments.get("path", "")
        target, error = _resolve_project_path(path_text)
        if error:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.spec.name,
                output=error,
                is_error=True,
            )

        if not target.exists():
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.spec.name,
                output=f"Path does not exist: {path_text}",
                is_error=True,
            )

        if not target.is_file():
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.spec.name,
                output=f"Path is not a file: {path_text}",
                is_error=True,
            )

        try:
            output = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.spec.name,
                output=f"File is not valid UTF-8 text: {path_text}",
                is_error=True,
            )

        return ToolResult(call_id=call.call_id, tool_name=self.spec.name, output=output)


def _resolve_project_path(path_text: str) -> tuple[Path | None, str | None]:
    project_root = Path.cwd().resolve()
    candidate = (project_root / path_text).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError:
        return None, f"Path escapes the current project: {path_text}"
    return candidate, None


def _display_path(path: Path) -> str:
    project_root = Path.cwd().resolve()
    relative_path = path.resolve().relative_to(project_root)
    suffix = "/" if path.is_dir() else ""
    return f"{relative_path}{suffix}"


DEFAULT_TOOLS = [
    FavoriteColorTool(),
    CalculatorTool(),
    GetTimeTool(),
    ListFilesTool(),
    ReadFileTool(),
]
