from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from . import memory_lab
from .agent import ConversationStrategy


@dataclass(frozen=True)
class MemoryLabCase:
    name: str
    description: str
    input_file: str
    history_mode: str
    last_n_turns: int = 3

    def argv(self, api_key_path: str) -> list[str]:
        return [
            "--api-key",
            api_key_path,
            "--input-file",
            self.input_file,
            "--history-mode",
            self.history_mode,
            "--last-n-turns",
            str(self.last_n_turns),
        ]


SUITE_CASES = [
    MemoryLabCase(
        name="exercise-server",
        description="Baseline conversational run with server-managed history.",
        input_file="inputs/exercise-plan-turns.txt",
        history_mode=ConversationStrategy.SERVER_MANAGED.value,
    ),
    MemoryLabCase(
        name="exercise-last-n-99",
        description="Baseline conversational run with a very wide local replay window.",
        input_file="inputs/exercise-plan-turns.txt",
        history_mode=ConversationStrategy.LOCAL_LAST_N.value,
        last_n_turns=99,
    ),
    MemoryLabCase(
        name="exercise-last-n-2",
        description="Baseline conversational run with only the last two turns replayed locally.",
        input_file="inputs/exercise-plan-turns.txt",
        history_mode=ConversationStrategy.LOCAL_LAST_N.value,
        last_n_turns=2,
    ),
    MemoryLabCase(
        name="forgetting-server",
        description="Control case for the forgetting scenario using service-managed memory.",
        input_file="inputs/last-n-forgetting-turns.txt",
        history_mode=ConversationStrategy.SERVER_MANAGED.value,
    ),
    MemoryLabCase(
        name="forgetting-last-n-2",
        description="Forgetting scenario with only the last two turns replayed locally.",
        input_file="inputs/last-n-forgetting-turns.txt",
        history_mode=ConversationStrategy.LOCAL_LAST_N.value,
        last_n_turns=2,
    ),
    MemoryLabCase(
        name="forgetting-last-n-3",
        description="Forgetting scenario with a slightly wider local replay window.",
        input_file="inputs/last-n-forgetting-turns.txt",
        history_mode=ConversationStrategy.LOCAL_LAST_N.value,
        last_n_turns=3,
    ),
    MemoryLabCase(
        name="overflow-last-n-99",
        description="Stress case showing how a very wide local replay window grows context quickly.",
        input_file="inputs/keep-all-overflow-turns.txt",
        history_mode=ConversationStrategy.LOCAL_LAST_N.value,
        last_n_turns=99,
    ),
    MemoryLabCase(
        name="overflow-server",
        description="Stress case with server-managed context for comparison against wide local replay.",
        input_file="inputs/keep-all-overflow-turns.txt",
        history_mode=ConversationStrategy.SERVER_MANAGED.value,
    ),
]


def get_case(name: str) -> MemoryLabCase:
    for case in SUITE_CASES:
        if case.name == name:
            return case
    raise KeyError(f"Unknown case: {name}")


def print_case_list() -> None:
    print("Memory lab suite cases:")
    print()
    for case in SUITE_CASES:
        print(
            f"- {case.name}: {case.description} "
            f"[input={case.input_file}, history_mode={case.history_mode}, last_n_turns={case.last_n_turns}]"
        )


def run_case(case: MemoryLabCase, api_key_path: str) -> None:
    print(f"=== Case: {case.name} ===")
    print(case.description)
    print(
        f"[suite input={case.input_file} history_mode={case.history_mode} last_n_turns={case.last_n_turns}]"
    )
    print()
    memory_lab.main(case.argv(api_key_path))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="CTAgentOpenAI memory lab suite")
    parser.add_argument("--api-key", help="OpenAI API key file path", required=False)
    parser.add_argument("--list", help="List the available suite cases.", action="store_true")
    parser.add_argument("--case", help="Run a single named suite case.", required=False)
    parser.add_argument("--all", help="Run all suite cases.", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        print_case_list()
        return

    if not args.case and not args.all:
        raise SystemExit("choose --list, --case NAME, or --all")

    if not args.api_key:
        raise SystemExit("--api-key is required when running suite cases")

    api_key_path = str(Path(args.api_key))
    if args.case:
        run_case(get_case(args.case), api_key_path)
        return

    for case in SUITE_CASES:
        run_case(case, api_key_path)


if __name__ == "__main__":
    main(sys.argv[1:])
