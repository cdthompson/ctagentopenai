from __future__ import annotations

from collections.abc import Callable, Iterable

from .agent import (
    Agent,
    AgentExecutionHalt,
    ConversationStrategy,
    DEFAULT_MODEL_CONFIG,
    incomplete_response_notice,
    usage_percentage,
)


EmitFn = Callable[[str], None]
PostTurnHook = Callable[[Agent, str, str], None]
InputFn = Callable[[str], str]


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


def print_agent_response(agent: Agent, response_text: str, emit: EmitFn = print) -> None:
    emit(response_text)
    notice = incomplete_response_notice(agent.last_response)
    if notice:
        emit(notice)
    emit(usage_text(agent))


def non_empty_input(prompt: str) -> str:
    user_input = None
    while not user_input:
        user_input = input(prompt)
    return user_input


def run_turns(
    agent: Agent,
    turns: Iterable[str],
    *,
    emit: EmitFn = print,
    post_turn_hook: PostTurnHook | None = None,
) -> None:
    for raw_turn in turns:
        user_input = raw_turn.strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        emit(prompt_text(agent) + user_input)
        emit("")
        try:
            response, _ = agent.inference_with_tools(user_input)
        except AgentExecutionHalt as exc:
            emit(exc.user_message)
            emit("")
            break
        print_agent_response(agent, response, emit=emit)
        if post_turn_hook is not None:
            post_turn_hook(agent, user_input, response)
        emit("")


def run_agent_loop(
    agent: Agent,
    *,
    emit: EmitFn = print,
    input_fn: InputFn = non_empty_input,
    post_turn_hook: PostTurnHook | None = None,
) -> None:
    try:
        user_input = input_fn(prompt_text(agent))
        while user_input.lower() not in ["exit", "quit"]:
            emit("")
            try:
                response, _ = agent.inference_with_tools(user_input)
            except AgentExecutionHalt as exc:
                emit(exc.user_message)
                emit("")
                return
            print_agent_response(agent, response, emit=emit)
            if post_turn_hook is not None:
                post_turn_hook(agent, user_input, response)
            emit("")
            user_input = input_fn(prompt_text(agent))
    except (EOFError, KeyboardInterrupt):
        emit("\nExiting...")
        return
