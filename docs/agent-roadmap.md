# Agent Roadmap

This document is a lightweight roadmap for exploring agent-building in this repo.
It is meant for a human reader, not as a rigid execution plan. The milestones are
suggestions to help build intuition about the OpenAI SDK, agent loops, tools,
context, and workflow design. They can be reordered, skipped, or expanded.

## Current Baseline

The project already has:

- a one-shot inference path
- a simple conversational loop
- API-managed conversation continuity via `previous_response_id`
- a cheap default model for experimentation

That is enough foundation to move from "chat app" toward "agent system."

## Milestone 1: Tool Calling Loop

Build the first real agent loop where one user turn can trigger multiple
model/tool/model steps before producing a final answer.

Suggested work:

- Add one or two local tools such as `get_time`, `list_files`, or `read_file`
- Define tool schemas manually
- Detect tool calls in the response
- Execute the tool in Python
- Send tool outputs back to the model
- Add a maximum iteration count so loops cannot run forever

What this teaches:

- how function calling actually works
- the boundary between model reasoning and deterministic code
- why loop termination and error handling matter

## Milestone 2: Conversation and Memory Control

Compare API-managed conversation state with app-managed history.

Suggested work:

- Keep your own transcript alongside `previous_response_id`
- Add a way to switch between the two approaches
- Print token usage and rough transcript size during development

What this teaches:

- what state the API is preserving for you
- how context grows over time
- when you need direct control over conversation state

## Milestone 3: Context Management

Start making deliberate decisions about what the model should remember.

Suggested work:

- Keep only the last N turns
- Summarize older turns into short memory notes
- Preserve a few pinned facts separately from normal chat history

What this teaches:

- context budgeting
- what information survives summarization well
- how memory quality affects agent behavior

## Milestone 4: Prompt Design

Split prompt responsibilities so behavior is easier to reason about.

Suggested work:

- Separate system behavior, task prompt, tool guidance, and customer-specific instructions
- Try a few prompt variants and observe what changes
- Keep notes on which prompt changes affect tone versus actual behavior

What this teaches:

- what prompting can solve cleanly
- where prompting becomes too fragile
- when code should enforce behavior instead

## Milestone 5: Observability

Make the agent inspectable instead of relying on intuition.

Suggested work:

- Log model name, latency, token usage, and tool calls
- Save interesting traces locally
- Print enough debug information to understand why a turn behaved a certain way

What this teaches:

- how to debug an agent without guessing
- what actually costs money
- where the slow or brittle parts are

## Milestone 6: Retrieval and Knowledge Access

Teach the agent to look things up instead of pretending it knows everything.

Suggested work:

- Start with a simple local document lookup tool
- Later compare that with chunked retrieval or hosted file search
- Keep the corpus small at first so behavior is easy to inspect

What this teaches:

- when a lookup tool is enough
- when you need retrieval rather than longer prompts
- the limits of naive knowledge injection

## Milestone 7: Multi-Model Routing

Use different models for different jobs.

Suggested work:

- Use a cheap model for simple classification, extraction, or routing
- Use a stronger model only for harder reasoning or final synthesis
- Log which model handled each step

What this teaches:

- cost versus quality tradeoffs
- where model quality actually matters
- how workflows can often matter more than raw model strength

## Milestone 8: Multimodal Inputs

Expand beyond text once the core loop feels solid.

Suggested work:

- Accept an image, screenshot, or PDF for one narrow task
- Keep the task focused, such as describing a screenshot or extracting key details
- Observe how prompt shape and validation change with multimodal input

What this teaches:

- how the input pipeline changes when text is not the only modality
- where model output needs additional verification
- what kinds of tasks become more valuable with multimodality

## Milestone 9: Evaluation and Regression Testing

Move from "it seems better" to "it is better for these cases."

Suggested work:

- Create a small local eval set of prompts
- Record expected tool behavior or answer properties
- Re-run the eval set after prompt, model, or workflow changes

What this teaches:

- how to compare changes honestly
- how regressions appear in agent systems
- which behaviors are stable and which are fragile

## Milestone 10: Workflow Patterns

Explore cases where a workflow engine matters more than a chat loop.

Suggested work:

- Add a planner/executor pattern
- Add a confirmation or approval step before side effects
- Try resumable tasks or checkpoints

What this teaches:

- the difference between assistant behavior and workflow behavior
- when human-in-the-loop design is necessary
- where real customer systems need explicit orchestration

## What Matters Most for Customer Work

If the goal is to design useful workflows or agents for specific customers, the
highest-value topics are usually:

- tool calling
- context management
- retrieval and knowledge access
- observability and evaluation
- workflow control, branching, and approvals

These areas tend to matter more than small prompt tweaks or raw model strength.

## Practical Rules of Thumb

- Use prompting when the need is mostly behavioral or stylistic
- Use tools when the system needs to take actions or fetch deterministic data
- Use retrieval when the problem is missing knowledge
- Use workflow code when order, branching, approvals, or recovery matter
- Use stronger models only after the surrounding workflow is already sound

## Suggested Next Step

The best next implementation step in this repo is Milestone 1: build a small
tool-calling loop with one or two local tools and strong guardrails.
