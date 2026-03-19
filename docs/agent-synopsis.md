# Agent Synopsis

This is a compact study guide for the main topics in the agent roadmap. It is
written for a human reader and is meant to help build intuition about common
approaches, tradeoffs, and where strong opinions diverge.

## 1. Tool Calling

Tool calling is the core pattern behind most practical agents:

1. the model chooses an action
2. the application executes it
3. the result is sent back to the model
4. the loop continues until a final answer is ready

Common strategies:

- structured function/tool calling with schemas
- deterministic controller loops in application code
- small, explicit tools rather than broad, fuzzy tools

Competing approaches:

- structured tool calls versus free-form ReAct-style text actions
- many small tools versus fewer, richer tools

Practical take:

Structured tool calling is the standard production approach because it is easier
to validate, log, and control.

## 2. Conversation and Memory Control

This is about how state persists across turns.

Common strategies:

- API-managed continuity for fast prototypes
- application-managed transcript for portability and control
- hybrid models with short-term model context and explicit durable memory

Competing approaches:

- keep the entire transcript versus aggressively manage memory
- rely on provider-managed state versus own the state yourself

Practical take:

API-managed conversation is convenient early, but app-managed state becomes more
useful as workflows get more complex or need to be portable across providers.

## 3. Context Management

Context management is the discipline of deciding what the model should see on
each turn.

Common strategies:

- sliding windows of recent turns
- summaries for older turns
- pinned facts or memory notes separate from chat history
- token budgeting before each request

Competing approaches:

- summarize early versus summarize late
- include as much context as possible versus aggressively filter context

Practical take:

More context is not always better. Irrelevant context often harms the model as
much as missing context.

## 4. Prompt Design

Prompt design works best when different instructions have different jobs.

Common strategies:

- separate behavior, task framing, tool guidance, output constraints, and environment context
- keep stable instructions separate from per-request instructions
- use code for guarantees and prompts for behavior shaping

Competing approaches:

- minimal prompts versus heavily engineered prompts
- prompt-first iteration versus workflow/code-first iteration

Practical take:

Prompting is best for shaping style and behavior. It is not a substitute for
workflow control or deterministic enforcement.

## 5. Observability

Observability is how you understand what the agent actually did.

Common strategies:

- log model, latency, token usage, tool calls, errors, and outputs
- save traces for failed or interesting runs
- keep raw response payloads available in debug mode

Competing approaches:

- log everything versus log selectively
- human-readable logs versus structured traces

Practical take:

Without observability, agent debugging becomes guesswork. This is one of the
highest-value investments once the loop is more than trivial.

## 6. Retrieval and Knowledge Access

This is how the system looks up information it does not already know.

Common strategies:

- start with simple file or document lookup
- move to chunking and retrieval when the knowledge base grows
- prefer targeted lookup over stuffing large documents into prompts

Competing approaches:

- naive keyword lookup versus embeddings-based retrieval
- hosted retrieval tools versus custom RAG infrastructure

Practical take:

Many systems do not need sophisticated RAG at first. A narrow lookup tool can be
enough to learn the pattern and validate the workflow.

## 7. Multi-Model Routing

This is the use of different models for different kinds of work.

Common strategies:

- cheap model for routing, classification, and extraction
- stronger model for synthesis or harder reasoning
- explicit routing rules before dynamic routing

Competing approaches:

- one strong model everywhere versus routed models
- controller-driven escalation versus model-decided escalation

Practical take:

Routing often saves money and can improve reliability, but only after the rest
of the workflow is already clear.

## 8. Multimodal Inputs

This expands the agent beyond plain text.

Common strategies:

- start with one narrow use case, such as screenshot understanding
- normalize or preprocess inputs when helpful
- validate outputs more carefully because multimodal extraction can be noisy

Competing approaches:

- end-to-end multimodal reasoning versus preprocess-then-reason pipelines

Practical take:

Multimodality makes systems more useful, but it also increases ambiguity and the
need for validation.

## 9. Evaluation and Regression Testing

Evaluation tells you whether changes actually improved the system.

Common strategies:

- maintain a small local eval set of representative tasks
- include expected tool usage or output properties
- re-run evals after prompt, model, or workflow changes

Competing approaches:

- manual spot-checking versus formal eval harnesses
- exact expected outputs versus rubric-based checks

Practical take:

Manual testing works at the beginning, but stable progress becomes much easier
once you have a few repeatable cases.

## 10. Workflow Patterns

This is where agents stop being chat loops and start becoming systems.

Common strategies:

- planner/executor separation
- human approval before side effects
- checkpoints and resumability
- deterministic orchestration around model decisions

Competing approaches:

- unconstrained autonomous agents versus constrained workflows
- one generalist agent versus multiple specialized agents

Practical take:

Most production systems lean more heavily on workflow control than on pure model
autonomy.

## Broad Consensus

The most widely accepted practices across providers and teams are:

- use structured tool calling instead of parsing free-form text
- keep deterministic application control around model loops
- manage context explicitly
- use retrieval for external knowledge instead of giant prompts
- add observability and basic evals early
- start with narrow, constrained workflows before chasing full autonomy

## Commonly Debated Areas

These topics often have strong but conflicting opinions:

- how much conversation history to keep versus summarize
- how much effort to put into prompt engineering before changing workflow code
- whether one model or multiple routed models is better
- when multi-agent architectures are actually worth the complexity
- whether to use hosted retrieval/tooling versus custom infrastructure

## Suggested Learning Order

If the goal is intuition plus practical engineering judgment, a good order is:

1. tool calling
2. memory and context management
3. observability
4. retrieval
5. workflow patterns
6. multi-model routing
7. prompt layering
8. evaluation
9. multimodality
10. fine-tuning or training only after the simpler levers are clearly insufficient

## Final Rule of Thumb

Teams often reach for fine-tuning or more powerful models too early. In many
cases, the real problem is better solved with:

- clearer tools
- better context selection
- retrieval
- stronger workflow control
- better logging and evaluation
