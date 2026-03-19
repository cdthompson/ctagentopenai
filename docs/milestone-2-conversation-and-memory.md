We are generating a conversational control and memory system for a toy agent we have developed for educational purposes.

The features we want to add to the current codebase are:
1. Add context window usage, as a percentage and token count, to the input prompt.
2. Add a schema for tracking conversation history. 
  a. This should be agnostic of which API we are using and what services that API provides for conversation management
  b. This schema should support three different conversation management strategies to start: server-managed, local-keep everything, local-"last N" only. Local conversation history should be kept in memory only.
3. Update the OpenAI usage to reflect our local conversation tracking configuration. If we are keeping everything, we should be able to overload the server's context window and get an error and not trigger auto-compaction from the API, which would mean our local conversation management is out of sync with the servers.
4. For server-managed history (current strategy of the code), parse the token usage from the API responses to still track context window usage. See API usage at https://developers.openai.com/api/reference/resources/responses/methods/retrieve to determine how.

When implementing, you should:
- Respect the existing codebase and keep modifications minimal so that it is still understandable by humans
- Do not eagerly create abstractions when a simple approach would suffice
- Be generous with logging and emitting turn-based state so the user can understand the internal workings
- If a requirement is ambiguous or has multiple options, ask for preference in implementation.
- Add unit tests for new code
- Update README to indicate new CLI flags added