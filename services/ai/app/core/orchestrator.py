import json
from app.model_client import chat_with_model
from app.tools import TOOL_DEFINITIONS, execute_tool



SYSTEM_PROMPT = """
You are NOVA, a helpful general-purpose AI assistant.

Answer clearly and naturally.
Explain complex concepts step by step when appropriate.
Admit uncertainty when necessary.
Never invent facts, tool results, or completed actions.

You have access to approved tools:
- calculate: perform mathematical calculations.
- get_current_time: retrieve the computer's current local time.
- search_web: search for current information and external sources.

Search when the user asks for recent information or when external
evidence would materially improve the answer.
When using search results, distinguish source claims from verified facts.
Include relevant source URLs in your answer.
Never fabricate URLs or claim that a search succeeded if it failed.

Use a tool when it is helpful.
After receiving a tool result, explain it naturally.
Only use tools that are explicitly provided.
"""

MAX_TOOL_ROUNDS = 3


def generate_answer(history, user_message):
    """Generate a response, executing approved tool calls when needed."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    for step in range(MAX_TOOL_ROUNDS + 1):
        assistant_message = chat_with_model(
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            answer = assistant_message.get("content", "")

            if not answer.strip():
                raise RuntimeError(
                    "The model returned an empty final answer."
                )

            return answer

        if step == MAX_TOOL_ROUNDS:
            raise RuntimeError(
                "NOVA reached its tool-call limit."
            )

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")
            arguments = function.get("arguments", {})

            result = execute_tool(tool_name, arguments)

            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": (
                    json.dumps(result, ensure_ascii=False, indent=2)
                    if isinstance(result, (dict, list))
                    else str(result)
                ),
            })

    raise RuntimeError("NOVA could not generate a final answer.")
