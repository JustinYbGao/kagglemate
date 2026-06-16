#!/usr/bin/env python3
"""Verify DeepSeek V4 Pro function calling works correctly.

This must pass before building any LangGraph nodes.
Run: python tests/test_deepseek_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI
from kagglemate.config import config


def test_single_tool_call():
    """Test: model calls the right tool with the right arguments."""
    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_competition_files",
                "description": "List all data files available in a Kaggle competition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "competition_slug": {
                            "type": "string",
                            "description": "e.g. 'titanic', 'playground-series-s5e6'",
                        }
                    },
                    "required": ["competition_slug"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Kaggle competition assistant. "
                "When a user asks about a competition, use the available tools."
            ),
        },
        {
            "role": "user",
            "content": "What data files are available for the titanic competition?",
        },
    ]

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    # Assert tool was called
    assert msg.tool_calls is not None, (
        f"Expected tool_calls, got None. Content: {msg.content}"
    )
    assert len(msg.tool_calls) == 1, f"Expected 1 tool call, got {len(msg.tool_calls)}"

    tc = msg.tool_calls[0]
    assert tc.function.name == "get_competition_files", (
        f"Expected 'get_competition_files', got '{tc.function.name}'"
    )

    args = json.loads(tc.function.arguments)
    assert "competition_slug" in args, f"Missing 'competition_slug' in args: {args}"
    assert "titanic" in args["competition_slug"].lower(), (
        f"Expected 'titanic', got '{args['competition_slug']}'"
    )

    print("  ✓ test_single_tool_call — model correctly called get_competition_files('titanic')")


def test_multi_tool_selection():
    """Test: given multiple tools, model picks the correct one."""
    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_competition_files",
                "description": "List data files in a Kaggle competition",
                "parameters": {
                    "type": "object",
                    "properties": {"competition_slug": {"type": "string"}},
                    "required": ["competition_slug"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "download_competition_data",
                "description": "Download and extract competition data files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "competition_slug": {"type": "string"},
                        "target_dir": {"type": "string", "description": "Where to save"},
                    },
                    "required": ["competition_slug", "target_dir"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_notebooks",
                "description": "List popular public notebooks for a competition",
                "parameters": {
                    "type": "object",
                    "properties": {"competition_slug": {"type": "string"}},
                    "required": ["competition_slug"],
                },
            },
        },
    ]

    tests = [
        {
            "label": "download request → download_competition_data",
            "user": "Download the titanic data to ./data/titanic/",
            "expected_tool": "download_competition_data",
        },
        {
            "label": "list files → get_competition_files",
            "user": "Show me what files are in the house-prices competition",
            "expected_tool": "get_competition_files",
        },
        {
            "label": "notebooks → list_notebooks",
            "user": "What are the most popular notebooks for the spaceship-titanic competition?",
            "expected_tool": "list_notebooks",
        },
    ]

    system_msg = {
        "role": "system",
        "content": "You are a Kaggle competition assistant. Use the right tool for each request.",
    }

    for test in tests:
        messages = [system_msg, {"role": "user", "content": test["user"]}]
        response = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        assert msg.tool_calls is not None, (
            f"[{test['label']}]: No tool call. Content: {msg.content}"
        )
        actual = msg.tool_calls[0].function.name
        assert actual == test["expected_tool"], (
            f"[{test['label']}]: Expected '{test['expected_tool']}', got '{actual}'"
        )
        print(f"  ✓ {test['label']}")


def test_tool_call_with_structured_output():
    """Test: multi-turn — model calls a tool, we simulate the result, model responds."""
    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_competition_files",
                "description": "List data files in a Kaggle competition",
                "parameters": {
                    "type": "object",
                    "properties": {"competition_slug": {"type": "string"}},
                    "required": ["competition_slug"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a Kaggle assistant. Use tools, then summarize for the user.",
        },
        {"role": "user", "content": "What files does the titanic competition have?"},
    ]

    # Turn 1: model calls tool
    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = response.choices[0].message
    assert msg.tool_calls is not None, f"Turn 1: no tool call"

    tc_id = msg.tool_calls[0].id
    tc_name = msg.tool_calls[0].function.name

    # Append assistant message and fake tool result
    messages.append(msg.model_dump())
    messages.append({
        "role": "tool",
        "tool_call_id": tc_id,
        "content": json.dumps([
            {"name": "train.csv", "size": "61180"},
            {"name": "test.csv", "size": "28629"},
            {"name": "gender_submission.csv", "size": "3258"},
        ]),
    })

    # Turn 2: model summarizes
    response2 = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}},  # disable thinking for summary
    )
    msg2 = response2.choices[0].message
    reply = msg2.content or getattr(msg2, "reasoning_content", None) or ""

    assert reply and len(reply) > 20, f"Turn 2 reply too short: '{reply}'"
    assert "train" in reply.lower(), f"Reply should mention train.csv: {reply}"

    print(f"  ✓ test_multi_turn — model response: {reply[:80]}...")


def test_code_generation():
    """Test: model can generate a simple training script from a prompt."""
    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a machine learning engineer. Write clean, working Python code.",
        },
        {
            "role": "user",
            "content": (
                "Write a Python function `load_titanic_data(data_dir: str)` that:\n"
                "1. Reads train.csv and test.csv from data_dir using pandas\n"
                "2. Returns (train_df, test_df) as a tuple\n"
                "3. Includes error handling for missing files\n"
                "Only output the function, no explanation."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        max_tokens=500,
        extra_body={"thinking": {"type": "disabled"}},  # disable thinking for code gen
    )

    msg = response.choices[0].message
    code = msg.content

    # DeepSeek may return content in reasoning_content when thinking is enabled
    if not code:
        code = getattr(msg, "reasoning_content", None) or ""

    # Verify it's valid Python
    assert code, (
        f"Empty response. reasoning_content={getattr(msg, 'reasoning_content', 'N/A')[:100]}"
    )
    assert "def load_titanic_data" in code, f"Missing function definition: {code[:100]}"
    assert "import pandas" in code or "pd." in code, f"Missing pandas: {code[:100]}"

    # Try to compile (syntax check)
    try:
        compile(code, "<test>", "exec")
    except SyntaxError:
        # Sometimes the model wraps in ```python — try stripping
        if "```" in code:
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
            compile(code.strip(), "<test>", "exec")
        else:
            raise AssertionError(f"Generated code has syntax error")

    print(f"  ✓ test_code_generation — valid Python, {len(code.splitlines())} lines")


def main():
    print("DeepSeek V4 Pro Tool Calling Verification\n")
    print(f"  Model: {config.DEEPSEEK_MODEL}")
    masked = config.DEEPSEEK_API_KEY[:8] + "..." + config.DEEPSEEK_API_KEY[-4:]
    print(f"  Key: {masked}")
    print(f"  Base URL: {config.DEEPSEEK_BASE_URL}")
    print()

    # Pre-flight: check key is set
    if not config.DEEPSEEK_API_KEY:
        print("[FAIL] DEEPSEEK_API_KEY is not set.")
        print("Create a .env file in the project root from .env.example")
        return False

    try:
        test_single_tool_call()
        test_multi_tool_selection()
        test_tool_call_with_structured_output()
        test_code_generation()
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        return False

    print("\n✅ All DeepSeek V4 Pro tool calling tests passed!")
    print("   You're ready to build the LangGraph agent.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
