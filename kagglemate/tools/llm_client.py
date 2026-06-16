"""LLM client — unified DeepSeek API wrapper.

Handles model selection (Pro vs Flash), thinking mode toggling,
and transforms between LangChain's ChatOpenAI and raw OpenAI client.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from openai import OpenAI

from kagglemate.config import config


def get_llm(use_flash: bool = False) -> ChatOpenAI:
    """Get a LangChain-compatible ChatOpenAI instance pointing at DeepSeek.

    Args:
        use_flash: If True, use deepseek-v4-flash (cheaper, faster).
                   Defaults to deepseek-v4-pro.
    """
    model = config.DEEPSEEK_FLASH_MODEL if use_flash else config.DEEPSEEK_MODEL

    return ChatOpenAI(
        model=model,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=4096,
        timeout=120,
        max_retries=2,
    )


def get_llm_with_thinking() -> ChatOpenAI:
    """Get a LangChain LLM with thinking enabled (better for code/math).

    Uses deepseek-v4-pro with the thinking reasoning block.
    """
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=8192,
        timeout=180,
        max_retries=2,
        model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
    )


def get_raw_client() -> OpenAI:
    """Get the raw OpenAI client for fine-grained control over tool calls."""
    return OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        timeout=180,
    )


def simple_prompt(prompt: str, use_flash: bool = False) -> str:
    """Send a single prompt and return the text response. No tool calls.

    The simplest possible LLM call — for when you just need text back.
    """
    client = get_raw_client()
    response = client.chat.completions.create(
        model=config.DEEPSEEK_FLASH_MODEL if use_flash else config.DEEPSEEK_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},
    )
    msg = response.choices[0].message
    return msg.content or getattr(msg, "reasoning_content", "") or ""
