"""LLM client — provider-agnostic wrapper.

Supports any OpenAI-compatible API: DeepSeek, OpenAI, Anthropic (via proxy),
Ollama, vLLM, Groq, etc. Just set LLM_PROVIDER in .env.

Handles model selection (main vs flash), thinking mode (provider-specific),
and transforms between LangChain's ChatOpenAI and raw OpenAI client.
"""

from __future__ import annotations

from kagglemate.config import config


def _is_deepseek() -> bool:
    return config.LLM_PROVIDER == "deepseek"


def _require_openai():
    """Lazy import the OpenAI SDK with a helpful error message."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK is required for LLM features. "
            "Install with: pip install -e '.[llm]'"
        ) from exc
    return OpenAI


def _require_chat_openai():
    """Lazy import LangChain's ChatOpenAI with a helpful error message."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain dependencies are required for LLM features. "
            "Install with: pip install -e '.[llm]'"
        ) from exc
    return ChatOpenAI


def get_llm(use_flash: bool = False):
    """Get a LangChain-compatible ChatOpenAI instance.

    Args:
        use_flash: If True, use the cheaper/faster model (e.g. V4 Flash, GPT-4.1-mini).
                   Defaults to the main model.
    """
    ChatOpenAI = _require_chat_openai()
    model = config.LLM_FLASH_MODEL if use_flash else config.LLM_MODEL

    kwargs = dict(
        model=model,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0.0,
        max_tokens=4096,
        timeout=120,
        max_retries=2,
    )

    # DeepSeek: disable thinking by default for tool calling
    if _is_deepseek():
        kwargs["model_kwargs"] = {"extra_body": {"thinking": {"type": "disabled"}}}

    return ChatOpenAI(**kwargs)


def get_llm_with_thinking():
    """Get an LLM with extended reasoning (provider-specific).

    Only DeepSeek supports a native thinking mode.
    For other providers, this returns the standard llm.
    """
    ChatOpenAI = _require_chat_openai()
    if _is_deepseek():
        return ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            temperature=0.0,
            max_tokens=8192,
            timeout=180,
            max_retries=2,
            model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
        )
    # Other providers: just use the standard llm with higher max_tokens
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0.0,
        max_tokens=8192,
        timeout=180,
        max_retries=2,
    )


def get_raw_client():
    """Get the raw OpenAI client for fine-grained control."""
    OpenAI = _require_openai()
    return OpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        timeout=180,
    )


def simple_prompt(prompt: str, use_flash: bool = False) -> str:
    """Send a single prompt and return the text response. No tool calls.

    The simplest possible LLM call — for when you just need text back.
    """
    OpenAI = _require_openai()
    client = OpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        timeout=180,
    )
    model = config.LLM_FLASH_MODEL if use_flash else config.LLM_MODEL

    extra_body = {}
    if _is_deepseek():
        extra_body = {"thinking": {"type": "disabled"}}

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
        extra_body=extra_body,
    )
    msg = response.choices[0].message
    return msg.content or getattr(msg, "reasoning_content", "") or ""
