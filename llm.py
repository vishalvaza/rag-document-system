"""LLM abstraction layer using LangChain for multiple providers (OpenAI, Ollama)."""
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel


def get_llm(provider_name: str = None, temperature: float = None) -> BaseChatModel:
    """
    Get a LangChain chat model based on provider name.

    Args:
        provider_name: "openai" or "ollama". If None, uses config.
        temperature: Override temperature. If None, uses config.

    Returns:
        LangChain BaseChatModel instance (ChatOpenAI or ChatOllama).

    Raises:
        ValueError: If provider is not configured or unavailable.
    """
    from config import (
        LLM_PROVIDER, LLM_TEMPERATURE,
        OPENAI_API_KEY, OPENAI_MODEL,
        OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY
    )

    if provider_name is None:
        provider_name = LLM_PROVIDER.lower()

    if temperature is None:
        temperature = LLM_TEMPERATURE

    if provider_name == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file."
            )

        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )

    elif provider_name == "ollama":
        # Check if using cloud API (has API key)
        if OLLAMA_API_KEY:
            # Cloud mode with API key
            return ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
                api_key=OLLAMA_API_KEY
            )
        else:
            # Local mode without API key
            return ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature
            )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            "Use 'openai' or 'ollama'."
        )


def list_available_providers() -> Dict[str, bool]:
    """
    Check which LLM providers are available.

    Returns:
        Dictionary mapping provider names to availability status.
    """
    from config import (
        OPENAI_API_KEY,
        OLLAMA_API_KEY
    )

    providers = {}

    # Check OpenAI
    providers["openai"] = bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here")

    # Check Ollama (either cloud with API key or local)
    # For simplicity, assume available if configured
    providers["ollama"] = True  # Will fail at runtime if not actually available

    return providers


def get_llm_name(provider_name: str = None) -> str:
    """
    Get a human-readable name for the LLM provider.

    Args:
        provider_name: "openai" or "ollama". If None, uses config.

    Returns:
        Human-readable provider name.
    """
    from config import (
        LLM_PROVIDER, OPENAI_MODEL, OLLAMA_MODEL, OLLAMA_API_KEY
    )

    if provider_name is None:
        provider_name = LLM_PROVIDER.lower()

    if provider_name == "openai":
        return f"OpenAI ({OPENAI_MODEL})"
    elif provider_name == "ollama":
        mode = "Cloud" if OLLAMA_API_KEY else "Local"
        return f"Ollama {mode} ({OLLAMA_MODEL})"
    else:
        return provider_name
