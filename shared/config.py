"""
Configuration for LLM providers and API keys.
"""

import os

# API Keys - load from environment variables
API_KEYS = {
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "gemini": os.environ.get("GEMINI_API_KEY", ""),
}

# Model configurations: model_id -> (provider, actual_model_name)
MODELS = {
    "claude-4.5-sonnet": ("anthropic", "claude-sonnet-4-5-20250929"),
    "gpt-5": ("openai", "gpt-5"),
    "gpt-5.1": ("openai", "gpt-5.1"),
    "gemini-2.5-pro": ("gemini", "gemini-2.5-pro"),
    "gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
    "gemini-3-pro": ("gemini", "gemini-3-pro-preview"),
}

def get_model_info(model_id: str) -> tuple:
    """Get provider and model name for a model ID."""
    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}")
    return MODELS[model_id]

def get_api_key(provider: str) -> str:
    """Get API key for a provider."""
    if provider not in API_KEYS:
        raise ValueError(f"Unknown provider: {provider}")
    return API_KEYS[provider]
