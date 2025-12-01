"""
Generic LLM player that can be used across different game types.
Supports Anthropic, OpenAI, and Gemini models.
"""

from dataclasses import dataclass
import anthropic
import openai
import requests

from .config import get_api_key


@dataclass
class PlayerResponse:
    """Response from an LLM player."""
    raw_response: str
    thinking: str = ""


def get_llm_response(
    provider: str,
    model: str,
    prompt: str,
    api_key: str = None
) -> PlayerResponse:
    """
    Get a response from an LLM.

    Args:
        provider: 'anthropic', 'openai', or 'gemini'
        model: The model name to use
        prompt: The full prompt to send
        api_key: Optional API key (uses env var if not provided)

    Returns:
        PlayerResponse with raw_response and thinking
    """
    if api_key is None:
        api_key = get_api_key(provider)

    thinking = ""
    raw = ""

    if provider == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 8000},
            messages=[{"role": "user", "content": prompt}]
        )
        for block in response.content:
            if block.type == "thinking":
                thinking = block.thinking
            elif block.type == "text":
                raw = block.text

    elif provider == "openai":
        client = openai.OpenAI(api_key=api_key)
        # Use responses API for GPT-5 models to get reasoning
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "medium"},
        )

        # Extract reasoning and text from response
        thinking = ""
        raw = ""
        for item in response.output:
            if item.type == "reasoning":
                # Reasoning summary is in item.summary (list of objects with text)
                if hasattr(item, 'summary') and item.summary:
                    thinking = "\n".join(s.text for s in item.summary if hasattr(s, 'text') and s.text)
            elif item.type == "message":
                # Message content is in item.content (list of objects with text attribute)
                if hasattr(item, 'content') and item.content:
                    raw = "\n".join(c.text for c in item.content if hasattr(c, 'text') and c.text)

        if not thinking:
            thinking = "(Reasoning not available - GPT-5 models do not expose raw reasoning)"
        if not raw:
            raw = "(No response)"

    elif provider == "gemini":
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'
        data = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'thinkingConfig': {'includeThoughts': True}
            }
        }
        response = requests.post(url, json=data)
        thinking = ""
        raw = ""
        if response.status_code == 200:
            result = response.json()
            parts = result['candidates'][0]['content']['parts']
            for part in parts:
                if part.get('thought'):
                    thinking = part.get('text', '')
                elif 'text' in part:
                    raw = part['text']
        else:
            raw = f"Error: {response.status_code}"
        if not thinking:
            thinking = "(No thinking available)"

    return PlayerResponse(raw_response=raw.strip(), thinking=thinking)


class LLMPlayer:
    """
    A generic LLM player that can be configured for any game type.

    Usage:
        player = LLMPlayer(
            player_id=1,
            model="claude-sonnet-4-5-20250929",
            provider="anthropic",
            system_prompt="You are playing a game..."
        )
        response = player.get_response("Current game state...")
    """

    def __init__(
        self,
        player_id: int,
        model: str,
        provider: str,
        system_prompt: str,
        api_key: str = None
    ):
        self.player_id = player_id
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.api_key = api_key or get_api_key(provider)

    def get_response(self, user_message: str) -> PlayerResponse:
        """
        Get a response from the LLM for a given game state.

        Args:
            user_message: The current game state / prompt

        Returns:
            PlayerResponse with raw_response and thinking
        """
        full_prompt = f"{self.system_prompt}\n\n{user_message}"
        return get_llm_response(
            provider=self.provider,
            model=self.model,
            prompt=full_prompt,
            api_key=self.api_key
        )
