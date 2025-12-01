"""
Algorithmic Pricing Game - Bertrand Duopoly with LLM Agents

Based on research investigating whether LLM-based pricing agents
can learn to tacitly collude in repeated pricing games.
"""

from .server import run, create_app

__all__ = ["run", "create_app"]
