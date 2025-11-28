"""
2x2 Matrix Games module.
Includes Battle of the Sexes, Prisoner's Dilemma, and other classic 2x2 games.
"""

from .game import Choice, GameState, RoundResult, play_round, get_prompt, parse_choice
from .server import Matrix2x2Manager, create_app, run

__all__ = [
    'Choice', 'GameState', 'RoundResult', 'play_round', 'get_prompt', 'parse_choice',
    'Matrix2x2Manager', 'create_app', 'run'
]
