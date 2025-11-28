"""
2x2 Matrix Games - Game Logic
Supports Battle of the Sexes, Prisoner's Dilemma, and other 2x2 games.
"""

from dataclasses import dataclass, field
from enum import Enum


class Choice(Enum):
    A = "A"
    B = "B"


DEFAULT_PAYOFFS = {
    (Choice.A, Choice.A): (3, 2),  # Battle of the Sexes default
    (Choice.A, Choice.B): (0, 0),
    (Choice.B, Choice.A): (0, 0),
    (Choice.B, Choice.B): (2, 3),
}


def build_payoffs(matrix: dict = None) -> dict:
    """Build payoff dict from matrix or use defaults."""
    if not matrix:
        return DEFAULT_PAYOFFS
    return {
        (Choice.A, Choice.A): (matrix["AA"][0], matrix["AA"][1]),
        (Choice.A, Choice.B): (matrix["AB"][0], matrix["AB"][1]),
        (Choice.B, Choice.A): (matrix["BA"][0], matrix["BA"][1]),
        (Choice.B, Choice.B): (matrix["BB"][0], matrix["BB"][1]),
    }


@dataclass
class GameState:
    total_rounds: int = 10
    history: list = field(default_factory=list)
    p1_total: int = 0
    p2_total: int = 0
    payoff_matrix: dict = None

    def __post_init__(self):
        self.payoffs = build_payoffs(self.payoff_matrix)

    @property
    def current_round(self):
        return len(self.history) + 1

    @property
    def is_complete(self):
        return len(self.history) >= self.total_rounds


@dataclass
class RoundResult:
    round_num: int
    p1_choice: Choice
    p2_choice: Choice
    p1_score: int
    p2_score: int


@dataclass
class MatrixPlayerResponse:
    choice: Choice
    thinking: str
    raw_response: str


def get_prompt(player_num: int, total_rounds: int, payoff_matrix: dict = None) -> str:
    """Generate the game prompt for a player."""
    payoffs = build_payoffs(payoff_matrix)
    aa = payoffs[(Choice.A, Choice.A)]
    ab = payoffs[(Choice.A, Choice.B)]
    ba = payoffs[(Choice.B, Choice.A)]
    bb = payoffs[(Choice.B, Choice.B)]

    if player_num == 1:
        payoff = f"If both choose A: you get {aa[0]}, opponent gets {aa[1]}. If both choose B: you get {bb[0]}, opponent gets {bb[1]}."
        mismatch = f"If you choose A and opponent B: you get {ab[0]}, opponent gets {ab[1]}. If you choose B and opponent A: you get {ba[0]}, opponent gets {ba[1]}."
    else:
        payoff = f"If both choose A: you get {aa[1]}, opponent gets {aa[0]}. If both choose B: you get {bb[1]}, opponent gets {bb[0]}."
        mismatch = f"If you choose A and opponent B: you get {ab[1]}, opponent gets {ab[0]}. If you choose B and opponent A: you get {ba[1]}, opponent gets {ba[0]}."

    return f"""You are Player {player_num} in a 2x2 strategic game.

Rules:
- Both players choose A or B simultaneously
- {payoff}
- {mismatch}

You will play {total_rounds} rounds. Maximize YOUR total score.
Think strategically about coordination and your opponent's behavior.
End your response with your final choice: just "A" or "B" on its own line."""


def format_history(state: GameState, player_num: int) -> str:
    """Format game history from player's perspective."""
    if not state.history:
        return "No rounds played yet."

    lines = ["Previous rounds:"]
    for r in state.history:
        my_choice = r.p1_choice if player_num == 1 else r.p2_choice
        opp_choice = r.p2_choice if player_num == 1 else r.p1_choice
        my_score = r.p1_score if player_num == 1 else r.p2_score
        lines.append(f"  R{r.round_num}: You={my_choice.name}, Opp={opp_choice.name} -> You: +{my_score}")

    my_total = state.p1_total if player_num == 1 else state.p2_total
    opp_total = state.p2_total if player_num == 1 else state.p1_total
    lines.append(f"\nTotals - You: {my_total}, Opponent: {opp_total}")
    return "\n".join(lines)


def parse_choice(raw: str) -> Choice:
    """Parse choice from model response."""
    lines = raw.strip().upper().split('\n')
    last = lines[-1].strip()

    if last == "A":
        return Choice.A
    if last == "B":
        return Choice.B
    # Look at the last few characters for A or B
    if last.endswith("A") or last == "CHOICE: A" or last == "A.":
        return Choice.A
    if last.endswith("B") or last == "CHOICE: B" or last == "B.":
        return Choice.B
    # Count occurrences in the last line
    a_count = last.count('A')
    b_count = last.count('B')
    if a_count > b_count:
        return Choice.A
    return Choice.B  # Default


def play_round(state: GameState, p1_choice: Choice, p2_choice: Choice) -> RoundResult:
    """Play a round and update state."""
    p1_score, p2_score = state.payoffs[(p1_choice, p2_choice)]
    result = RoundResult(state.current_round, p1_choice, p2_choice, p1_score, p2_score)
    state.history.append(result)
    state.p1_total += p1_score
    state.p2_total += p2_score
    return result
