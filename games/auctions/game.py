"""
Sealed-Bid Auction Game - First-Price and Second-Price Auctions

Implements sealed-bid auctions with independent private values (IPV).
Each round, bidders receive a private value drawn uniformly and submit bids.

Key concepts:
- First-price auction: Winner pays their own bid
- Second-price (Vickrey) auction: Winner pays second-highest bid
- Bid shading: Bidding below value to increase profit margin
- Allocative efficiency: Whether highest-value bidder wins

Theoretical equilibria (2 bidders, uniform values on [0, max_value]):
- Second-price: Bid = value (dominant strategy)
- First-price: Bid = value/2 (Bayesian Nash equilibrium)
"""

from dataclasses import dataclass, field
import random
from typing import List, Optional
from enum import Enum


class AuctionFormat(Enum):
    FIRST_PRICE = "first_price"
    SECOND_PRICE = "second_price"


# Default parameters
DEFAULT_MAX_VALUE = 100  # Values drawn from [0, max_value]
DEFAULT_BID_STEP = 5     # Bid grid: 0, 5, 10, ..., max_value
DEFAULT_NUM_ROUNDS = 30


def generate_bid_grid(max_value: int = DEFAULT_MAX_VALUE,
                      step: int = DEFAULT_BID_STEP) -> List[int]:
    """Generate discrete bid grid."""
    return list(range(0, max_value + 1, step))


@dataclass
class AuctionConfig:
    """Configuration for the auction game."""
    max_value: int = DEFAULT_MAX_VALUE
    bid_step: int = DEFAULT_BID_STEP
    auction_format: AuctionFormat = AuctionFormat.FIRST_PRICE

    def __post_init__(self):
        self.bid_grid = generate_bid_grid(self.max_value, self.bid_step)
        self.num_bid_levels = len(self.bid_grid)

    @property
    def format_name(self) -> str:
        if self.auction_format == AuctionFormat.FIRST_PRICE:
            return "First-Price"
        return "Second-Price (Vickrey)"

    @property
    def payment_rule(self) -> str:
        if self.auction_format == AuctionFormat.FIRST_PRICE:
            return "Winner pays their own bid"
        return "Winner pays the second-highest bid"


def draw_value(config: AuctionConfig) -> int:
    """Draw a private value uniformly from bid grid."""
    # Draw from the bid grid to ensure values align with possible bids
    return random.choice(config.bid_grid)


@dataclass
class RoundResult:
    """Result of a single auction round."""
    round_num: int
    # Bidder 1
    b1_value: int
    b1_bid_index: int
    b1_bid: int
    b1_profit: float
    b1_won: bool
    # Bidder 2
    b2_value: int
    b2_bid_index: int
    b2_bid: int
    b2_profit: float
    b2_won: bool
    # Auction outcome
    winner: int  # 1 or 2
    price_paid: int
    efficient: bool  # Did highest-value bidder win?


@dataclass
class BidderResponse:
    """Response from a bidder."""
    bid_index: int
    bid: int
    thinking: str
    raw_response: str


@dataclass
class GameState:
    """State of the auction game."""
    total_rounds: int = DEFAULT_NUM_ROUNDS
    history: List[RoundResult] = field(default_factory=list)
    b1_total_profit: float = 0.0
    b2_total_profit: float = 0.0
    config: AuctionConfig = field(default_factory=AuctionConfig)
    # Current round values (set at round start)
    current_b1_value: Optional[int] = None
    current_b2_value: Optional[int] = None

    @property
    def current_round(self) -> int:
        return len(self.history) + 1

    @property
    def is_complete(self) -> bool:
        return len(self.history) >= self.total_rounds

    @property
    def efficiency_rate(self) -> float:
        """Fraction of rounds where highest-value bidder won."""
        if not self.history:
            return 0.0
        efficient_rounds = sum(1 for r in self.history if r.efficient)
        return efficient_rounds / len(self.history)

    @property
    def total_revenue(self) -> float:
        """Total revenue (sum of prices paid)."""
        return sum(r.price_paid for r in self.history)

    def get_avg_bid_shading(self, bidder: int) -> float:
        """Average (value - bid) / value for a bidder when they bid > 0."""
        if bidder == 1:
            relevant = [(r.b1_value, r.b1_bid) for r in self.history if r.b1_value > 0]
        else:
            relevant = [(r.b2_value, r.b2_bid) for r in self.history if r.b2_value > 0]

        if not relevant:
            return 0.0

        shading_ratios = [(v - b) / v for v, b in relevant if v > 0]
        return sum(shading_ratios) / len(shading_ratios) if shading_ratios else 0.0

    def start_new_round(self):
        """Draw new values for both bidders."""
        self.current_b1_value = draw_value(self.config)
        self.current_b2_value = draw_value(self.config)


def get_prompt(bidder_num: int, config: AuctionConfig, total_rounds: int) -> str:
    """Generate the system prompt for a bidder."""
    bids_str = ", ".join([f"${b}" for b in config.bid_grid])

    if config.auction_format == AuctionFormat.FIRST_PRICE:
        payment_explanation = """PAYMENT RULE (First-Price):
- The highest bidder wins the item
- The winner pays their OWN bid
- If you win with bid $X and value $V, your profit = $V - $X
- If you lose, your profit = $0

STRATEGIC INSIGHT:
- Bidding your true value guarantees zero profit if you win
- Bidding below your value ("bid shading") increases profit margin but risks losing
- The optimal bid depends on what you expect your opponent to bid"""
    else:
        payment_explanation = """PAYMENT RULE (Second-Price / Vickrey):
- The highest bidder wins the item
- The winner pays the SECOND-highest bid (opponent's bid)
- If you win and opponent bid $X, your profit = your value - $X
- If you lose, your profit = $0

STRATEGIC INSIGHT:
- In this auction format, bidding your true value is a dominant strategy
- You cannot increase profit by bidding above or below your value
- Your bid only affects whether you win, not what you pay"""

    return f"""You are Bidder {bidder_num} in a repeated sealed-bid auction.

AUCTION STRUCTURE:
- Each round, you receive a PRIVATE VALUE for an item (only you know your value)
- You and one opponent simultaneously submit sealed bids
- {config.payment_rule}
- Ties are broken randomly

{payment_explanation}

AVAILABLE BIDS (choose one):
{bids_str}

YOUR GOAL:
Maximize YOUR total profit over {total_rounds} rounds.
Consider: How much should you "shade" your bid below your value? What patterns do you observe in opponent behavior?

RESPONSE FORMAT:
Analyze the situation briefly, then end your response with your chosen BID AMOUNT (a number) on its own line.
Example: if you want to bid $30, end with just "30"."""


def format_history(state: GameState, bidder_num: int, last_n: int = 10) -> str:
    """Format recent auction history for a bidder."""
    if not state.history:
        return "No rounds played yet. This is the first round."

    recent = state.history[-last_n:]
    lines = [f"Recent rounds (showing last {len(recent)} of {len(state.history)}):"]

    for r in recent:
        if bidder_num == 1:
            my_value, my_bid, my_profit, i_won = r.b1_value, r.b1_bid, r.b1_profit, r.b1_won
            opp_value, opp_bid = r.b2_value, r.b2_bid
        else:
            my_value, my_bid, my_profit, i_won = r.b2_value, r.b2_bid, r.b2_profit, r.b2_won
            opp_value, opp_bid = r.b1_value, r.b1_bid

        result = "WON" if i_won else "lost"
        lines.append(
            f"  R{r.round_num}: Value=${my_value}, Bid=${my_bid} -> {result}, "
            f"Profit=${my_profit:.0f} | Opp: Value=${opp_value}, Bid=${opp_bid}"
        )

    my_total = state.b1_total_profit if bidder_num == 1 else state.b2_total_profit
    opp_total = state.b2_total_profit if bidder_num == 1 else state.b1_total_profit

    lines.append(f"\nCumulative profits - You: ${my_total:.0f}, Opponent: ${opp_total:.0f}")
    lines.append(f"Allocation efficiency so far: {state.efficiency_rate:.1%}")

    return "\n".join(lines)


def format_round_start(state: GameState, bidder_num: int) -> str:
    """Format the round start message with private value."""
    value = state.current_b1_value if bidder_num == 1 else state.current_b2_value
    return f"""Round {state.current_round} of {state.total_rounds}

YOUR PRIVATE VALUE THIS ROUND: ${value}

This is how much the item is worth to you. Your opponent has their own private value (unknown to you).
Choose your bid wisely - remember the payment rule: {state.config.payment_rule.lower()}.

What is your bid amount (e.g. 30)?"""


def parse_bid_choice(raw: str, config: AuctionConfig) -> int:
    """Parse bid amount from model response and return closest index."""
    lines = raw.strip().split('\n')
    import re

    # Helper to find closest index for a price
    def get_closest_idx(price: int) -> int:
        return min(range(len(config.bid_grid)),
                   key=lambda i: abs(config.bid_grid[i] - price))

    # Pattern for prices (e.g. $30, 30, $ 30)
    # We look for the last number in the response
    for line in reversed(lines[-5:]):
        # Look for $XX
        matches = re.findall(r'\$\s*(\d+)', line)
        if matches:
            try:
                val = int(matches[-1])
                return get_closest_idx(val)
            except:
                pass
        
        # Look for just numbers at the end of the line
        matches = re.findall(r'(?:^|\s)(\d+)(?:\s|$)', line)
        if matches:
            try:
                val = int(matches[-1])
                return get_closest_idx(val)
            except:
                pass

    # Default to middle bid if parsing fails
    return len(config.bid_grid) // 2


def play_round(state: GameState, b1_bid_index: int, b2_bid_index: int) -> RoundResult:
    """Play an auction round and update state."""
    config = state.config

    b1_value = state.current_b1_value
    b2_value = state.current_b2_value
    b1_bid = config.bid_grid[b1_bid_index]
    b2_bid = config.bid_grid[b2_bid_index]

    # Determine winner (higher bid wins, ties broken randomly)
    if b1_bid > b2_bid:
        winner = 1
    elif b2_bid > b1_bid:
        winner = 2
    else:
        winner = random.choice([1, 2])

    # Determine price based on auction format
    if config.auction_format == AuctionFormat.FIRST_PRICE:
        price_paid = b1_bid if winner == 1 else b2_bid
    else:  # Second-price
        price_paid = b2_bid if winner == 1 else b1_bid

    # Calculate profits
    if winner == 1:
        b1_profit = b1_value - price_paid
        b2_profit = 0
        b1_won, b2_won = True, False
    else:
        b1_profit = 0
        b2_profit = b2_value - price_paid
        b1_won, b2_won = False, True

    # Check efficiency (did highest-value bidder win?)
    if b1_value > b2_value:
        efficient = (winner == 1)
    elif b2_value > b1_value:
        efficient = (winner == 2)
    else:
        efficient = True  # Tie in values, either winner is efficient

    result = RoundResult(
        round_num=state.current_round,
        b1_value=b1_value,
        b1_bid_index=b1_bid_index,
        b1_bid=b1_bid,
        b1_profit=b1_profit,
        b1_won=b1_won,
        b2_value=b2_value,
        b2_bid_index=b2_bid_index,
        b2_bid=b2_bid,
        b2_profit=b2_profit,
        b2_won=b2_won,
        winner=winner,
        price_paid=price_paid,
        efficient=efficient
    )

    state.history.append(result)
    state.b1_total_profit += b1_profit
    state.b2_total_profit += b2_profit

    return result


def compute_theoretical_bid(value: int, config: AuctionConfig) -> int:
    """Compute theoretical equilibrium bid for comparison."""
    if config.auction_format == AuctionFormat.SECOND_PRICE:
        # Dominant strategy: bid = value
        theoretical = value
    else:
        # First-price with 2 bidders: bid = value/2
        theoretical = value // 2

    # Find closest bid in grid
    closest_idx = min(range(len(config.bid_grid)),
                      key=lambda i: abs(config.bid_grid[i] - theoretical))
    return config.bid_grid[closest_idx]
