"""
WebSocket server for Algorithmic Pricing Game.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from shared.config import MODELS, get_api_key
from shared.llm_player import LLMPlayer, PlayerResponse
from .game import (
    GameState, MarketConfig, PricingPlayerResponse,
    play_round, get_prompt, format_history, parse_price_choice,
    compute_nash_profit, compute_monopoly_profit
)


class PricingGameManager:
    """Game manager for algorithmic pricing game."""

    def __init__(self):
        self.connections = []
        self.game_running = False
        self.stop_game = False
        self.human_choice_event = None
        self.human_choice = None
        self.current_state = None
        self.game_task = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)
        if not self.connections and self.game_running:
            self.stop_game = True
            if self.human_choice_event:
                self.human_choice_event.set()

    async def broadcast(self, msg: dict):
        for ws in self.connections:
            try:
                await ws.send_json(msg)
            except:
                pass

    def set_human_choice(self, choice: int):
        """Called when human submits their price choice."""
        self.human_choice = choice
        if self.human_choice_event:
            self.human_choice_event.set()

    async def get_human_choice(self, player_num: int, state: GameState) -> PricingPlayerResponse:
        """Wait for human to make a price choice."""
        self.human_choice_event = asyncio.Event()
        self.human_choice = None

        await self.broadcast({
            "type": "human_turn",
            "player": player_num,
            "round": state.current_round,
            "total_rounds": state.total_rounds,
            "prices": state.config.prices
        })

        await self.human_choice_event.wait()

        # If game was stopped while waiting, return None to signal abort
        if self.stop_game:
            return None

        price_index = self.human_choice
        return PricingPlayerResponse(
            price_index=price_index,
            price=state.config.prices[price_index],
            thinking="Human player",
            raw_response=f"Human chose price index {price_index}"
        )

    async def get_llm_choice(self, player: LLMPlayer, state: GameState, player_num: int) -> PricingPlayerResponse:
        """Get price choice from LLM player."""
        history = format_history(state, player_num)
        user_msg = f"{history}\n\nRound {state.current_round} of {state.total_rounds}. Choose your price (0-{len(state.config.prices)-1}):"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, player.get_response, user_msg)

        price_index = parse_price_choice(response.raw_response, state.config)
        return PricingPlayerResponse(
            price_index=price_index,
            price=state.config.prices[price_index],
            thinking=response.thinking,
            raw_response=response.raw_response
        )

    async def run_game(self, p1_model: str, p2_model: str, rounds: int = 50, config_params: dict = None):
        # Note: handle_message now handles stopping any previous game before calling this
        self.game_running = True

        # Build market config from parameters
        config = MarketConfig(**(config_params or {}))
        state = GameState(total_rounds=rounds, config=config)
        self.current_state = state

        p1_is_human = (p1_model == "human")
        p2_is_human = (p2_model == "human")

        player1 = None
        player2 = None

        if not p1_is_human:
            p1_provider, p1_name = MODELS[p1_model]
            system_prompt = get_prompt(1, config, rounds)
            player1 = LLMPlayer(1, p1_name, p1_provider, system_prompt)

        if not p2_is_human:
            p2_provider, p2_name = MODELS[p2_model]
            system_prompt = get_prompt(2, config, rounds)
            player2 = LLMPlayer(2, p2_name, p2_provider, system_prompt)

        # Send game start with market info
        nash_profit = compute_nash_profit(config)
        mono_profit = compute_monopoly_profit(config)

        await self.broadcast({
            "type": "game_start",
            "total_rounds": rounds,
            "p1_model": p1_model,
            "p2_model": p2_model,
            "prices": config.prices,
            "cost": config.cost,
            "nash_price": config.prices[config.nash_index],
            "monopoly_price": config.prices[config.monopoly_index],
            "nash_profit": nash_profit,
            "monopoly_profit": mono_profit
        })

        while not state.is_complete and not self.stop_game:
            await self.broadcast({"type": "round_start", "round": state.current_round})

            # Get player 1's choice
            await self.broadcast({"type": "thinking", "player": 1})
            if p1_is_human:
                p1_resp = await self.get_human_choice(1, state)
                if p1_resp is None:  # Game was stopped
                    break
            else:
                p1_resp = await self.get_llm_choice(player1, state, 1)

            if self.stop_game:  # Check again after LLM call
                break

            await self.broadcast({
                "type": "choice_made",
                "player": 1,
                "price_index": p1_resp.price_index,
                "price": p1_resp.price,
                "thinking": p1_resp.thinking,
                "response": p1_resp.raw_response
            })

            # Get player 2's choice
            await self.broadcast({"type": "thinking", "player": 2})
            if p2_is_human:
                p2_resp = await self.get_human_choice(2, state)
                if p2_resp is None:  # Game was stopped
                    break
            else:
                p2_resp = await self.get_llm_choice(player2, state, 2)

            if self.stop_game:  # Check again after LLM call
                break

            await self.broadcast({
                "type": "choice_made",
                "player": 2,
                "price_index": p2_resp.price_index,
                "price": p2_resp.price,
                "thinking": p2_resp.thinking,
                "response": p2_resp.raw_response
            })

            await asyncio.sleep(0.3)

            result = play_round(state, p1_resp.price_index, p2_resp.price_index)

            await self.broadcast({
                "type": "round_result",
                "round": result.round_num,
                "p1_price": result.p1_price,
                "p2_price": result.p2_price,
                "p1_price_index": result.p1_price_index,
                "p2_price_index": result.p2_price_index,
                "p1_profit": result.p1_profit,
                "p2_profit": result.p2_profit,
                "p1_demand": result.p1_demand,
                "p2_demand": result.p2_demand,
                "p1_total": state.p1_total_profit,
                "p2_total": state.p2_total_profit,
                "profit_gain": state.profit_gain
            })

            await asyncio.sleep(0.5)

        # Only broadcast game end if game completed naturally (not stopped)
        if not self.stop_game:
            winner = 1 if state.p1_total_profit > state.p2_total_profit else (
                2 if state.p2_total_profit > state.p1_total_profit else 0)

            await self.broadcast({
                "type": "game_end",
                "p1_total": state.p1_total_profit,
                "p2_total": state.p2_total_profit,
                "winner": winner,
                "profit_gain": state.profit_gain,
                "collusion_level": "High" if state.profit_gain > 0.7 else (
                    "Moderate" if state.profit_gain > 0.3 else "Low/Competitive")
            })

        self.game_running = False

    async def stop_current_game(self):
        """Stop the current game and notify clients."""
        if self.game_running:
            self.stop_game = True
            if self.human_choice_event:
                self.human_choice_event.set()
            msg = {"type": "game_stopped"}
            if self.current_state:
                msg["stopped_at_round"] = self.current_state.current_round - 1
                msg["total_rounds"] = self.current_state.total_rounds
                msg["p1_total"] = self.current_state.p1_total_profit
                msg["p2_total"] = self.current_state.p2_total_profit
                msg["profit_gain"] = self.current_state.profit_gain
            await self.broadcast(msg)
            self.game_running = False

    async def handle_message(self, data: dict):
        """Handle incoming WebSocket messages."""
        if data.get("type") == "start_game":
            # Stop any existing game first and wait for it to finish
            if self.game_running or self.game_task:
                self.stop_game = True
                if self.human_choice_event:
                    self.human_choice_event.set()
                # Wait for existing game task to finish
                if self.game_task and not self.game_task.done():
                    try:
                        await asyncio.wait_for(self.game_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        self.game_task.cancel()
                        try:
                            await self.game_task
                        except asyncio.CancelledError:
                            pass
                self.game_task = None
                self.game_running = False

            # Now reset stop flag and start new game
            self.stop_game = False
            self.game_task = asyncio.create_task(self.run_game(
                data.get("p1_model", "claude-4.5-sonnet"),
                data.get("p2_model", "claude-4.5-sonnet"),
                data.get("rounds", 50),
                data.get("config")
            ))
        elif data.get("type") == "stop_game":
            await self.stop_current_game()
        elif data.get("type") == "human_choice":
            self.set_human_choice(data.get("price_index", 7))


manager = PricingGameManager()


def create_app() -> FastAPI:
    """Create the FastAPI app for pricing game."""
    app = FastAPI(title="Algorithmic Pricing Game")

    module_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(module_dir, "static")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await manager.connect(ws)
        try:
            while True:
                data = await ws.receive_json()
                await manager.handle_message(data)
        except WebSocketDisconnect:
            manager.disconnect(ws)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


def run(host: str = "0.0.0.0", port: int = 8000):
    """Run the server."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
