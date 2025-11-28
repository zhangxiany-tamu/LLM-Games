"""
WebSocket server for 2x2 Matrix Games.
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from shared.config import MODELS, get_api_key
from shared.llm_player import LLMPlayer, PlayerResponse
from .game import GameState, Choice, play_round, get_prompt, format_history, parse_choice, MatrixPlayerResponse


class Matrix2x2Manager:
    """Game manager for 2x2 matrix games."""

    def __init__(self):
        self.connections = []
        self.game_running = False
        self.human_choice_event = None
        self.human_choice = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg: dict):
        for ws in self.connections:
            try:
                await ws.send_json(msg)
            except:
                pass

    def set_human_choice(self, choice: str):
        """Called when human submits their choice."""
        self.human_choice = choice
        if self.human_choice_event:
            self.human_choice_event.set()

    async def get_human_choice(self, player_num: int, state: GameState) -> MatrixPlayerResponse:
        """Wait for human to make a choice."""
        self.human_choice_event = asyncio.Event()
        self.human_choice = None

        await self.broadcast({
            "type": "human_turn",
            "player": player_num,
            "round": state.current_round,
            "total_rounds": state.total_rounds
        })

        await self.human_choice_event.wait()

        choice = Choice.A if self.human_choice == "A" else Choice.B
        return MatrixPlayerResponse(
            choice=choice,
            thinking="Human player",
            raw_response=f"Human chose {self.human_choice}"
        )

    async def get_llm_choice(self, player: LLMPlayer, state: GameState, player_num: int, payoff_matrix: dict) -> MatrixPlayerResponse:
        """Get choice from LLM player."""
        history = format_history(state, player_num)
        user_msg = f"{history}\n\nRound {state.current_round} of {state.total_rounds}. Choose A or B:"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, player.get_response, user_msg)

        choice = parse_choice(response.raw_response)
        return MatrixPlayerResponse(
            choice=choice,
            thinking=response.thinking,
            raw_response=response.raw_response
        )

    async def run_game(self, p1_model: str, p2_model: str, rounds: int = 10, payoff_matrix: dict = None):
        if self.game_running:
            return

        self.game_running = True
        state = GameState(total_rounds=rounds, payoff_matrix=payoff_matrix)

        p1_is_human = (p1_model == "human")
        p2_is_human = (p2_model == "human")

        player1 = None
        player2 = None

        if not p1_is_human:
            p1_provider, p1_name = MODELS[p1_model]
            system_prompt = get_prompt(1, rounds, payoff_matrix)
            player1 = LLMPlayer(1, p1_name, p1_provider, system_prompt)

        if not p2_is_human:
            p2_provider, p2_name = MODELS[p2_model]
            system_prompt = get_prompt(2, rounds, payoff_matrix)
            player2 = LLMPlayer(2, p2_name, p2_provider, system_prompt)

        await self.broadcast({
            "type": "game_start",
            "total_rounds": rounds,
            "p1_model": p1_model,
            "p2_model": p2_model
        })

        while not state.is_complete:
            await self.broadcast({"type": "round_start", "round": state.current_round})

            # Get player 1's choice
            await self.broadcast({"type": "thinking", "player": 1})
            if p1_is_human:
                p1_resp = await self.get_human_choice(1, state)
            else:
                p1_resp = await self.get_llm_choice(player1, state, 1, payoff_matrix)

            await self.broadcast({
                "type": "choice_made",
                "player": 1,
                "choice": p1_resp.choice.name,
                "thinking": p1_resp.thinking,
                "response": p1_resp.raw_response
            })

            # Get player 2's choice
            await self.broadcast({"type": "thinking", "player": 2})
            if p2_is_human:
                p2_resp = await self.get_human_choice(2, state)
            else:
                p2_resp = await self.get_llm_choice(player2, state, 2, payoff_matrix)

            await self.broadcast({
                "type": "choice_made",
                "player": 2,
                "choice": p2_resp.choice.name,
                "thinking": p2_resp.thinking,
                "response": p2_resp.raw_response
            })

            await asyncio.sleep(0.5)

            result = play_round(state, p1_resp.choice, p2_resp.choice)

            await self.broadcast({
                "type": "round_result",
                "round": result.round_num,
                "p1_choice": result.p1_choice.name,
                "p2_choice": result.p2_choice.name,
                "p1_score": result.p1_score,
                "p2_score": result.p2_score,
                "p1_total": state.p1_total,
                "p2_total": state.p2_total
            })

            await asyncio.sleep(1)

        winner = 1 if state.p1_total > state.p2_total else (2 if state.p2_total > state.p1_total else 0)
        await self.broadcast({
            "type": "game_end",
            "p1_total": state.p1_total,
            "p2_total": state.p2_total,
            "winner": winner
        })

        self.game_running = False

    async def handle_message(self, data: dict):
        """Handle incoming WebSocket messages."""
        if data.get("type") == "start_game":
            asyncio.create_task(self.run_game(
                data.get("p1_model", "claude-4.5-sonnet"),
                data.get("p2_model", "claude-4.5-sonnet"),
                data.get("rounds", 10),
                data.get("payoff_matrix")
            ))
        elif data.get("type") == "human_choice":
            self.set_human_choice(data.get("choice"))


# Module-level manager instance
manager = Matrix2x2Manager()


def create_app() -> FastAPI:
    """Create the FastAPI app for 2x2 games."""
    app = FastAPI(title="2x2 Matrix Games")

    # Get the directory where this file is located
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
