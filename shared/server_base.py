"""
Base WebSocket server for game experiments.
Provides common functionality for all game types.
"""

import asyncio
from abc import ABC, abstractmethod
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import MODELS, get_api_key


class BaseGameManager(ABC):
    """
    Abstract base class for game managers.
    Subclasses implement specific game logic.
    """

    def __init__(self):
        self.connections = []
        self.game_running = False
        self.human_choice_event = None
        self.human_choice = None

    async def connect(self, ws: WebSocket):
        """Accept and register a WebSocket connection."""
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        """Remove a WebSocket connection."""
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg: dict):
        """Broadcast a message to all connected clients."""
        for ws in self.connections:
            try:
                await ws.send_json(msg)
            except:
                pass

    def set_human_choice(self, choice):
        """Called when human submits their choice."""
        self.human_choice = choice
        if self.human_choice_event:
            self.human_choice_event.set()

    async def wait_for_human_choice(self):
        """Wait for human to make a choice."""
        self.human_choice_event = asyncio.Event()
        self.human_choice = None
        await self.human_choice_event.wait()
        return self.human_choice

    @abstractmethod
    async def run_game(self, **kwargs):
        """Run the game. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def handle_message(self, data: dict):
        """Handle incoming WebSocket messages. Must be implemented by subclasses."""
        pass


def create_game_app(
    manager: BaseGameManager,
    static_dir: str,
    index_file: str = "index.html"
) -> FastAPI:
    """
    Create a FastAPI app for a game.

    Args:
        manager: The game manager instance
        static_dir: Path to the static files directory
        index_file: Name of the main HTML file

    Returns:
        Configured FastAPI app
    """
    app = FastAPI()

    @app.get("/")
    async def root():
        return FileResponse(f"{static_dir}/{index_file}")

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


def run_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
