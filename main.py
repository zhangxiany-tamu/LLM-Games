#!/usr/bin/env python3
"""
LLM Game Theory Arena - Main Launcher

Run different game theory experiments with LLMs.
"""

import argparse
import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


GAMES = {
    "matrix_2x2": {
        "name": "2x2 Matrix Games",
        "description": "Classic 2x2 games: Battle of the Sexes, Prisoner's Dilemma, etc.",
        "module": "games.matrix_2x2"
    },
    "pricing": {
        "name": "Algorithmic Pricing",
        "description": "Bertrand duopoly - investigating tacit collusion in LLM pricing agents",
        "module": "games.pricing"
    },
    "auctions": {
        "name": "Sealed-Bid Auctions",
        "description": "First-price and second-price auctions with independent private values",
        "module": "games.auctions"
    },
    # Future games can be added here:
    # "bargaining": {
    #     "name": "Bargaining Games",
    #     "description": "Ultimatum game, Nash bargaining, etc.",
    #     "module": "games.bargaining"
    # },
}


def list_games():
    """Print available games."""
    print("\nAvailable Games:")
    print("-" * 50)
    for game_id, info in GAMES.items():
        print(f"  {game_id:15} - {info['name']}")
        print(f"                    {info['description']}")
    print()


def create_app() -> FastAPI:
    """Create the main FastAPI app with home page and game routes."""
    app = FastAPI(title="LLM Game Theory Arena")

    # Get the directory where this file is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "static")

    # Import and mount game apps
    from games.matrix_2x2.server import create_app as create_matrix_app, manager as matrix_manager
    matrix_app = create_matrix_app()
    app.mount("/games/matrix_2x2", matrix_app)

    from games.pricing.server import create_app as create_pricing_app, manager as pricing_manager
    pricing_app = create_pricing_app()
    app.mount("/games/pricing", pricing_app)

    from games.auctions.server import create_app as create_auctions_app, manager as auctions_manager
    auctions_app = create_auctions_app()
    app.mount("/games/auctions", auctions_app)

    # Serve home page
    @app.get("/")
    async def home():
        return FileResponse(os.path.join(static_dir, "index.html"))

    # Mount static files for home page
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


def run_app(host: str = "0.0.0.0", port: int = 8000):
    """Run the main app with all games."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)


def run_game(game_id: str, host: str = "0.0.0.0", port: int = 8000):
    """Run a specific game server directly (legacy mode)."""
    if game_id not in GAMES:
        print(f"Error: Unknown game '{game_id}'")
        list_games()
        sys.exit(1)

    game_info = GAMES[game_id]
    print(f"\nStarting {game_info['name']}...")
    print(f"Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    # Import and run the game module
    if game_id == "matrix_2x2":
        from games.matrix_2x2 import run
        run(host=host, port=port)
    elif game_id == "pricing":
        from games.pricing import run
        run(host=host, port=port)
    elif game_id == "auctions":
        from games.auctions import run
        run(host=host, port=port)
    # Add other games here as they are implemented


def main():
    parser = argparse.ArgumentParser(
        description="LLM Game Theory Arena - Run game theory experiments with LLMs"
    )
    parser.add_argument(
        "game",
        nargs="?",
        default=None,
        help="Game to run directly (optional, runs full arena by default)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available games"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to run on (default: 8000)"
    )

    args = parser.parse_args()

    if args.list:
        list_games()
        return

    if args.game:
        # Run specific game directly
        run_game(args.game, host=args.host, port=args.port)
    else:
        # Run full arena with home page
        print("\nStarting LLM Game Theory Arena...")
        print(f"Server running at http://localhost:{args.port}")
        print("Press Ctrl+C to stop\n")
        run_app(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
