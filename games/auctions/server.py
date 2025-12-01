import asyncio
import os
import json
import uuid
from typing import Optional, List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from shared.llm_player import get_llm_response
from shared.config import get_model_info
from games.auctions.game import (
    GameState, AuctionConfig, AuctionFormat, RoundResult,
    get_prompt, format_history, format_round_start, parse_bid_choice,
    play_round, compute_theoretical_bid
)

class AuctionGameManager:
    """Game manager for sealed-bid auctions."""
    def __init__(self):
        self.connections: List[WebSocket] = []
        self.game_running = False
        self.stop_game = False
        # Separate events and choices for each bidder (for human vs human)
        self.human_choice_events = {1: asyncio.Event(), 2: asyncio.Event()}
        self.human_choices: Dict[int, Optional[int]] = {1: None, 2: None}
        self.current_game_task: Optional[asyncio.Task] = None
        self.current_game_id: Optional[str] = None # Unique ID for the current game

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Inject current_game_id if available and not already in message
        if self.current_game_id and "game_id" not in message:
            message["game_id"] = self.current_game_id
            
        for connection in self.connections:
            try:
                await connection.send_json(message)
            except:
                pass

    async def run_game(self, b1_model: str, b2_model: str, rounds: int = 30, config_params: dict = None, game_id: str = None):
        self.game_running = True
        self.stop_game = False
        self.human_choices = {1: None, 2: None} # Reset human choices
        self.human_choice_events[1].clear()
        self.human_choice_events[2].clear()
        
        # Setup config
        config = AuctionConfig()
        if config_params:
            if "auction_format" in config_params:
                try:
                    config.auction_format = AuctionFormat(config_params["auction_format"])
                except ValueError:
                    pass # Keep default
            if "max_value" in config_params:
                config.max_value = int(config_params["max_value"])
            if "bid_step" in config_params:
                config.bid_step = int(config_params["bid_step"])
            # Re-init to update bid grid
            config.__post_init__()

        state = GameState(total_rounds=rounds, config=config)

        # Notify start
        await self.broadcast({
            "type": "game_start",
            "game_id": game_id,
            "b1_model": b1_model,
            "b2_model": b2_model,
            "total_rounds": rounds,
            "bid_grid": config.bid_grid,
            "format_name": config.format_name,
            "payment_rule": config.payment_rule,
            "auction_format": config.auction_format.value
        })

        while not state.is_complete and not self.stop_game:
            state.start_new_round()
            
            # Only send each bidder their OWN value - private values are hidden until after bids
            # We track which model is human to send appropriate info
            for connection in self.connections:
                try:
                    await connection.send_json({
                        "type": "round_start",
                        "round": state.current_round,
                        # Only send b1_value if b1 is human, b2_value if b2 is human
                        # For LLM vs LLM or spectator mode, don't reveal values upfront
                        "b1_value": state.current_b1_value if b1_model == "human" else None,
                        "b2_value": state.current_b2_value if b2_model == "human" else None
                    })
                except:
                    pass
            
            # Helper for getting bid
            async def get_bid(p_num, model):
                if model == "human":
                    # Wait for human input - use per-bidder event/choice
                    self.human_choice_events[p_num].clear()
                    await self.broadcast({
                        "type": "human_turn",
                        "bidder": p_num,
                        "value": state.current_b1_value if p_num == 1 else state.current_b2_value
                    })
                    # Loop to check stop_game while waiting
                    while not self.human_choice_events[p_num].is_set():
                        if self.stop_game:
                            return None, None
                        await asyncio.sleep(0.1)
                    return self.human_choices[p_num], "Human Player (Manual Bid)"
                else:
                    await self.broadcast({"type": "thinking", "bidder": p_num})
                    system_prompt = get_prompt(p_num, state.config, state.total_rounds)
                    history_str = format_history(state, p_num)
                    round_prompt = format_round_start(state, p_num)
                    full_prompt = f"{system_prompt}\n\n{history_str}\n\n{round_prompt}"

                    provider, actual_model = get_model_info(model)
                    
                    # Last second check before making the expensive/slow network call
                    if self.stop_game:
                        return None, None

                    # Run sync LLM call in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, lambda: get_llm_response(provider, actual_model, full_prompt)
                    )
                    bid_idx = parse_bid_choice(response.raw_response, state.config)
                    return bid_idx, response.thinking

            # Execute bids
            task1 = asyncio.create_task(get_bid(1, b1_model))
            task2 = asyncio.create_task(get_bid(2, b2_model))
            
            try:
                b1_res, b2_res = await asyncio.gather(task1, task2)
            except asyncio.CancelledError:
                # Ensure sub-tasks are cancelled immediately when the game is stopped
                task1.cancel()
                task2.cancel()
                raise

            if self.stop_game or b1_res[0] is None or b2_res[0] is None:
                break
                
            b1_idx, b1_thinking = b1_res
            b2_idx, b2_thinking = b2_res

            # Calculate theoretical bids
            b1_theo = compute_theoretical_bid(state.current_b1_value, config)
            b2_theo = compute_theoretical_bid(state.current_b2_value, config)

            # IMPORTANT: Only reveal both bids/values AFTER both players have submitted
            # This is critical for sealed-bid auction integrity
            # Send both choices together so opponent info is only revealed after bidding
            await self.broadcast({
                "type": "bids_revealed",
                "round": state.current_round, # Critical fix: Add missing round field
                "b1_bid": config.bid_grid[b1_idx],
                "b1_value": state.current_b1_value,
                "b1_thinking": b1_thinking,
                "b1_theoretical_bid": b1_theo,
                "b2_bid": config.bid_grid[b2_idx],
                "b2_value": state.current_b2_value,
                "b2_thinking": b2_thinking,
                "b2_theoretical_bid": b2_theo
            })

            await asyncio.sleep(1) # Suspense
            
            # Resolve round
            result = play_round(state, b1_idx, b2_idx)
            
            await self.broadcast({
                "type": "round_result",
                "round": result.round_num,
                "winner": result.winner,
                "price_paid": result.price_paid,
                "b1_value": result.b1_value,
                "b1_bid": result.b1_bid,
                "b1_profit": result.b1_profit,
                "b2_value": result.b2_value,
                "b2_bid": result.b2_bid,
                "b2_profit": result.b2_profit,
                "efficient": result.efficient,
                "b1_total": state.b1_total_profit,
                "b2_total": state.b2_total_profit,
                "efficiency_rate": state.efficiency_rate
            })
            
            await asyncio.sleep(2) # Read time

        if not self.stop_game:
            winner = 1 if state.b1_total_profit > state.b2_total_profit else (
                2 if state.b2_total_profit > state.b1_total_profit else 0)
            
            await self.broadcast({
                "type": "game_end",
                "winner": winner,
                "b1_total": state.b1_total_profit,
                "b2_total": state.b2_total_profit,
                "efficiency_rate": state.efficiency_rate,
                "b1_avg_shading": state.get_avg_bid_shading(1),
                "b2_avg_shading": state.get_avg_bid_shading(2)
            })
        
        self.game_running = False

    async def stop_current_game(self):
        """Stop the current game and notify clients."""
        # Cancel the actual asyncio task if it exists
        if self.current_game_task and not self.current_game_task.done():
            self.current_game_task.cancel()
            try:
                await self.current_game_task
            except asyncio.CancelledError:
                pass
            self.current_game_task = None

        if self.game_running:
            self.stop_game = True
            # Wake up any waiting human input (both bidders)
            self.human_choice_events[1].set()
            self.human_choice_events[2].set()
            await self.broadcast({"type": "game_stopped"})
            self.game_running = False
            # Clear game ID so any further stray messages aren't tagged with a valid ID
            self.current_game_id = None

    async def handle_message(self, data: dict):
        if data["type"] == "start_game":
            # Stop any existing game first
            await self.stop_current_game()
            # Give explicit time for tasks to cancel and cleanup to happen
            # This prevents the race where the old loop sends a message just as the new one starts
            await asyncio.sleep(0.2)
            
            # Generate unique game ID for this session
            game_id = str(uuid.uuid4())
            self.current_game_id = game_id
            
            # Start new game task and track it
            self.current_game_task = asyncio.create_task(self.run_game(
                data.get("b1_model", "claude-4.5-sonnet"),
                data.get("b2_model", "claude-4.5-sonnet"),
                data.get("rounds", 30),
                data.get("config"),
                game_id=game_id
            ))
        elif data["type"] == "stop_game":
            await self.stop_current_game()
        elif data["type"] == "human_choice":
            bidder = data.get("bidder", 1)  # Default to bidder 1 for backwards compatibility
            self.human_choices[bidder] = data.get("bid_index")
            self.human_choice_events[bidder].set()

manager = AuctionGameManager()

def create_app() -> FastAPI:
    app = FastAPI(title="Sealed-Bid Auction Game")
    
    # Get absolute path to static directory
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
        except Exception as e:
            print(f"Error: {e}")
            manager.disconnect(ws)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    return app

# For debugging/running directly
def run(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port)

if __name__ == "__main__":
    run()
