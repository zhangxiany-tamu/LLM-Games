<p align="center">
  <img src="logo.png" alt="LLM Arena Logo" width="300">
</p>

# LLM Game Theory Arena

**[Try it live](https://llm-games-253773978748.us-central1.run.app)**

A platform for running game theory experiments with Large Language Models. Watch AI agents compete in strategic games, compare behavior across models, and play against them yourself.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Open http://localhost:8000 in your browser.

## API Keys

Set at least one of these environment variables before running:

```bash
# Anthropic (Claude models)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (GPT models)
export OPENAI_API_KEY=sk-proj-...

# Google (Gemini models)
export GEMINI_API_KEY=AIza...
```

Or pass them inline:

```bash
ANTHROPIC_API_KEY=your_key OPENAI_API_KEY=your_key python main.py
```

You only need keys for the providers you want to use. Games will work with any single provider.

**Where to get keys:**
- Anthropic: https://console.anthropic.com/settings/keys
- OpenAI: https://platform.openai.com/api-keys
- Google: https://aistudio.google.com/app/apikey

## Available Games

### 2x2 Matrix Games
Classic simultaneous-move games including Battle of the Sexes, Prisoner's Dilemma, Stag Hunt, and more. Configurable payoff matrices, human vs AI play supported.

### Algorithmic Pricing
Bertrand duopoly simulation. Two firms set prices each round; consumers buy from the cheaper firm. Investigates whether LLM pricing agents develop tacit collusion.

### Sealed-Bid Auctions
First-price and second-price (Vickrey) auctions with independent private values. Tests bid shading strategies and compares LLM behavior to theoretical equilibria.

## Features

- **LLM vs LLM**: Pit different models against each other
- **Human vs LLM**: Play against AI opponents
- **Real-time visualization**: Watch strategies evolve with live charts
- **Model reasoning**: View LLM thinking process for each decision

## Project Structure

```
├── main.py          # Entry point
├── games/           # Game modules (matrix_2x2, pricing, auctions)
└── shared/          # LLM API and config
```
