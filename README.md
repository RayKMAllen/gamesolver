# gamesolver

Board-game solver by reinforcement-learning self-play. A convolutional
Q-network plays both sides of a game against itself, stores the transitions in
a prioritised replay buffer and trains on them (double Q-learning). TicTacToe
learns to play; Checkers has the full rules and random play and is the next
target. The long-horizon goal is a general solver: a new game is a short
encoding of its state and move spaces, and the learning algorithm is a
selectable component.

Python, PyTorch (torch-directml for the GPU on Windows), NumPy, matplotlib.
Personal research project; everything runs from one script.

## Run

    pip install numpy torch torch-directml matplotlib nest_asyncio
    python dev.py

`dev.py` builds the network and two players sharing one agent, then loops:
train 100 batches, play 100 self-play games, and every tenth round test the
greedy policy as X and as O against the exploring one. Loss curves are plotted
with matplotlib. Hyper-parameters (network shape, gamma, prioritised replay
on or off, epsilon) are set in the script just above the loop.

Files:

- `game.py`: `Game`, `Player` and `Board` abstractions, `TicTacToe`, `Checkers`
- `agent.py`: `Agent` base class and `RandomAgent`; `DLAgent` lives in `dev.py`
- `history.py`: replay buffer with optional prioritised experience replay
  (sum tree adapted from Howuhh/prioritized_experience_replay)
- `utils.py`: index helpers, one-hot masks, a small profiler
- `notes.txt`: requirements and open questions

## Roadmap

- A more comprehensive abstract game framework: simple encoding of complex
  games, with potentially complex input and action spaces.
- Make the learning algorithm configurable and selectable (DQN, PPO,
  AlphaZero-style, ...) instead of the single double-Q agent.
- Train on Checkers, then larger games.
- Shared or separate history per player (open question in `notes.txt`).

## Licence

MIT, see `LICENSE`.
