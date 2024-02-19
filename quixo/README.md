# Quixo
## Requirements install
    python -m pip install -r requirements.txt
## Player Classes
### My Player (MyPlayer class)
#### Description:

A general player class used as parent, can be used as random Player.\
Composed mainly of utily functions relative to the board for the children classes.

#### Methods:

`make_move(game: Game) -> tuple[tuple[int, int], Move]`: Makes a move on the game board chosing randomly over the possible moves.\
`_getPossibleMoves(player: int, board: numpy.ndarray) -> list`: Determines the possible moves for the player on the current game board.\
`__getPossibleSlides(coords: tuple[int, int]) -> tuple[Move]`: Determines the possible slide directions for a given position on the game board.\
`_hashBoard(board: numpy.ndarray) -> str`: Hashes the game board state into a string representation.\
`_getNextState(from_pos: tuple[int, int], slide: Move, playerID: int, initialBoard: numpy.ndarray) -> numpy.ndarray`: Calculates the next game board state after making a move, without modifing the actual one.\
`__playerWinner(board: numpy.ndarray, playerID: int) -> bool`: Checks if the specified player has won the game on the given board.\
`_checkWinner(board: numpy.ndarray) -> int`: Checks if there is a winner on the given board and returns its id or -1 if there's none.

### Tr_RL_Player (Tr_RL_Player class)
#### Description:

A training player that uses Monte Carlo Reinforcement Learning to make moves on the game board.

#### Attributes:
`_trajectory`: List holding all the hashed string representing the trajectory of a single game. Used to update the states based on the reward.\
`_expRate`: exploration rate, defines the rate of exploration over exploitation during the training.\
`_lRate`: learning rate, defines the extent at which the algorithm updates its policy.\
`_states`: default dictionary holding the policies of the algorithm.\

#### Methods:

`make_move(game: Game) -> tuple[tuple[int, int], Move]`: Makes a move on the game board using Monte Carlo Reinforcement Learning.\
`evaluteStates(reward: float) -> None`: Evaluates the states visited during the game trajectory and updates their values.\
`getFileName(player:int) -> str`: Generates the filename for saving/loading RL player states.\
`_loadStates() -> None`: Loads the RL player states from a file.\
`saveStates() -> None`: Saves the RL player states to a file.

### RL_Player (RL_Player class)
#### Description:

The testing version of the Tr_RL_Player class. Only uses the reinforcement learning method to choose a move.
#### Methods:

Inherits all methods from Tr_RL_Player.\
Overrides the make_move() method for improved performance avoiding the stochastic part relative to the exploration which is used into the training.

### Minimax Player (Minimax_Player class)
#### Description:

A player that uses the Minimax algorithm with alpha-beta pruning to make optimal moves on the game board.
#### Attributes:
`_mmDepth`: saves the depth set for the minimax algorithm.\
`_biasRedu`: saves the reduction value for the evaluation.\
`_evalKernel`: kernel used for the 2D convolution in the evaluation.\
`_states`: default dictionary holding the precomputed policies of the algorithm.

#### Methods:

`make_move(game: Game) -> tuple[tuple[int, int], Move]`: Makes an optimal move on the game board using the Minimax algorithm with alpha-beta pruning.\
`_minimax_alpha_beta(game: Game, board: numpy.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple[float, tuple[tuple[int,int], Move]]`: Implements the Minimax algorithm with alpha-beta pruning.\
`_evaluate(board: numpy.ndarray) -> float`: Evaluates the game board state for the Minimax algorithm.\
`getFileName(player:int) -> str`: Generates the filename for saving/loading Minimax player states.\
`_loadStates() -> None`: Loads the Minimax player states from a file.\
`saveStates() -> None`: Saves the Minimax player states to a file.

#### getReward Function

`def getReward(playerID: int, winner: int) -> float`: This function calculates the reward for a player based on the outcome of the game.

#### game.py
The game class has been modified to include a "draw match" result after 500 moves

## Process


In the first part of the development process, I created a reinforcement learning player. I trained this player with different exploration rates to strike the right balance between exploring new strategies and exploiting known ones. After experimentation, I found that an exploration rate of around 0.3 worked best. Additionally, I adjusted the learning rate for the Monte Carlo method, settling on a value of approximately 0.1. The training was conducted by putting the agent against another identical reinforcement learning agent.

The results showed a win rate of 78% when playing first and 72% when playing second against a random player.

In the second part of the development, I implemented a minimax agent with a specialized data structure to expedite the evaluation process. The evaluation of the states is based on a 2D convolution using a 3x3 kernel over the game board. When tested against a random player, this minimax agent achieved near-perfect scores regardless of whether it played first or second.

When putting one agent against the other, both being deterministic, the results varied between tests. However, for each individual test, one agent emerged as the victor with a consistent 100% success rate. The winning agent varied between tests, but for each test, the victorious agent remained unbeaten.

## References
[Alpha Beta Pruning miniMax](https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/)\
[Quixo is Solved](https://arxiv.org/abs/2007.15895)
