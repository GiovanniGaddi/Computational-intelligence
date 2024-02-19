import random
import pickle
import numpy
from copy import deepcopy
from collections import defaultdict, namedtuple
from tqdm import tqdm
from game import Game, Move, Player
from scipy import signal
import sys

MoveSlides = namedtuple('MoveSlides', ['coord', 'slides'])

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class MyPlayer(Player):
    def __init__(self) -> None:
        # Mask to get only the board's edge
        self._viable_moves_mask = numpy.zeros((5,5), dtype=bool)
        self._viable_moves_mask[0, :] = True
        self._viable_moves_mask[-1, :] = True
        self._viable_moves_mask[:, 0] = True
        self._viable_moves_mask[:, -1] = True
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # Random move based on possible actions
        possibleMoves = self._getPossibleMoves(game.get_current_player(),game.get_board())
        randomMove = random.choice(possibleMoves)
        from_pos = randomMove.coord
        slide = random.choice(randomMove.slides)
        # print('Random Player:',from_pos, slide)
        return from_pos, slide
    
    def _getPossibleMoves(self, player: int, board)-> list:

        viableMoves = ((board == -1) | (board == player)) & self._viable_moves_mask
        movesCoords = numpy.argwhere(viableMoves).tolist()
        #map viable coords to possible actions
        movesCoords = [MoveSlides((coords[1], coords[0]), self.__getPossibleSlides(coords)) for coords in movesCoords]
        return movesCoords
    
    def __getPossibleSlides(self, coords)->tuple[Move]:
        # Get possible slides given board coordinates
        if coords[0] < 0 | coords[0] > 4 | coords[1] < 0 | coords[1] > 4:
            return
        slides = []
        if coords[0] != 0:
            slides.append(Move.TOP)
        if coords[0] != 4:
            slides.append(Move.BOTTOM)
        if coords[1] != 0:
            slides.append(Move.LEFT)
        if coords[1] != 4:
            slides.append(Move.RIGHT)
        return tuple(slides)

    def _hashBoard(self, board) -> str:
        # Create 25 character Hash from a board
        return ''.join(map(str,numpy.where(board == -1, ' ', numpy.where(board == 0, 'X', 'O')).flatten()))
    
    def _getNextState(self, from_pos: tuple[int, int], slide: Move, playerID: int, initialBoard: numpy.ndarray) -> numpy.ndarray:
        # generate next state based on the initial board and action
        board = deepcopy(initialBoard)
        board[from_pos[1], from_pos[0]] = playerID

        if slide == Move.RIGHT:
            board[from_pos[1], from_pos[0]:] = numpy.roll(board[from_pos[1], from_pos[0]:], -1)
        elif slide == Move.LEFT:
            board[from_pos[1], :from_pos[0]+1] = numpy.roll(board[from_pos[1], :from_pos[0]+1], 1)
        elif slide == Move.BOTTOM:
            board[from_pos[1]:, from_pos[0]] = numpy.roll(board[from_pos[1]:, from_pos[0]], -1)
        elif slide == Move.TOP:
            board[:from_pos[1]+1, from_pos[0]] = numpy.roll(board[:from_pos[1]+1, from_pos[0]], 1)
        return board
    
    def __playerWinner(self,board: numpy.ndarray, playerID: int) -> bool:
        # Check if a player won
        vfboard = board == playerID
        return (numpy.all(vfboard, axis=1).any() or #any row
            numpy.all(vfboard, axis=0).any() or #any col
            numpy.all(numpy.diag(vfboard)) or #main diagonal
            numpy.all(numpy.diag(numpy.fliplr(vfboard)))) #second diagonal

    def _checkWinner(self, board: numpy.ndarray)-> int:
        #check if any player won
        if self.__playerWinner(board, 1):
            return 1
        if self.__playerWinner(board, 0):
            return 0
        return -1

class Tr_RL_Player(MyPlayer):
    def __init__(self, root: str, playerID: int = -1, expRate: float = 0.3, learningRate: float = 0.1) -> None:
        self._trajectory = list()
        # exploration rate
        self._expRate = expRate
        # learning rate
        self._lRate = learningRate

        self._root = root
        self._playerID = playerID
        #load states from memory
        self._loadStates()
        super().__init__()

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        #Save the assigned playerID
        if self._playerID < 0:
            self._playerID = game.get_current_player()
            self._loadStates()

        #append to trajectory actual state
        state = game.get_board()
        self._trajectory.append(super()._hashBoard(state))

        #get possible moves given the player
        possibleMoves = super()._getPossibleMoves(self._playerID, state)
        
        #initialize return values
        fromPos = possibleMoves[0].coord
        move = possibleMoves[0].slides[0]

        stateHash = ''

        if  self._expRate < random.random(): #Random choice
            moves = random.choice(possibleMoves)
            fromPos = moves.coord
            move = random.choice(moves.slides)
            #generate next state Hash to save in the trajectory
            nextState = super()._getNextState(fromPos, move, self._playerID, state)
            stateHash = super()._hashBoard(nextState)
        else:
            max_value = float('-inf')
            for moves in possibleMoves:
                for slide in moves.slides:
                    #get future state based on actual board and action
                    tempState = super()._getNextState(moves.coord, slide, self._playerID, state)
                    tempStateHash = super()._hashBoard(tempState)
                    #evaluate all possible moves
                    if max_value < self._states.get(tempStateHash, 0.0):
                        max_value = self._states.get(tempStateHash, 0.0)
                        fromPos = moves.coord
                        move = slide
                        stateHash = tempStateHash
        #add to trajectory the chosen next state
        self._trajectory.append(stateHash)
        return (fromPos, move)

    def evaluteStates(self, reward: float)-> None:
        #update or create values for the explored states
        for stateHash in reversed(self._trajectory):
            self._states[stateHash] = self._states.get(stateHash, 0.0) + self._lRate * (reward - self._states.get(stateHash, 0.0))
        self._trajectory.clear()

    def getFileName(self, player:int) -> str:
        return f'{self._root}/data/RL{'O' if player else 'X'}_{self._expRate:.2f}-{self._lRate:.2f}.rld'

    def _loadStates(self) -> None:
        if self._playerID >= 0:
            try:
                with open(self.getFileName(self._playerID), 'rb') as file:
                    self._states = pickle.load(file)
            except FileNotFoundError:
                self._states = defaultdict(float)

    def saveStates(self) -> None:
        if self._playerID >= 0:
            with open(self.getFileName(self._playerID), 'wb') as file:
                pickle.dump(self._states, file)

class RL_Player(Tr_RL_Player):
    def __init__(self, root: str, playerID: int = -1, expRate: float = 0.3, learningRate: float = 0.1) -> None:
        super().__init__(root, playerID, expRate, learningRate)
    
    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        #Save the assigned playerID
        if self._playerID < 0:
            self._playerID = game.get_current_player()
            super()._loadStates()

        #append actual state
        state = game.get_board()
        self._trajectory.append(super()._hashBoard(state))

        #get possible moves given the player
        possibleMoves = super()._getPossibleMoves(self._playerID, state)

        max_value = float('-inf')
        for moves in possibleMoves:
            for slide in moves.slides:
                #generate next states
                tempState = super()._getNextState(moves.coord, slide, self._playerID, state)
                tempStateHash = super()._hashBoard(tempState)
                #evaluate all possible moves
                if max_value < self._states.get(tempStateHash, 0.0):
                    max_value = self._states.get(tempStateHash, 0.0)
                    fromPos = moves.coord
                    move = slide
                    stateHash = tempStateHash
        #add to trajectory the chosen next state
        self._trajectory.append(stateHash)
        return (fromPos, move)
    
    def evaluteStates(self, reward: float) -> None:
        return super().evaluteStates(reward)
    
    def saveStates(self) -> None:
        return super().saveStates()


class Minimax_Player(MyPlayer):
    def __init__(self, root, playerID: int = -1, minimaxDepth: int = 4, biasReduction: float = 8) -> None:
        self._root = root
        self._playerID = playerID
        # minimax depth
        self._mmDepth = minimaxDepth
        # evaluation scaling
        self._biasRedu = biasReduction
        #kernel for 2D convolution
        self._evalKernel = numpy.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self._loadStates()
        super().__init__()

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        #Save the assigned playerID
        if(self._playerID < 0):
            self._playerID = game.get_current_player()
            self._loadStates()
        #get actual State and pass it to minimax 
        state = game.get_board()
        _, (from_pos, move) = self._minimax_alpha_beta(game, state, self._mmDepth, float('-inf'), float('inf'), True)

        return (from_pos, move)

    def _minimax_alpha_beta(self, game: Game, board: numpy.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool)-> tuple[float, tuple[tuple[int,int], Move]]:
        #try reading terminal state
        try:
            terminal = self._states[super()._hashBoard(board)]
            if terminal == float('inf') or terminal == float('-inf'):
                return terminal, tuple()
        except KeyError:
            pass
        #check if the state is terminal and not already saved
        terminal = super()._checkWinner(board)
        if terminal >=0:
            #if winner max reward
            if terminal == self._playerID:
                self._states[super()._hashBoard(board)] = float('inf')
                return float('inf'), tuple()
            #Opponent Winner min reward
            else: 
                self._states[super()._hashBoard(board)] = float('-inf')
                return float('-inf'), tuple()
        # guarantee not terminal state

        # if reached maximum depth evaluate
        if depth == 0:
            return self.__evaluate(board), tuple()
        possibleMoves = super()._getPossibleMoves((self._playerID + int(not maximizing_player))%2, board)
        
        #initializing best move
        best_move = (possibleMoves[0].coord, possibleMoves[0].slides[0])

        if maximizing_player:
            max_eval = float('-inf')
            for moves in possibleMoves:
                for slide in moves.slides:
                    nextBoard = super()._getNextState(moves.coord, slide, game.get_current_player(), board)
                    eval, _ = self._minimax_alpha_beta(game, nextBoard, depth - 1, alpha, beta, False)
                    if(eval > max_eval):
                        max_eval = eval
                        best_move = (moves.coord, slide)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for moves in possibleMoves:
                for slide in moves.slides:
                    nextBoard = super()._getNextState(moves.coord, slide, (game.get_current_player()+1)%2,board)
                    eval, _ = self._minimax_alpha_beta(game, nextBoard, depth - 1, alpha, beta, True)
                    if(eval < min_eval):
                        min_eval = eval
                        best_move = (moves.coord, slide)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval, best_move

    def __evaluate(self, board: numpy.ndarray) -> float:
        try:
            #check if value already present in memory
            value = self._states[super()._hashBoard(board)]
        except KeyError:
            #compure the 2D convolution and save the data
            value = numpy.sum(signal.convolve2d(board == self._playerID, self._evalKernel, mode= 'same', boundary= 'fill', fillvalue= False)*(board == self._playerID)/self._biasRedu)
            self._states[super()._hashBoard(board)] = value
        return value
    
    def getFileName(self, player:int) -> str:
        return f'{self._root}/data/mM{'O' if player else 'X'}.rld'

    def _loadStates(self) -> None:
        if self._playerID >= 0:
            try:
                with open(self.getFileName(self._playerID), 'rb') as file:
                    self._states = pickle.load(file)
            except FileNotFoundError:
                self._states = defaultdict(float)

    def saveStates(self) -> None:
        print("called save")
        if self._playerID >= 0:
            with open(self.getFileName(self._playerID), 'wb') as file:
                pickle.dump(self._states, file)

def getReward(playerID: int, winner: int) -> float:
    if playerID == 0:
        if winner == playerID:
            #win
            return 1.0
        elif winner == -1:
            #draw
            return 0.0
        #lost
        return -0.3
    if playerID == 1:
        if winner == playerID:
            #win
            return 1.2
        elif winner == -1:
            #draw
            return 0.5
        #lost
        return -0.1


if __name__ == '__main__':

    TRAINING = 100_000 
    TESTING = 1_000
    mode = "testing"
    if mode == "training":
        
        player1 = Tr_RL_Player('quixo')
        player2 = Tr_RL_Player('quixo')

        for i in tqdm(range(TRAINING)):
            g = Game()
            winner = g.play(player1, player2)
            player1.evaluteStates(getReward(0, winner))
            player2.evaluteStates(getReward(1, winner))
        player1.saveStates()
        player2.saveStates()
        print(f"Winner: Player {winner}")

    elif mode == "testing":
        player2 = MyPlayer()
        player1 = Minimax_Player('quixo', minimaxDepth= 2)
        win = [0,0]
        for i in tqdm(range(TESTING)):
            g = Game()
            winner = g.play(player1, player2)
            if winner >= 0:
                win[winner] += 1
            try:
                player1.value_states(getReward(0, winner))
            except AttributeError:
                pass
            try:
                player2.value_states(getReward(1, winner))
            except AttributeError:
                pass
        try:
            player1.saveStates()
        except AttributeError:
            pass
        try:
            player2.saveStates()
        except AttributeError:
            pass
        print(f'Player1 win rate: {win[0]/TESTING:.2%}')
        print(f'Player2 win rate: {win[1]/TESTING:.2%}')

