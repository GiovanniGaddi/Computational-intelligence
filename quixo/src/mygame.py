from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from collections import namedtuple
import numpy 

# Rules on PDF

Slide = namedtuple('Slide', ['roll', 'axis'])


class Move(Enum):
    '''
    Selects where you want to place the taken piece. The rest of the pieces are shifted
    '''
    TOP = Slide(-1, 0)
    BOTTOM = Slide(1, 0)
    LEFT = Slide(-1, 1)
    RIGHT = Slide(1, 1)


class Player(ABC):
    def __init__(self) -> None:
        '''You can change this for your player if you need to handle state/have memory'''
        pass

    @abstractmethod
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''
        The game accepts coordinates of the type (X, Y). X goes from left to right, while Y goes from top to bottom, as in 2D graphics.
        Thus, the coordinates that this method returns shall be in the (X, Y) format.

        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
        pass


class Game(object):
    def __init__(self) -> None:
        self._board = numpy.ones((5, 5), dtype=numpy.uint8) * -1
        self.current_player_idx = 1

    def get_board(self) -> numpy.ndarray:
        '''
        Returns the board
        '''
        return deepcopy(self._board)

    def get_current_player(self) -> int:
        '''
        Returns the current player
        '''
        return deepcopy(self.current_player_idx)

    def print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        print(self._board)

    def check_winner(self, board = numpy.array([])) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        tBoard = board if board.size else self._board
        # for each row
        for x in range(tBoard.shape[0]):
            # if a player has completed an entire row
            if tBoard[x, 0] != -1 and all(tBoard[x, :] == tBoard[x, 0]):
                # return the relative id
                return tBoard[x, 0]
        # for each column
        for y in range(tBoard.shape[1]):
            # if a player has completed an entire column
            if tBoard[0, y] != -1 and all(tBoard[:, y] == tBoard[0, y]):
                # return the relative id
                return tBoard[0, y]
        # if a player has completed the principal diagonal
        if tBoard[0, 0] != -1 and all(
            [tBoard[x, x]
                for x in range(tBoard.shape[0])] == tBoard[0, 0]
        ):
            # return the relative id
            return tBoard[0, 0]
        # if a player has completed the secondary diagonal
        if tBoard[0, -1] != -1 and all(
            [tBoard[x, -(x + 1)]
            for x in range(tBoard.shape[0])] == tBoard[0, -1]
        ):
            # return the relative id
            return tBoard[0, -1]
        return -1

    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        movesCounter = 500
        while winner < 0 and movesCounter >0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            #print(f'{movesCounter:03d}-Playing:{self.current_player_idx}')
            ok = False
            printCounter = 4
            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(
                    self)
                ok = self.__move(from_pos, slide, self.current_player_idx)
                # if not ok and printCounter > 0:
                #     self.print()
                #     print(self.get_current_player(),from_pos, slide)
                printCounter -=1
                # if printCounter<0:
                #     sys.exit()
            winner = self.check_winner()
            movesCounter -= 1
        return winner

    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take(from_pos, player_id)
        # if not acceptable:
        #     print("Move not acceptable")
        if acceptable:
            acceptable = self.__slide(from_pos, slide)
            if not acceptable:
                #print("Slide not acceptable")
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        #print(self._board[from_pos], from_pos)
        if acceptable:
            self._board[from_pos] = player_id
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is 
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable

class quixo(Game):
    def __init__(self) -> None:
        super().__init__()
    
    def print(self):
        print(self.__printBoard(self._board))
    
    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        
        if slide == Move.BOTTOM and from_pos[0] == 0:
            return False
        if slide == Move.TOP and from_pos[0] == 4:
            return False
        if slide == Move.RIGHT and from_pos[1] == 0:
            return False
        if slide == Move.LEFT and from_pos[1] == 4:
            return False
        if slide.value.axis:
            #axis = 1 = LEFT or RIGHT
            self._board[:, from_pos[1]] = numpy.roll(self._board[:, from_pos[1]], slide.value.roll)
        else:
            #axis =  0 = TOP or BOTTOM
            self._board[from_pos[0], :] = numpy.roll(self._board[from_pos[0], :], slide.value.roll)
        return True
    
    def __printBoard(self, arr):
        symbols = numpy.where(arr == -1, ' ', numpy.where(arr == 0, 'X', 'O'))
        return symbols

    def reset(self):
        self._board = numpy.ones((5, 5), dtype=numpy.uint8) * -1
        self.current_player_idx = 1


