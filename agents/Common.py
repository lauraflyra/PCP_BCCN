from enum import Enum
from typing import Optional
import numpy as np
from numpy import ndarray

BoardPiece = np.int8  # The data type (dtype) of the board

NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece

NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

translation = str.maketrans("012[]", " XO||")


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    board: ndarray = np.zeros((6, 7), dtype=BoardPiece)
    return board


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    human_board: str = "|=============|\n"
    flipped_board = np.flipud(board)
    string_board: str = np.array2string(flipped_board)[1:-1].replace("\n ", "\n")
    human_board += string_board.translate(translation)
    human_board += "\n|=============|\n"
    human_board += "|0 1 2 3 4 5 6|"
    return human_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    pp_board = pp_board.replace(" ", "0").replace("X", "1").replace("O", "2").replace("|", "")
    partial_board = np.array(pp_board.splitlines()[1:-2])

    new_board = np.zeros((6, 13))

    for i in range(len(partial_board)):
        new_board[i] = np.array(list(partial_board[i]), dtype=BoardPiece)
    index = [i for i in range(1, 13, 2)]

    new_board = np.delete(new_board, index, 1)
    return np.flipud(new_board)


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy:
        board = np.copy(board)
    row = np.where(board.T[PlayerAction] == 0)[0][0]
    board[row, PlayerAction] = player
    return board


def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.
    taken from https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []  # No match found


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    sequence = player * np.ones(4)
    board_transposed = board.T
    board_flipped = np.flipud(board)
    rows, columns = board.shape
    for row in range(rows):
        found = np.array(search_sequence_numpy(board[row], sequence))
        if found.shape[0] != 0:
            return True

    for column in range(columns):
        found = np.array(search_sequence_numpy(board_transposed[column], sequence))
        if found.shape[0] != 0:
            return True

    for offset in range(-2, 2):
        found = np.array(search_sequence_numpy(np.diag(board, k=offset), sequence))
        if found.shape[0] != 0:
            return True
        else:
            found = np.array(search_sequence_numpy(np.diag(board_flipped, k=offset), sequence))
            if found.shape[0] != 0:
                return True

    return False


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    check = connected_four(board, player)
    if check:
        return GameState.IS_WIN
    else:
        if len(np.where(board == 0)[0]) != 0:
            return GameState.STILL_PLAYING
        else:
            return GameState.IS_DRAW
