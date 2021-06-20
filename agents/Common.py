from enum import Enum
from typing import Optional
import numpy as np
from numpy import ndarray
from typing import Callable, Tuple


BoardPiece = np.int8  # The data type (dtype) of the board

NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece

NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

translation = str.maketrans("012[]", " XO||")


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    :return board: an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    board: ndarray = np.zeros((6, 7), dtype=BoardPiece)
    return board


def pretty_print_board(board: np.ndarray) -> str:
    """
    :param board: current state of the board in a ndarray, shape (6, 7) and data type (dtype) BoardPiece
    :return human_board: `board` converted to a human readable string representation,
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
    :param pp_board: output of pretty_print_board
    :return new_board: takes pp_board and turns it back into an ndarray.
    """
    pp_board = pp_board.replace(" ", "0").replace("X", "1").replace("O", "2").replace("|", "")
    partial_board = np.array(pp_board.splitlines()[1:-2])

    new_board = np.zeros((6, 13))

    for i in range(len(partial_board)):
        new_board[i] = np.array(list(partial_board[i]), dtype=BoardPiece)
    index = [i for i in range(1, 13, 2)]

    new_board = np.delete(new_board, index, 1).astype(BoardPiece)
    return np.flipud(new_board)


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    :param board: current state of the board
    :param action: which column does the player wants to play
    :param player: who's playing the current round
    :param copy: if should make a copy of the board before modifying it.
    :return board after applying the player action
    """
    if copy:
        board = np.copy(board)
    row = np.where(board.T[action] == NO_PLAYER)[0][0]
    board[row, action] = player
    return board


def search_sequence_numpy(arr, seq) -> np.ndarray:
    """ Find sequence in an array using NumPy only.
    taken from https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array
    ------
    :return : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    na, nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    m = (arr[np.arange(na - nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if m.any() > 0:
        return np.where(np.convolve(m, np.ones((nseq), dtype=int)) > 0)[0]
    else:
        return np.array([])  # No match found


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    :param board: board that is going to be evaluated
    :param player: player for which we look for a sequence of 4 pieces
    :param last_action: If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    :return: True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.

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

    for offset in range(-3, 3):
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
    :param board: board that is going to be evaluated
    :param player: who's playing the current round
    :param last_action: If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    :return: Returns the current game state for the current `player`, i.e. has their last
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


def connected_some(
        board: np.ndarray, player: BoardPiece, sequence, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    :param board: board that is going to be evaluated
    :param player: player for which we look for some sequence of 4 pieces, not necessarily 4 of the same, can be 3 of the same and one without a piece
    :param last_action: If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    :param sequence: sequence that we want to find in the board
    :return: True if there are four pieces equal to the sequence provided arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.

    """
    board_transposed = board.T
    board_flipped = np.flipud(board)
    rows, columns = board.shape
    sequence = player*sequence
    shuffled_rows = np.arange(rows)
    np.random.shuffle(shuffled_rows)
    shuffled_columns = np.arange(columns)
    np.random.shuffle(shuffled_columns)

    for row in shuffled_rows:
        found = np.array(search_sequence_numpy(board[row], sequence))
        if found.shape[0] != 0:
            return True

    for column in shuffled_columns:
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
