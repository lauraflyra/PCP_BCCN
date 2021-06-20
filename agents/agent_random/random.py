import numpy as np
from agents.Common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Tuple


def generate_move_random(
        board: np.ndarray, player: BoardPiece = None, saved_state: Optional[SavedState] = None
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    :param board: current state of the board
    :param player: player who's playing the next round
    :param saved_state: 'The idea is that the first time in a game that generate_move is called,
    the value of that argument is None.
    Then in the process of choosing its first action,
    your agent might do a bunch of computation that it could reuse for future moves.
    Instead of just throwing that away, you can put it in an instance of your SavedState class'

    :return: move that the current player chose (randomly) and saved_state again,
    because it's not going to be used for now
    """

    # Choose a valid, non-full column randomly and return it as `action`
    # get all free columns from the board
    free_columns = np.array(np.unique(np.where(board == 0)[1]), dtype=PlayerAction)
    action = np.random.choice(free_columns)

    return action, saved_state
