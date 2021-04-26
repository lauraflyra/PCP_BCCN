import numpy as np
from agents.Common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Tuple


def generate_move_random(
        board: np.ndarray, player: BoardPiece = None, saved_state: Optional[SavedState] = None
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    # get all free columns from the board
    free_columns = np.array(np.unique(np.where(board == 0)[1]), dtype=PlayerAction)
    action = np.random.choice(free_columns)

    return action, saved_state
