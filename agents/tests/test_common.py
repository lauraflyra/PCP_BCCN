import numpy as np
from ..agents.Common import BoardPiece, NO_PLAYER

from Tutorial_1.agents.Common import PlayerAction


def test_initialize_game_state():
    from ..agents.Common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board(board: np.ndarray):
    from ..agents.Common import pretty_print_board

    ret = pretty_print_board(board)

    assert isinstance(ret, str)
    assert len(ret) == 16 * 9  # 16 caracters per line, 9 lines


def test_string_to_board(pp_board: str):
    from ..agents.Common import string_to_board

    ret = string_to_board(pp_board)
    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)


def test_apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False):
    from ..agents.Common import apply_player_action

    ret = apply_player_action(board, action, player, copy)
    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)


def test_connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None):
    from ..agents.Common import connected_four

    ret = connected_four(board, player, last_action)
    assert isinstance(ret, bool)


def test_check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None):
    from ..agents.Common import check_end_state, GameState

    ret = check_end_state(board, player, last_action)
    assert isinstance(ret, GameState)
