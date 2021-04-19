import numpy as np
from agents.Common import BoardPiece, NO_PLAYER, PlayerAction
from agents.tests.test_helpers import *


def test_initialize_game_state():
    from agents.Common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    from agents.Common import pretty_print_board

    test_board = initialize_test_board()
    ret = pretty_print_board(test_board)

    assert isinstance(ret, str)
    assert len(ret) == 16 * 9 - 1  # 16 characters per line, 9 lines
    assert ret == pretty_print_test_board()


def test_string_to_board():
    from agents.Common import string_to_board

    test_board = initialize_test_board()
    pp_test_board = pretty_print_test_board()
    ret = string_to_board(pp_test_board)
    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == test_board)


'''
def test_apply_player_action():
    from agents.Common import apply_player_action

    ret = apply_player_action(board, action, player, copy)
    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)


def test_connected_four():
    from agents.Common import connected_four

    ret = connected_four(board, player, last_action)
    assert isinstance(ret, bool)


def test_check_end_state():
    from agents.Common import check_end_state, GameState

    ret = check_end_state(board, player, last_action)
    assert isinstance(ret, GameState)
'''
