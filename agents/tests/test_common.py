import numpy as np
from agents.Common import BoardPiece, NO_PLAYER, PlayerAction, PLAYER1, PLAYER2
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


def test_apply_player_action():
    from agents.Common import apply_player_action

    test_board = initialize_test_board()
    player = PLAYER1
    action = PlayerAction(6)
    ret = apply_player_action(test_board, action, player)
    test_board[0, action] = PLAYER1
    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == test_board)


def test_connected_four():
    from agents.Common import connected_four

    test_board = initialize_test_board()
    player = PLAYER1
    ret_not = connected_four(test_board, player)
    test_board[0, 5] = PLAYER1
    test_board[0, 0] = PLAYER2
    test_board[0, 6] = PLAYER1
    ret_win = connected_four(test_board, player)
    assert not ret_not
    assert ret_win


def test_check_end_state():
    from agents.Common import check_end_state, GameState

    test_board = initialize_test_board()
    player = PLAYER1
    ret_playing = check_end_state(test_board, player)
    test_board[0, 5] = PLAYER1
    test_board[0, 0] = PLAYER2
    test_board[0, 6] = PLAYER1
    ret_win = check_end_state(test_board, player)

    assert isinstance(ret_playing, GameState)
    assert ret_playing == GameState.STILL_PLAYING
    assert isinstance(ret_win, GameState)
    assert ret_win == GameState.IS_WIN
