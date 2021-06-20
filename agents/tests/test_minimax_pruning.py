import numpy as np
from agents.Common import BoardPiece, NO_PLAYER, PlayerAction, PLAYER1, PLAYER2
from agents.tests.test_helpers import *
from typing import Tuple, Union, Any


def test_board_children():
    from agents.agent_minimax_prunning.minimax_with_prunning import board_children
    from agents.Common import apply_player_action
    test_board = initialize_test_board()
    players = [PLAYER1, PLAYER2]
    for p in players:
        ret = board_children(test_board, p)
        assert isinstance(ret, Tuple)
        assert np.all(ret[0] == np.arange(7))
        for i in range(7):
            assert ret[1][i].shape == (6, 7)
            assert np.all(ret[1][i] == apply_player_action(test_board, action=i, player=p, copy=True))


def test_change_player():
    from agents.agent_minimax_prunning.minimax_with_prunning import change_player
    players = [PLAYER1, PLAYER2]
    assert change_player(players[0]) == players[1]
    assert change_player(players[1]) == players[0]


def test_maximize_minimize():
    """
    if minimize and maximize work, calling maximize for PLAYER1 (for example) should return the
    same move than calling minimize for it's opponent, PLAYER2 (for example).
    """
    from agents.agent_minimax_prunning.minimax_with_prunning import maximize, minimize

    test_board = initialize_test_board()

    alpha = -np.inf
    beta = np.inf

    ret_maximize_1 = maximize(test_board, agent=PLAYER1, opponent=PLAYER2, current_depth=BoardPiece(0))
    ret_maximize_2 = maximize(test_board, agent=PLAYER2, opponent=PLAYER1, current_depth=BoardPiece(0))
    ret_minimize_1 = minimize(test_board, agent=PLAYER2, opponent=PLAYER1, current_depth=BoardPiece(0))
    ret_minimize_2 = minimize(test_board, agent=PLAYER1, opponent=PLAYER2, current_depth=BoardPiece(0))

    assert ret_maximize_1[1] > alpha
    assert ret_maximize_1[1] > alpha
    assert ret_minimize_1[1] < beta
    assert ret_minimize_1[1] < beta

    assert ret_maximize_1[0] == ret_minimize_1[0]
    assert ret_maximize_2[0] == ret_minimize_2[0]
