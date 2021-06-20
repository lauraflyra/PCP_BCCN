import numpy as np
from agents.Common import BoardPiece, NO_PLAYER, PlayerAction, PLAYER1, PLAYER2
from agents.tests.test_helpers import *
from typing import Tuple


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


def test_maximize():
    from agents.agent_minimax_prunning.minimax_with_prunning import maximize, generate_move_minimax_pruning, minimize
    test_board = initialize_test_board()
    # by looking just one step into the future, the agent should
    # print(maximize(test_board, agent=PLAYER1, opponent=PLAYER2, current_depth=BoardPiece(0)))
    # print(minimize(test_board, agent=PLAYER1, opponent=PLAYER2, current_depth=DEPTH-1))
    # print(generate_move_minimax_pruning(test_board, PLAYER1))
    ret = maximize(test_board, agent=PLAYER1, opponent=PLAYER2, current_depth=BoardPiece(0))
    gen_move = generate_move_minimax_pruning(test_board, PLAYER1)
    print(ret, gen_move)
    ret = maximize(test_board, agent=PLAYER2, opponent=PLAYER1, current_depth=BoardPiece(0))
    gen_move = generate_move_minimax_pruning(test_board, PLAYER2)
    print(ret, gen_move)
