from agents.tests.test_helpers import *
from agents.Common import BoardPiece


def test_generate_move_random():
    from agents.agent_random import random
    test_board = initialize_test_board()
    test_board[4, 2] = 1
    test_board[5, 2] = 2

    ret, _ = random.generate_move_random(test_board)
    assert isinstance(ret, BoardPiece)
    for i in range(50):
        ret, _ = random.generate_move_random(test_board)
        assert 2 != ret  # second column is full in this example board
