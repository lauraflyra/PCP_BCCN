import numpy as np
from agents.Common import BoardPiece, PLAYER1, PLAYER2, initialize_game_state


def initialize_test_board() -> np.ndarray:
    test_board = initialize_game_state()
    test_board[0, 1] = PLAYER2
    test_board[1, 1] = PLAYER2
    test_board[0, 2] = PLAYER2
    test_board[2, 2] = PLAYER2
    test_board[1, 3] = PLAYER2
    test_board[1, 4] = PLAYER2
    test_board[1, 2] = PLAYER1
    test_board[3, 2] = PLAYER1
    test_board[0, 3] = PLAYER1
    test_board[2, 3] = PLAYER1
    test_board[3, 3] = PLAYER1
    test_board[0, 4] = PLAYER1
    test_board[2, 4] = PLAYER1

    return test_board


def pretty_print_test_board():
    printed_test_board = """|=============|
|             |
|             |
|    X X      |
|    O X X    |
|  O X O O    |
|  O O X X    |
|=============|
|0 1 2 3 4 5 6|"""
    return printed_test_board
