import numpy as np
from typing import Tuple, Optional
from agents.Common import BoardPiece, PlayerAction, check_end_state, GameState, apply_player_action, SavedState
from agents.Common import connected_four, connected_some
from more_itertools import distinct_permutations


DEPTH = BoardPiece(3)
GOOD_SEQUENCE = np.array([1, 1, 1, 0])


def board_children(board: np.ndarray, player: BoardPiece) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param player: current player
    :param board: parent board
    :return: board children
    """
    free_columns = np.array(np.unique(np.where(board == 0)[1]), dtype=PlayerAction)
    children = np.zeros((len(free_columns), 6, 7))
    temp_board = np.copy(board)
    for i in range(len(free_columns)):
        children[i] = apply_player_action(board, action=free_columns[i], player=player)
        board = np.copy(temp_board)
    return free_columns, children


def change_player(player: BoardPiece) -> BoardPiece:
    """
    :param player: current player
    :return: the other possible player
    """
    if player == BoardPiece(1):
        return BoardPiece(2)
    if player == BoardPiece(2):
        return BoardPiece(1)


def maximize(board: np.ndarray, agent: BoardPiece, opponent: BoardPiece, current_depth: BoardPiece):
    """
    :param board: gets the board as input
    :param player: player that has its next move maximized
    :param current_depth: how deep we are in the search tree
    :return: the move with maximum utility, the maximum utility
    """

    check_status = check_end_state(board, agent)
    if check_status != GameState.STILL_PLAYING or current_depth == DEPTH:
        return None, calculate_utility(board, agent, opponent)

    max_utility = -np.inf
    move_max_utility = None
    move_possibilities, children = board_children(board, agent)
    for child, move in enumerate(move_possibilities):
        _, utility = minimize(children[child], agent, opponent, current_depth + BoardPiece(1))

        if utility > max_utility:
            move_max_utility = move
            max_utility = utility

    if move_max_utility == None:
        move_max_utility = np.min(move_possibilities)
    return move_max_utility, max_utility


def minimize(board: np.ndarray, agent: BoardPiece, opponent: BoardPiece, current_depth: BoardPiece):
    """
    :param opponent:
    :param agent:
    :param board: gets the board as input
    :param player: player that has its next move minimized
    :param current_depth: how deep we are in the search tree
    :return: the move with maximum utility, the maximum utility
    """
    check_status = check_end_state(board, opponent)
    if check_status != GameState.STILL_PLAYING or current_depth == DEPTH:
        return None, calculate_utility(board, agent, opponent)

    min_utility = np.inf
    move_min_utility = None

    move_possibilities, children = board_children(board, opponent)
    for child, move in enumerate(move_possibilities):
        _, utility = maximize(children[child], agent, opponent, current_depth + BoardPiece(1))

        if utility < min_utility:
            move_min_utility = move
            min_utility = utility

    if move_min_utility == None:
        move_min_utility = np.min(move_possibilities)

    return move_min_utility, min_utility


def calculate_utility(board: np.ndarray, agent: BoardPiece, opponent: BoardPiece):
    """
    :param board: board for which utility is being calculated
    :param agent: player for which utility is being calculated
    :param opponent: opponent player
    :return: utility of the board given, considering max utility as winning, minimum utility as loosing
    and intermediate values for sequences with 3 pieces in a row
    """

    if connected_four(board, opponent):
        return -np.inf

    if connected_four(board, agent):
        return np.inf

    utility = 0
    good_sequences = np.asarray(list(distinct_permutations(GOOD_SEQUENCE)))
    for sequence in good_sequences:
        if connected_some(board, opponent, sequence):
            return -18
        if connected_some(board, agent, sequence):
            utility += 4
    else:
        utility = np.random.randint(0, 6)

    return utility


def generate_move_minimax(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState] = None
) -> Tuple[PlayerAction, Optional[SavedState]]:
    global agent
    global opponent
    agent = player
    opponent = change_player(agent)
    action, _ = maximize(board, agent, opponent, current_depth=BoardPiece(0))

    return action, saved_state