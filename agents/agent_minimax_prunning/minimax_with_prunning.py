import numpy as np
from typing import Tuple, Optional, Union, Any
from agents.Common import BoardPiece, PlayerAction, check_end_state, GameState, apply_player_action, SavedState
from agents.Common import connected_four, connected_some
from more_itertools import distinct_permutations

DEPTH = BoardPiece(5)
GOOD_SEQUENCE = np.array([1, 1, 1, 0])


def board_children(board: np.ndarray, player: BoardPiece) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find all board children by applying all available actions into a temporary board,
    which is a copy of the one given as parent
    :param player: current player
    :param board: parent board
    :return: available columns and all board children
    """
    free_columns = np.array(np.unique(np.where(board == 0)[1]), dtype=PlayerAction)
    children = np.zeros((len(free_columns), 6, 7), dtype=BoardPiece)
    for i in range(len(free_columns)):
        children[i] = apply_player_action(board, action=free_columns[i], player=player, copy = True)
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


def maximize(board: np.ndarray, agent: BoardPiece, opponent: BoardPiece, current_depth: BoardPiece,
             alpha: float = -np.inf, beta: float = np.inf) -> \
        Tuple[None, Union[Union[float, int], Any]]:
    """
    :param board: gets the board as input
    :param agent: player that has its next move maximized
    :param opponent: player that has its next move minimized
    :param current_depth: how deep we are in the search tree
    :param alpha: value of alpha for the pruning, changes over calls of maximize,
    is the value of maximum utility found in the children
    :param beta: value of beta for the pruning, changes over calls of minimize,
    is the value of minimum utility found in the children
    :return: the move with maximum utility and the maximum utility,i.e,
    move that the agent is going to play and the utility associated
    """

    check_status = check_end_state(board, agent)
    if check_status != GameState.STILL_PLAYING or current_depth == DEPTH:
        return None, calculate_utility(board, agent, opponent)

    max_utility = alpha
    move_max_utility = None
    move_possibilities, children = board_children(board, agent)
    for child, move in enumerate(move_possibilities):
        _, utility = minimize(children[child], agent, opponent, current_depth + BoardPiece(1), max_utility, beta)

        if utility > max_utility:
            move_max_utility = move
            max_utility = utility
        if max_utility >= beta:
            break

    if move_max_utility is None:
        move_max_utility = np.min(move_possibilities)
    return move_max_utility, max_utility


def minimize(board: np.ndarray, agent: BoardPiece, opponent: BoardPiece, current_depth: BoardPiece,
             alpha: float = -np.inf, beta: float = np.inf) -> \
        Tuple[None, Union[Union[float, int], Any]]:
    """
    :param opponent: player that has its next move minimized
    :param agent: player that has its next move maximized
    :param board: gets the board as input
    :param current_depth: how deep we are in the search tree
    :param alpha: value of alpha for the pruning, changes over calls of maximize,
    is the value of maximum utility found in the children
    :param beta: value of beta for the pruning, changes over calls of minimize, i
    s the value of minimum utility found in the children
    :return: the move with minimum utility and the minimum utility, i.e,
    move that the opponent is going to play and the utility associated with it
    """
    check_status = check_end_state(board, opponent)
    if check_status != GameState.STILL_PLAYING or current_depth == DEPTH:
        return None, calculate_utility(board, agent, opponent)

    min_utility = beta
    move_min_utility = None

    move_possibilities, children = board_children(board, opponent)
    for child, move in enumerate(move_possibilities):
        _, utility = maximize(children[child], agent, opponent, current_depth + BoardPiece(1), alpha, min_utility)

        if utility <= min_utility:
            move_min_utility = move
            min_utility = utility

        if min_utility <= alpha:
            break

    if move_min_utility is None:
        move_min_utility = np.min(move_possibilities)

    return move_min_utility, min_utility


def calculate_utility(board: np.ndarray, agent: BoardPiece, opponent: BoardPiece) -> np.float_:
    """
    :param board: board for which utility is being calculated
    :param agent: player for whom utility is being maximized
    :param opponent: opponent player for whom utility is being minimized
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


def generate_move_minimax_pruning(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState] = None
) -> Tuple[None, Optional[SavedState]]:
    """
    :param board: current state of the board
    :param player: player who's playing the next round
    :param saved_state: 'The idea is that the first time in a game that generate_move is called,
    the value of that argument is None.
    Then in the process of choosing its first action,
    your agent might do a bunch of computation that it could reuse for future moves.
    Instead of just throwing that away, you can put it in an instance of your SavedState class'

    :return: move that the current player chose (with minimax and alpha beta pruning)
    and saved_state again, because it's not going to be used for now
    """
    global agent
    global opponent
    agent = player
    opponent = change_player(agent)
    action, _ = maximize(board, agent, opponent, current_depth=BoardPiece(0))

    return action, saved_state
