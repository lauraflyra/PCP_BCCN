import timeit
from agents.Common import connected_four, initialize_game_state, BoardPiece, search_sequence_numpy

board = initialize_game_state()

number = 10**4

res = timeit.timeit("connected_four(board, player)",
                    number=number,
                    globals=dict(connected_four=connected_four,
                                 board=board,
                                 player=BoardPiece(1)))

print(f"Python iteration-based: {res/number*1e6 : .1f} us per call")

"""
DISCLAIMER: I tried to improve performance using numba, but my connect_four function
calls the function search sequence numpy that uses numpy.all(axis = 1) and numba only accepts it
without optional arguments, which is not the case. Given that it took me a while to 
notice that, I couldn't think of a way to go around this problem.
"""