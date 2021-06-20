import cProfile
from agents.agent_minimax_prunning import generate_move_minimax_pruning
from main import human_vs_agent

cProfile.run(
"human_vs_agent(generate_move_minimax_pruning, generate_move_minimax_pruning)", "mmab"
)

import pstats
p = pstats.Stats("mmab")
p.sort_stats("tottime").print_stats(50)