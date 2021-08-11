from functions import *


# Game parameters

NUMBER_OF_PLAYERS = 12
NUMBER_OF_NON_NULL_TIERS = 9

# Hastings-Metropolis parameters

NUMBER_OF_RUNS = 10
NUMBER_OF_STEPS = 100_000
ACCEPTANCE_POWER = 100



if __name__ == "__main__":
    print("Running the algorithm...")
    matrix, cost = do_the_hastings_metropolis(NUMBER_OF_RUNS, NUMBER_OF_STEPS, NUMBER_OF_PLAYERS, NUMBER_OF_NON_NULL_TIERS, ACCEPTANCE_POWER)
    
    print("\n", "Solution (players are columns, turns are rows):")
    print(matrix)

    print("\n", f"Solution cost: {cost}")

    print("\n", "Players' draft scores:")
    print(calculate_scores(matrix, NUMBER_OF_NON_NULL_TIERS))
