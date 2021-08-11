from numpy import ndarray
from typing import Tuple
import numpy as np


def calculate_scores(players_tiers_matrix: ndarray, tiers_nb: int = 9) -> ndarray:
    mshape = players_tiers_matrix.shape
    players_nb = mshape[1]
    scores_without_order_multiplier = np.maximum((tiers_nb + 1) * np.ones(mshape) - players_tiers_matrix, np.zeros(mshape))
    order_multiplier = np.linspace([1] * players_nb, [0] * players_nb, players_nb)
    scores = np.sum(scores_without_order_multiplier * order_multiplier, axis=0)
    return scores


def calculate_cost(players_tiers_matrix: ndarray, tiers_nb: int = 9) -> float:
    scores = calculate_scores(players_tiers_matrix, tiers_nb)
    return np.std(scores)


def _build_initial_solution_from_partial_solution(partial_solution: ndarray, players_nb: int, current_i: int, current_j: int) -> ndarray:
    all_values = list(range(1, players_nb+1))
    forbidden_values = []
    solution = None
    while solution is None:
        possible_values = [value for value in all_values if (value not in partial_solution[current_i, :] and value not in partial_solution[:, current_j] and value not in forbidden_values)]
        if len(possible_values) == 0:
            return None
        solution = np.copy(partial_solution)
        solution[current_i, current_j] = np.random.choice(possible_values)
        if current_i == current_j == players_nb - 1:
            return solution
        else:
            if current_j == players_nb - 1:
                new_i = current_i + 1
                new_j = 0
            else:
                new_i = current_i
                new_j = current_j + 1
            candidate_solution = _build_initial_solution_from_partial_solution(solution, players_nb, new_i, new_j)
            if candidate_solution is not None:
                return candidate_solution
            else:
                forbidden_values.append(solution[current_i, current_j])
                solution = None
    return None

def build_initial_solution(players_nb: int) -> ndarray:
    return _build_initial_solution_from_partial_solution(np.zeros((players_nb, players_nb)), players_nb, 0, 0)


def get_player_swapped_solution(solution_matrix: ndarray) -> ndarray:
    matrix = np.copy(solution_matrix)
    
    all_players = list(range(matrix.shape[1]))
    random_player_1 = np.random.choice(all_players)
    random_player_2 = np.random.choice([player for player in all_players if player != random_player_1])

    all_turns = all_players
    random_turn_1 = np.random.choice(all_turns)
    random_turn_2 = np.random.choice([turn for turn in all_turns if turn != random_turn_1])

    turns_to_swap = [random_turn_1, random_turn_2]
    tiers_for_player_1 = set(matrix[turns_to_swap, random_player_1])
    tiers_for_player_2 = set(matrix[turns_to_swap, random_player_2])

    while tiers_for_player_1 != tiers_for_player_2:
        all_tiers_to_swap = tiers_for_player_1.union(tiers_for_player_2)
        turns_to_swap = [turn for turn, tier in enumerate(matrix[:, random_player_1]) if tier in all_tiers_to_swap]
        tiers_for_player_1 = set(matrix[turns_to_swap, random_player_1])
        tiers_for_player_2 = set(matrix[turns_to_swap, random_player_2])

    # Do the swap
    matrix[turns_to_swap, random_player_1] = solution_matrix[turns_to_swap, random_player_2]
    matrix[turns_to_swap, random_player_2] = solution_matrix[turns_to_swap, random_player_1]

    return matrix


def get_turn_swapped_solution(solution_matrix: ndarray) -> ndarray:
    matrix = solution_matrix.T
    matrix = get_player_swapped_solution(matrix)
    return matrix.T


def get_modified_solution(solution_matrix: ndarray) -> ndarray:
    if np.random.random() > 0.5:
        return get_player_swapped_solution(solution_matrix)
    else:
        return get_turn_swapped_solution(solution_matrix)


def do_the_hastings_metropolis(runs_nb: int, steps_nb: int, players_nb: int, tiers_nb: int, acceptance_power: float = 1.0) -> Tuple[ndarray, float]:
    best_cost = None
    best_solution = None

    for run_index in range(runs_nb):
        print(f"Starting run #{run_index} ...")
        current_solution = build_initial_solution(players_nb)
        current_cost = calculate_cost(current_solution, tiers_nb)

        if best_solution is None:
            best_solution = current_solution
            best_cost = current_cost

        for _ in range(steps_nb):
            new_solution = get_modified_solution(current_solution)
            new_cost = calculate_cost(new_solution, tiers_nb)

            acceptance_probability = min((current_cost / new_cost) ** acceptance_power, 1)

            if np.random.random() <= acceptance_probability:
                current_solution = new_solution
                current_cost = new_cost                                       

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution
                print(f"--> Found better solution with cost {best_cost}")

    return best_solution, best_cost


