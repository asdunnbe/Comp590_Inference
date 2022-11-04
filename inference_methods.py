from typing import List, Any, Dict

import numpy as np

np.random.seed(225530)


def viterbi(observed_sequence: List[Any], possible_hidden_states: List[Any], transition_matrix: List[List[float]],
            seq_item_to_emission_probabilities: Dict[Any, Dict[Any, float]]):
    """Implements viterbi for a model that has transition and emission probabilities with a markov assumption.
    Finds the most likely sequence for a given observed sequence. Observed sequence should be an embedded version
    """
    score_table: List[List[int]] = [[0 for _ in range(len(transition_matrix))] for _ in range(len(observed_sequence))]
    best_table: List[List[int]] = [[0 for _ in range(len(transition_matrix))] for _ in range(len(observed_sequence))]
    for idx, hidden in enumerate(possible_hidden_states):
        score_table[0][idx] = seq_item_to_emission_probabilities[observed_sequence[0]][hidden]

    for word_idx, observed in enumerate(observed_sequence[1:]):
        for tag_idx, hidden in enumerate(possible_hidden_states):
            best_score = 0
            for old_tag in range(len(possible_hidden_states)):
                local_score = score_table[word_idx][old_tag] * transition_matrix[old_tag][tag_idx] * \
                              seq_item_to_emission_probabilities[observed][hidden]
                if best_score < local_score:
                    score_table[word_idx + 1][tag_idx] = local_score
                    best_table[word_idx + 1][tag_idx] = old_tag
                    best_score = local_score

    current_best = np.argmax(best_table[len(observed_sequence) - 1])
    most_likely_hidden = [possible_hidden_states[current_best]]
    for i in range(len(observed_sequence) - 1, 0, -1):
        current_best = best_table[i][current_best]
        most_likely_hidden.append(possible_hidden_states[current_best])
    most_likely_hidden.reverse()
    return most_likely_hidden


def gibbs_sampling(observed_sequence: List[Any], possible_hidden_states: List[Any],
                   transition_matrix: List[List[float]],
                   seq_item_to_emission_probabilities: Dict[Any, Dict[Any, float]],
                   hidden_to_idx: Dict[Any, int], no_iterations: int = 5):
    initial_guess = [[i, possible_hidden_states[0]] for i in observed_sequence]

    # print(f"CHECK ME : {possible_hidden_states}")
    def get_denominator(current_observed, prev_hidden=None, next_hidden=None):
        res = 0
        for hidden_idx, hidden in enumerate(possible_hidden_states):
            if next_hidden:
                p_next_y = transition_matrix[hidden_idx][next_hidden]
            else:
                p_next_y = 1
            if prev_hidden:
                p_current_y = transition_matrix[prev_hidden][hidden_idx]
            else:
                p_current_y = 1
            res += p_next_y \
                   * p_current_y \
                   * seq_item_to_emission_probabilities[current_observed][hidden]
        return res

    for i in range(no_iterations):
        for idx, x_and_y in enumerate(initial_guess):
            if idx != 0:
                prev_hidden = hidden_to_idx[initial_guess[idx - 1][1]]
            else:
                prev_hidden = None
            if idx != len(initial_guess) - 1:
                next_hidden = hidden_to_idx[initial_guess[idx + 1][1]]
            else:
                next_hidden = None

            local_const_denominator = get_denominator(x_and_y[0], prev_hidden, next_hidden)
            distribution = []

            for hidden_idx, hidden in enumerate(possible_hidden_states):
                if next_hidden:
                    p_next_y = transition_matrix[hidden_idx][next_hidden]
                else:
                    p_next_y = 1
                if prev_hidden:
                    p_current_y = transition_matrix[prev_hidden][hidden_idx]
                else:
                    p_current_y = 1
                numer = p_next_y \
                        * p_current_y \
                        * seq_item_to_emission_probabilities[x_and_y[0]][hidden]
                distribution.append(numer / local_const_denominator)
            initial_guess[idx][1] = possible_hidden_states[
                np.random.choice(range(len(possible_hidden_states)), p=distribution)]
    return [i[1] for i in initial_guess]
