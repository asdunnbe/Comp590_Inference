from typing import List, Any, Dict

import numpy as np


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
