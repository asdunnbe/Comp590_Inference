import multiprocessing
from multiprocessing import Process
from typing import List, Any, Dict, Callable

import numpy as np
import tqdm

np.random.seed(225530)


def viterbi(observed_sequence: List[Any], possible_hidden_states: List[Any], transition_matrix: Callable,
            seq_item_to_emission_probabilities: Callable, hidden_to_emission_representation: Dict[Any, Any]):
    """Implements viterbi for a model that has transition and emission probabilities with a markov assumption.
    Finds the most likely sequence for a given observed sequence.
    """
    score_table: List[List[int]] = [[0 for _ in range(len(possible_hidden_states))] for _ in range(len(observed_sequence))]
    best_table: List[List[int]] = [[0 for _ in range(len(possible_hidden_states))] for _ in range(len(observed_sequence))]
    for idx, hidden in enumerate(possible_hidden_states):
        score_table[0][idx] = seq_item_to_emission_probabilities(observed_sequence[0], hidden_to_emission_representation[hidden], len(observed_sequence))

    for word_idx, observed in enumerate(observed_sequence[1:]):
        for tag_idx, hidden in enumerate(possible_hidden_states):
            best_score = 0
            for old_tag in range(len(possible_hidden_states)):
                local_score = score_table[word_idx][old_tag] * transition_matrix(old_tag, tag_idx, len(observed_sequence)) * \
                              seq_item_to_emission_probabilities(observed, hidden_to_emission_representation[hidden], len(observed_sequence))
                if local_score == 0:
                    local_score += 1e-15
                assert local_score > 0, "Numerical instability in viterbi detected!"
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
                   transition_matrix: Callable,
                   seq_item_to_emission_probabilities: Callable,
                   hidden_to_idx: Dict[Any, int], hidden_to_emission_representation: Dict[Any, Any], no_iterations: int = 5):
    initial_guess = [[i, possible_hidden_states[0]] for i in observed_sequence]

    # print(f"CHECK ME : {possible_hidden_states}")
    def get_denominator(current_observed, prev_hidden=None, next_hidden=None):
        res = 0
        for hidden_idx, hidden in enumerate(possible_hidden_states):
            if next_hidden:
                p_next_y = transition_matrix(hidden_idx, next_hidden, len(observed_sequence))
            else:
                p_next_y = 1
            if prev_hidden:
                p_current_y = transition_matrix(prev_hidden, hidden_idx, len(observed_sequence))
            else:
                p_current_y = 1
            res += p_next_y \
                   * p_current_y \
                   * seq_item_to_emission_probabilities(current_observed, hidden_to_emission_representation[hidden], len(observed_sequence))
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
                    p_next_y = transition_matrix(hidden_idx, next_hidden, len(observed_sequence))
                else:
                    p_next_y = 1
                if prev_hidden:
                    p_current_y = transition_matrix(prev_hidden, hidden_idx, len(observed_sequence))
                else:
                    p_current_y = 1
                numer = p_next_y \
                        * p_current_y \
                        * seq_item_to_emission_probabilities(x_and_y[0], hidden_to_emission_representation[hidden], len(observed_sequence))
                distribution.append(numer / local_const_denominator)
            initial_guess[idx][1] = possible_hidden_states[
                np.random.choice(range(len(possible_hidden_states)), p=distribution)]
    return [i[1] for i in initial_guess]


def variational_inference(observed_sequence, possible_hidden_states, transition_matrix: Callable,
                          seq_item_to_emission_probabilities: Callable, hidden_to_emission_representation: Dict[Any, Any],
                          parallelize: bool = True):
    """ len(observed_sequence) = N
    """
    q = []
    for i in observed_sequence:
        local_list = np.random.random(len(possible_hidden_states))
        local_list = local_list / sum(local_list)
        q.append(local_list)

    # def sub_calculate(q_i, q_idx, y_i, y_idx, update_dict):
    #     running_sum = 0
    #     # Computes P(X_i, Y_i-1, y_i, y_i+1)
    #     # Marginalize y_i-1 and y_i+1
    #     for y_i_minus_1_idx, y_i_minus_1 in enumerate(possible_hidden_states):
    #         for y_i_plus_1_idx, y_i_plus_1 in enumerate(possible_hidden_states):
    #             prob_before = 1 if q_idx == 0 else q[q_idx - 1][y_i_minus_1_idx]
    #             prob_after = 1 if q_idx == len(q) - 1 else q[q_idx + 1][y_i_plus_1_idx]
    #
    #             transition_before = 1 if q_idx == 0 else transition_matrix(y_i_minus_1_idx, y_idx, len(observed_sequence))
    #             transition_after = 1 if q_idx == len(q) - 1 else transition_matrix(y_idx, y_i_plus_1_idx, len(observed_sequence))
    #             running_sum += np.log(
    #                 seq_item_to_emission_probabilities(observed_sequence[q_idx], y_i, len(observed)) \
    #                 * transition_before \
    #                 * transition_after
    #             ) * prob_before * prob_after
    #     update_dict.put((y_idx, np.exp(running_sum) + 1E-15))  # in case np.exp(x) is too small

    def calculate(q_i, q_idx, update_dict):
        q_i_update = np.array([0.0 for _ in range(len(possible_hidden_states))])
        for y_idx, y_i in enumerate(possible_hidden_states):
            running_sum = 0
            for y_i_minus_1_idx, y_i_minus_1 in enumerate(possible_hidden_states):
                for y_i_plus_1_idx, y_i_plus_1 in enumerate(possible_hidden_states):
                    prob_before = 1 if q_idx == 0 else q[q_idx - 1][y_i_minus_1_idx]
                    prob_after = 1 if q_idx == len(q) - 1 else q[q_idx + 1][y_i_plus_1_idx]

                    transition_before = 1 if q_idx == 0 else transition_matrix(y_i_minus_1_idx, y_idx, len(observed_sequence))
                    transition_after = 1 if q_idx == len(q) - 1 else transition_matrix(y_idx, y_i_plus_1_idx, len(observed_sequence))
                    running_sum += np.log(
                        seq_item_to_emission_probabilities(observed_sequence[q_idx], hidden_to_emission_representation[y_i], len(observed_sequence)) \
                        * transition_before \
                        * transition_after
                    ) * prob_before * prob_after
            q_i_update[y_idx] = np.exp(running_sum) + 1E-15
        q_i_update = q_i_update / sum(q_i_update)
        if isinstance(update_dict, multiprocessing.Queue):
            update_dict.put((q_idx, q_i_update))
        else:
            update_dict.append((q_idx, q_i_update))

    for i in tqdm.tqdm(range(15)):
        update = []
        # Only have n threads at once
        if parallelize:
            qi_update = multiprocessing.Queue(maxsize=len(observed_sequence))
            max_no_threads = 16
            for i in range(0, len(q), max_no_threads):
                threads = []
                no_queued = 0
                for q_idx, q_i in enumerate(q[i:i+max_no_threads]):
                    no_queued += 1
                    # q_i should be 1 x K
                    t = Process(target=calculate, args=(q_i,q_idx+i, qi_update))
                    threads.append(t)
                    t.start()
                [thread.join() for thread in threads]

                # Stops the Queue from filling up
                update.extend([qi_update.get() for i in range(no_queued)])
        else:
            for q_idx, q_i in enumerate(q):
                calculate(q_i, q_idx+1, update)

        all_eq = 0
        for key, value in update:
            prev_values = q[key]
            equal = 0
            for prev, val in zip(prev_values, value):
                if abs(prev - val) <= 1E-9:
                    equal += 1
            if equal:
                all_eq += 1
            q[key] = value
        if all_eq == len(q): break

        # TODO: Figure out how this works with minimizing ELBO

    return [possible_hidden_states[np.argmax(dist)] for dist in q]
