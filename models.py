from collections import defaultdict
from typing import List, Set, Any, Tuple, Dict

import nltk
from sklearn.model_selection import train_test_split

# Needed since we want different defaults if it is in or not in the dictionary
from inference_methods import viterbi


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class HMM:
    def __init__(self, vocabulary=None):
        # Note we make this a dictionary as otherwise we would have to handle mapping a word to its index
        # in our vocabulary, that would be somewhat annoying and potentially bug-prone.
        self.observed_to_emission_probabilities: Dict[Any, Dict[Any, float]] = None
        self.transition_matrix: List[List[float]] = None
        self.vocabulary: Set[Any] = None
        self.possible_hiddens: List[Any] = None
        self._hidden_to_idx: Dict[Any, int] = defaultdict(int)
        self._hidden_occurences: Dict[Any, int] = defaultdict(int)

    def _build_emission_probabilities(self, observed_and_hidden: List[Tuple[Any, Any]]):
        """
        Part B:
        Calculates p(w | tag) by doing p(w and tag) / p ( tag)
        """
        for observed, hidden in observed_and_hidden:
            self.observed_to_emission_probabilities[observed][hidden] += 1
            self._hidden_occurences[hidden] += 1

        for possible_words in self.vocabulary:
            for hiddens in self.possible_hiddens:
                current_count = self.observed_to_emission_probabilities[possible_words][hiddens]
                self.observed_to_emission_probabilities[possible_words][hiddens] = current_count / self._hidden_occurences[
                    hiddens]

    def _build_transition_probabilities(self, observed_and_hidden: List[Tuple[Any, Any]]):
        hidden_combined_to_denominator_key: Dict[Tuple[int, 2], Any] = {}
        hidden_combined_to_count: Dict[Tuple[int, 2], int] = defaultdict(int)
        for idx, obv_and_hid in enumerate(observed_and_hidden):
            cur_hidden = obv_and_hid[1]
            try:
                next_hidden = observed_and_hidden[idx+1][1]
            except IndexError:
                break
            cur_idx = self._hidden_to_idx[cur_hidden]
            next_idx = self._hidden_to_idx[next_hidden]
            hidden_combined_to_count[(cur_idx, next_idx)] += 1
            hidden_combined_to_denominator_key[(cur_idx, next_idx)] = cur_hidden

        final_list = []
        for cur_idx in range(len(self.possible_hiddens)):
            sub_list = []
            for next_idx in range(len(self.possible_hiddens)):
                numerator = hidden_combined_to_count[(cur_idx, next_idx)]
                denominator = self._hidden_occurences[hidden_combined_to_denominator_key[(cur_idx, next_idx)]]
                sub_list.append(numerator / denominator)
            final_list.append(sub_list)
        self.transition_matrix = final_list

    def fit(self, observed_and_hidden):
        self.vocabulary, self.possible_hiddens = set(), set()
        for observed, hidden in observed_and_hidden:
            self.vocabulary.add(observed)
            self.possible_hiddens.add(hidden)

        for idx, hidden in enumerate(self.possible_hiddens):
            self._hidden_to_idx[hidden] = idx

        default_probability: float = 1 / len(self.possible_hiddens)

        # Handles unknown words with 1/K
        self.observed_to_emission_probabilities = keydefaultdict(lambda x: defaultdict(int) if x in self.vocabulary else defaultdict(lambda: default_probability))
        self._build_emission_probabilities(observed_and_hidden)
        self._build_transition_probabilities(observed_and_hidden)


if __name__ == "__main__":
    # Sanity checks
    nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset="universal"))
    train_set, test_set = train_test_split(nltk_data, train_size=0.95, test_size=0.05, random_state=123)
    train_tagged_words = [tup for sent in train_set for tup in sent]
    test_tagged_words = [tup for sent in test_set for tup in sent]
    test = HMM()
    test.fit(train_tagged_words)
    print(test.observed_to_emission_probabilities)
    print(test.transition_matrix)

    for potential_tag in test.possible_hiddens:
        running_sum = 0
        for word in test.observed_to_emission_probabilities.keys():
            running_sum += test.observed_to_emission_probabilities[word][potential_tag]
        print(running_sum == 1, running_sum)
    print("=====")
    for idx, potential_tags in enumerate(test.possible_hiddens):
        running_sum = 0
        for idx_2, potential_tags_2 in enumerate(test.possible_hiddens):
            running_sum += test.transition_matrix[idx][idx_2]
        print(running_sum==1, running_sum)

    sentences = []
    local_sent = []
    num_found = 0
    for item in test_tagged_words:
        if item[0] == ".":
            local_sent.append(item)
            sentences.append(local_sent)
            local_sent = []
            num_found += 1

        else:
            local_sent.append(item)

    # Sanity Check
    for sentence in sentences[0:3]:
        print([i[0] for i in sentence])
        print([i[1] for i in sentence])
        print(viterbi([i[0] for i in sentence], list(test.possible_hiddens), test.transition_matrix, test.observed_to_emission_probabilities))

    num_correct = 0
    for sentence in sentences:
        out = viterbi([i[0] for i in sentence], list(test.possible_hiddens), test.transition_matrix, test.observed_to_emission_probabilities)
        num_correct += len([i for idx, i in enumerate(out) if i == sentence[idx][1]])

    print(num_correct / len(test_tagged_words))
