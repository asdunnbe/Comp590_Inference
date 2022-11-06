import datetime
import enum
from collections import defaultdict
from typing import List, Set, Any, Tuple, Dict
import scipy
import tqdm
import nltk
import numpy as np
from sklearn.model_selection import train_test_split

from inference_methods import viterbi, gibbs_sampling, variational_inference


# Needed since we want different defaults if it is in or not in the dictionary
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class InferenceMethods(enum.Enum):
    VITERBI = 0
    VARIATIONAL_INFERENCE = 1
    CONSTRAINED_INFERENCE = 2
    INTEGER_PROGRAMMING = 3
    GIBBS = 4


class HMM:
    def __init__(self, vocabulary=None):
        # Note we make this a dictionary as otherwise we would have to handle mapping a word to its index
        # in our vocabulary, that would be somewhat annoying and potentially bug-prone.
        self.observed_to_emission_probabilities: Dict[Any, Dict[Any, float]] = None
        self.transition_matrix: List[List[float]] = None
        self.vocabulary: Set[Any] = None
        self.possible_hiddens: List[Any] = None
        self.hidden_to_idx: Dict[Any, int] = defaultdict(int)
        self._hidden_occurences: Dict[Any, int] = defaultdict(int)

    def _build_emission_probabilities(self, observed_and_hidden: List[Tuple[Any, Any]]):
        """
        Builds emission probabilities by looping through the training data only once
        """
        for observed, hidden in observed_and_hidden:
            self.observed_to_emission_probabilities[observed][hidden] += 1
            self._hidden_occurences[hidden] += 1

        for possible_words in self.vocabulary:
            local_normalization = 0
            for hiddens in self.possible_hiddens:
                current_count = self.observed_to_emission_probabilities[possible_words][hiddens]
                if current_count == 0:
                    self.observed_to_emission_probabilities[possible_words][hiddens] = 1E-10
                else:
                    self.observed_to_emission_probabilities[possible_words][hiddens] = current_count / \
                                                                                       self._hidden_occurences[
                                                                                           hiddens]
                local_normalization += self.observed_to_emission_probabilities[possible_words][hiddens]

            for hiddens in self.possible_hiddens:
                self.observed_to_emission_probabilities[possible_words][hiddens] /= local_normalization

    def _build_transition_probabilities(self, observed_and_hidden: List[Tuple[Any, Any]]):
        """
        Builds transition probabilities by looping through the training data only once
        """
        hidden_combined_to_denominator_key: Dict[Tuple[int, 2], Any] = {}
        hidden_combined_to_count: Dict[Tuple[int, 2], int] = defaultdict(int)
        for idx, obv_and_hid in enumerate(observed_and_hidden):
            # Takes the current index idx as y_i-1 and then does p(y | y_i-1)
            cur_hidden = obv_and_hid[1]
            try:
                next_hidden = observed_and_hidden[idx + 1][1]
            except IndexError:
                break
            cur_idx = self.hidden_to_idx[cur_hidden]
            next_idx = self.hidden_to_idx[next_hidden]
            hidden_combined_to_count[(cur_idx, next_idx)] += 1
            hidden_combined_to_denominator_key[(cur_idx, next_idx)] = cur_hidden

        final_list = []
        for cur_idx in range(len(self.possible_hiddens)):
            sub_list = []
            for next_idx in range(len(self.possible_hiddens)):
                if (cur_idx, next_idx) not in hidden_combined_to_denominator_key:
                    # These two tags never appear together in the dataset, make it a small non-zero
                    # value (type of smoothing)
                    sub_list.append(1E-10)
                else:
                    numerator = hidden_combined_to_count[(cur_idx, next_idx)]
                    denominator = self._hidden_occurences[hidden_combined_to_denominator_key[(cur_idx, next_idx)]]
                    sub_list.append(numerator / denominator)
            # softmax so that we don't have to worry about an invalid probability dist from
            # the case where tags don't appear together
            final_list.append(np.array(sub_list) / sum(sub_list))
        self.transition_matrix = final_list

    def fit(self, observed_and_hidden):
        """
        Fits model to the data
        """
        self.vocabulary, self.possible_hiddens = set(), []
        for observed, hidden in observed_and_hidden:
            self.vocabulary.add(observed)
            if hidden not in self.possible_hiddens:
                self.possible_hiddens.append(hidden)

        for idx, hidden in enumerate(self.possible_hiddens):
            self.hidden_to_idx[hidden] = idx

        default_probability: float = 1 / len(self.possible_hiddens)

        # Handles unknown words with 1/K
        self.observed_to_emission_probabilities = keydefaultdict(
            lambda x: defaultdict(int) if x in self.vocabulary else defaultdict(lambda: default_probability))
        self._build_emission_probabilities(observed_and_hidden)
        self._build_transition_probabilities(observed_and_hidden)

    def inference(self, test_data, algorithm: InferenceMethods = InferenceMethods.VITERBI):
        """
        # TODO: Implement Remaining Methods
        # TODO: Test time-to-infer
        # TODO: Report all evaluation metrics we care about
        """
        if algorithm == InferenceMethods.VITERBI:
            num_correct: int = 0
            total: int = 0
            for seq in tqdm.tqdm(test_data):
                out = viterbi([i[0] for i in seq], self.possible_hiddens, self.transition_matrix,
                              self.observed_to_emission_probabilities)
                num_correct += len([i for idx, i in enumerate(out) if i == seq[idx][1]])
                total += len(seq)
            return num_correct / total
        elif algorithm == InferenceMethods.GIBBS:
            num_correct: int = 0
            total: int = 0
            for seq in tqdm.tqdm(test_data):
                total += len(seq)
                out = gibbs_sampling([i[0] for i in seq], self.possible_hiddens, self.transition_matrix,
                                     self.observed_to_emission_probabilities, self.hidden_to_idx, no_iterations=50)
                num_correct += len([i for idx, i in enumerate(out) if i == seq[idx][1]])
            return num_correct / total
        elif algorithm == InferenceMethods.VARIATIONAL_INFERENCE:
            num_correct: int = 0
            total: int = 0
            for seq in tqdm.tqdm(test_data):
                out = variational_inference([i[0] for i in seq], self.possible_hiddens, self.transition_matrix,
                                            self.observed_to_emission_probabilities, )
                total += len(seq)
                num_correct += len([i for idx, i in enumerate(out) if i == seq[idx][1]])
            return num_correct / total
        elif algorithm == InferenceMethods.CONSTRAINED_INFERENCE:
            raise NotImplementedError("Constrained Inference is not Implemented!")
        elif algorithm == InferenceMethods.INTEGER_PROGRAMMING:
            raise NotImplementedError("Integer Programming is not Implemented!")


class Baseline:
    def __init__(self):
        self.hidden_to_idx = {}
        self.vocabulary = None
        self.possible_hiddens = None
        self.word_to_tag = None
        self.tag_to_tag = None

    def fit(self, observed_and_hidden):
        self.vocabulary, self.possible_hiddens = set(), []
        for observed, hidden in observed_and_hidden:
            self.vocabulary.add(observed)
            if hidden not in self.possible_hiddens:
                self.possible_hiddens.append(hidden)

        print(len(self.possible_hiddens))
        for idx, hidden in enumerate(self.possible_hiddens):
            self.hidden_to_idx[hidden] = idx

        self.word_to_tag = keydefaultdict(
            lambda x: [0 for _ in self.possible_hiddens] if x in self.vocabulary else self.possible_hiddens[0])
        self.tag_to_tag = defaultdict(lambda: [0 for _ in self.possible_hiddens])

        for idx, word_tag in enumerate(observed_and_hidden):
            word = word_tag[0]
            current_tag = word_tag[1]
            current_tag_idx = self.hidden_to_idx[current_tag]
            self.word_to_tag[word][current_tag_idx] += 1

            if idx != len(observed_and_hidden) - 1:
                next_label = observed_and_hidden[idx + 1][1]
                next_label_idx = self.hidden_to_idx[next_label]
                self.tag_to_tag[current_tag_idx][next_label_idx] += 1

        for key in self.word_to_tag:
            self.word_to_tag[key] = np.argmax(self.word_to_tag[key])
        for key in self.tag_to_tag:
            self.tag_to_tag[key] = np.argmax(self.tag_to_tag[key])

    def inference(self, test_data):
        no_correct = 0
        total = 0
        for seq in test_data:
            for word, tag in seq:
                pred = self.word_to_tag[word]
                try:
                    pred = self.possible_hiddens[pred]
                except TypeError:
                    pass
                if pred == tag:
                    no_correct += 1
                total += 1
        return no_correct / total


if __name__ == "__main__":
    # Sanity checks
    np.random.seed(225530)
    nltk_data = list(nltk.corpus.treebank.tagged_sents())
    train_set, test_set = train_test_split(nltk_data, train_size=0.95, test_size=0.05, random_state=123)
    train_tagged_words = [tup for sent in train_set for tup in sent]
    test_tagged_words = [tup for sent in test_set for tup in sent]

    baseline = Baseline()
    baseline.fit(train_tagged_words)

    test = HMM()
    test.fit(train_tagged_words)

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
    accuracy_b = baseline.inference(sentences)
    print(f"Baseline: {accuracy_b}")
    # TODO: Viterbi degrades in performance when using all words in a row (instead of sentences) - perhaps there is a bug?
    accuracy_g = test.inference([test_tagged_words], algorithm=InferenceMethods.GIBBS)
    print(f"Gibbs: {accuracy_g}")
    accuracy_v = test.inference([test_tagged_words], algorithm=InferenceMethods.VITERBI)
    print(f"Viterbi: {accuracy_v}")
    accuracy_vi = test.inference([test_tagged_words], algorithm=InferenceMethods.VARIATIONAL_INFERENCE)
    print(f"Variational Inference: {accuracy_vi}")
    with open(f"Results.txt", "w") as f:
        f.write(f"RUN {datetime.datetime.now()}\n")
        f.write(f"Baseline: {accuracy_b}\n")
        f.write(f"Viterbi: {accuracy_v}\n")
        f.write(f"Gibbs: {accuracy_g}\n")
        f.write(f"Variational Inference: {accuracy_vi}\n")


    # Can be used to verify proper probability distributions
    # for potential_tag in test.possible_hiddens:
    #     running_sum = 0
    #     for word in test.observed_to_emission_probabilities.keys():
    #         running_sum += test.observed_to_emission_probabilities[word][potential_tag]
    #
    # for idx, potential_tags in enumerate(test.possible_hiddens):
    #     running_sum = 0
    #     for idx_2, potential_tags_2 in enumerate(test.possible_hiddens):
    #         running_sum += test.transition_matrix[idx][idx_2]
