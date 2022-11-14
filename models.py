import datetime
import enum
from collections import defaultdict
from typing import List, Set, Any, Tuple, Dict
import scipy
import tqdm
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from inference_methods import viterbi, gibbs_sampling, variational_inference, variational_inference_special
from time import process_time_ns


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
                    self.observed_to_emission_probabilities[possible_words][hiddens] = 1E-15
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
                    sub_list.append(1E-15)
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
            lambda x: defaultdict(lambda: 1E-15) if x in self.vocabulary else {})
        self._build_emission_probabilities(observed_and_hidden)
        self._build_transition_probabilities(observed_and_hidden)

    def get_transition_probabilities(self, prev, current, *args):
        return self.transition_matrix[prev][current]

    def get_emission_probabilities(self, observed, hidden, *args):
        try:
            return self.observed_to_emission_probabilities[observed][hidden]
        except KeyError:
            # The word was not seen at all in our training set
            return 1/len(self.possible_hiddens)

    def get_possible_hiddens(self, *args):
        return self.possible_hiddens

    def inference(self, test_data, algorithm: InferenceMethods = InferenceMethods.VITERBI, alt: bool = False):
        """
        # TODO: Implement Remaining Methods
        # TODO: Test time-to-infer
        # TODO: Report all evaluation metrics we care about
        """
        if algorithm == InferenceMethods.VITERBI:
            num_correct: int = 0
            total: int = 0
            t = process_time_ns()
            for seq, hidden_to_rep in tqdm.tqdm(test_data):
                out = viterbi([i[0] for i in seq], self.get_possible_hiddens(len(seq)), self.get_transition_probabilities,
                              self.get_emission_probabilities, hidden_to_rep)
                num_correct += len([i for idx, i in enumerate(out) if i == seq[idx][1]])
                total += len(seq)
            final = process_time_ns() - t
            return num_correct / total, final
        elif algorithm == InferenceMethods.GIBBS:
            num_correct: int = 0
            total: int = 0
            t = process_time_ns()
            for seq, hidden_to_rep in tqdm.tqdm(test_data):
                total += len(seq)
                out = gibbs_sampling([i[0] for i in seq], self.get_possible_hiddens(len(seq)), self.get_transition_probabilities,
                                     self.get_emission_probabilities, self.hidden_to_idx, hidden_to_rep, no_iterations=5)
                num_correct += len([i for idx, i in enumerate(out) if i == seq[idx][1]])
            final = process_time_ns() - t
            return num_correct / total, final
        elif algorithm == InferenceMethods.VARIATIONAL_INFERENCE:
            num_correct: int = 0
            total: int = 0
            t = process_time_ns()
            for seq, hidden_to_rep in tqdm.tqdm(test_data):
                if alt:
                    out = variational_inference_special([i[0] for i in seq], self.get_possible_hiddens(len(seq)), self.get_transition_probabilities,
                                                self.get_emission_probabilities, hidden_to_rep)
                else:
                    out = variational_inference([i[0] for i in seq], self.get_possible_hiddens(len(seq)), self.get_transition_probabilities,
                                                self.get_emission_probabilities, hidden_to_rep)
                total += len(seq)
                num_correct += len([i for idx, i in enumerate(out) if i == seq[idx][1]])
            final = process_time_ns() - t
            return num_correct / total, final
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
        print(test_data)
        for seq, _ in test_data:
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


class AlignmentHMM(HMM):
    def __init__(self, alignment_to_word, *args, **kwargs):
        self.alignment_to_word = alignment_to_word
        super(AlignmentHMM, self).__init__()

    def get_possible_hiddens(self, *args):
        x = args[0]
        return [i for i in range(x)]

    def get_emission_probabilities(self, observed, hidden, *args):
        try:
            return self.observed_to_emission_probabilities[observed][hidden]
        except KeyError:
            # The word was not seen at all in our training set
            return 1/args[0]

    def get_transition_probabilities(self, prev, current, *args):
        # args[0] should be the sequence length
        prob = abs(current-prev) / sum([abs(k-prev) for k in range(args[0])])
        if prob == 0:
            prob += 1E-10
        return prob

    def _build_emission_probabilities(self, observed_and_hidden: List[Tuple[Any, Any]]):
        idx = 0
        local_occurences = defaultdict(int)
        local_possible_hidden = set()
        for observed, hidden in observed_and_hidden:
            local_hidden = self.alignment_to_word[idx][hidden]
            self.observed_to_emission_probabilities[observed][local_hidden] += 1
            self._hidden_occurences[hidden] += 1
            local_occurences[local_hidden] += 1
            local_possible_hidden.add(local_hidden)
            idx += 1

        for possible_words in self.vocabulary:
            local_normalization = 0
            for hiddens in local_possible_hidden:
                current_count = self.observed_to_emission_probabilities[possible_words][hiddens]
                if current_count == 0:
                    self.observed_to_emission_probabilities[possible_words][hiddens] = 1E-5
                else:
                    self.observed_to_emission_probabilities[possible_words][hiddens] = current_count / \
                                                                                       local_occurences[
                                                                                           hiddens]
                local_normalization += self.observed_to_emission_probabilities[possible_words][hiddens]

            for hiddens in local_possible_hidden:
                self.observed_to_emission_probabilities[possible_words][hiddens] /= local_normalization


def evaluate_alignment():
    stemmer = PorterStemmer()
    from nltk.corpus import comtrans
    words = comtrans.aligned_sents('alignment-en-fr.txt')
    words_train, words_test = train_test_split(words, train_size=0.95, test_size=0.05, random_state=123)
    words_train_processed = []
    words_test_processed = []
    maps_for_training = defaultdict(dict)
    idx = 0
    for train in words_train:
        seen = set()
        for alignment in train.alignment:
            if train.mots[alignment[1]].lower() in seen: continue
            else:
                english_idx = alignment[0]
                if english_idx in maps_for_training[idx]:continue
                else:
                    maps_for_training[idx][english_idx] = train.words[english_idx].lower()
                    idx += 1
                    seen.add(train.mots[alignment[1]].lower())
        seq = []
        seen = set()
        for alignment in train.alignment:
            if train.mots[alignment[1]].lower() in seen: continue
            else:
                seq.append((train.mots[alignment[1]].lower(), alignment[0]))
                seen.add(train.mots[alignment[1]].lower())
        words_train_processed.extend(seq)
    for idx, test in enumerate(words_test):
        def get_default(ind):
            try:
                return test.words[ind]
            except IndexError:
                return test.words[0]

        mapper = keydefaultdict(lambda ind: get_default(ind))
        seen = set()
        for alignment in test.alignment:
            if test.mots[alignment[1]].lower() in seen: continue
            english_idx = alignment[0]
            if english_idx in maps_for_training[idx]:continue
            else:
                mapper[english_idx] = test.words[english_idx].lower()
                idx += 1
                seen.add(test.mots[alignment[1]].lower())
        seq = []
        seen = set()
        for alignment in test.alignment:
            if test.mots[alignment[1]].lower() in seen: continue
            else:
                seq.append((test.mots[alignment[1]].lower(), alignment[0]))
                seen.add(test.mots[alignment[1]].lower())
        words_test_processed.append((seq, mapper))

    tester = AlignmentHMM(maps_for_training)
    tester.fit(words_train_processed[0:500000])
    test_seq = words_test_processed[0:10]
    print("Now doing inference")
    accuracy_g = tester.inference(test_seq, algorithm=InferenceMethods.GIBBS)
    print(f"Gibbs: {accuracy_g}")
    accuracy_v = tester.inference(test_seq, algorithm=InferenceMethods.VITERBI)
    print(f"Viterbi: {accuracy_v}")
    accuracy_vi = tester.inference(test_seq, algorithm=InferenceMethods.VARIATIONAL_INFERENCE)
    print(f"Variational Inference: {accuracy_vi}")
    accuracy_vi_s = tester.inference(test_seq, algorithm=InferenceMethods.VARIATIONAL_INFERENCE, alt=True)
    print(f"Variational Inference SPECIAL: {accuracy_vi_s}")

def evaluate_pos():
    nltk_data = list(nltk.corpus.treebank.tagged_sents())
    train_set, test_set = train_test_split(nltk_data, train_size=0.95, test_size=0.05, random_state=123)
    train_tagged_words = [tup for sent in train_set for tup in sent]
    test_tagged_words = [tup for sent in test_set for tup in sent]

    baseline = Baseline()
    print(len(train_tagged_words))
    print(len(test_tagged_words))
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
    print(len(sentences))

    test_seqs = [test_tagged_words]
    processed_test_seqs = []
    for seq in test_seqs:
        mapped = keydefaultdict(lambda x: x)
        processed_test_seqs.append((seq, mapped))

    processed_test_seqs = [(processed_test_seqs[0][0][0:1000], processed_test_seqs[0][1])]
    print(processed_test_seqs)
    # accuracy_b = baseline.inference(processed_test_seqs)
    # print(f"Baseline: {accuracy_b}")
    # accuracy_g = test.inference(processed_test_seqs, algorithm=InferenceMethods.GIBBS)
    # print(f"Gibbs: {accuracy_g}")
    # accuracy_v = test.inference(processed_test_seqs, algorithm=InferenceMethods.VITERBI)
    # print(f"Viterbi: {accuracy_v}")
    accuracy_vi = test.inference(processed_test_seqs, algorithm=InferenceMethods.VARIATIONAL_INFERENCE)
    print(f"Variational Inference: {accuracy_vi}")
    accuracy_vi_s = test.inference(processed_test_seqs, algorithm=InferenceMethods.VARIATIONAL_INFERENCE, alt=True)
    print(f"Variational Inference SPECIAL: {accuracy_vi_s}")
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


if __name__ == "__main__":
    # Sanity checks
    np.random.seed(225530)
    # evaluate_pos()
    evaluate_alignment()
