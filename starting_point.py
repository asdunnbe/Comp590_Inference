import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# nltk.download('treebank')
# nltk.download('universal_tagset')
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset="universal"))
train_set, test_set = train_test_split(nltk_data, train_size=0.95, test_size=0.05, random_state=123)

train_tagged_words = [tup for sent in train_set for tup in sent]
test_tagged_words = [tup for sent in test_set for tup in sent]
test_words_without_tags = [tup[0] for sent in test_set for tup in sent]
unique_tags = list(set([i[1] for i in train_tagged_words]))

print(len(test_words_without_tags))
vocabulary = set([i[0] for i in train_tagged_words])
print(f"Part A: There are {len(vocabulary)} in the treebank training set")


# TODO: Verify correctness
em_dict = {}
def emission_probability(word, tag):
    """
    Part B:
    Calculates p(w | tag) by doing p(w and tag) / p ( tag)
    """
    # NOTE This is where we are handling OOV (I just make it a uniform distribution = very high entropy)

    if word not in vocabulary:
        return 1 / len(unique_tags)


    if f"{word}_{tag}" in em_dict:
        return em_dict[f"{word}_{tag}"]

    found = 0
    total = 0
    for train_w, train_t in train_tagged_words:
        if train_w == word and train_t == tag:
            found += 1
            total += 1
        elif train_t == tag:
            total += 1
    em_dict[f"{word}_{tag}"] = found / total
    return found / total


# TODO: Verify correctness
def transition_probability(tag1, tag2):
    """Calculates P(y2 | y1) = p(y2 and y1) / p(y1)
    :param tag1:
    :param tag2:
    :return:
    """
    found = 0
    total = 0
    for idx in range(0, len(train_tagged_words), 2):
        search_tag_1 = train_tagged_words[idx][1]
        try:
            search_tag_2 = train_tagged_words[idx + 1][1]
        except IndexError:
            break
        if tag1 == search_tag_1 and tag2 == search_tag_2:
            found += 1
            total += 1
        elif tag1 == search_tag_1:
            total += 1
    return found / total


transition_matrix = []
for tag1 in unique_tags:
    local = []
    for tag2 in unique_tags:
        local.append(transition_probability(tag1, tag2))
    transition_matrix.append(local)

print(f"Question D: T is {len(unique_tags)}")
for i in range(3):
    print(f"Question D: row {i} is {[round(i, 3) for i in transition_matrix[i]]}")


def score(y, x, i, pre_comp_table=None):
    # computes score(y, i)
    if pre_comp_table is None:
        pre_comp_table = {}

    if i == 0:
        return emission_probability(x[i], y), ["<START>"]

    max_s = 0
    max_prev_y = None
    best_prev = []
    best_possible_y = None
    for possible_y in unique_tags:
        if i - 1 in pre_comp_table:
            if possible_y in pre_comp_table[i - 1]:
                prev_score, prev_y = pre_comp_table[i - 1][possible_y]
            else:
                prev_score, prev_y = score(possible_y, x, i - 1, pre_comp_table)
                pre_comp_table[i - 1][possible_y] = [prev_score, prev_y]
        else:
            prev_score, prev_y = score(possible_y, x, i - 1, pre_comp_table)
            pre_comp_table[i - 1] = {}
            pre_comp_table[i - 1][possible_y] = [prev_score, prev_y]
        if (new_max := transition_probability(possible_y, y) * emission_probability(x[i], y) * prev_score) > max_s:
            max_s = new_max
            best_possible_y = possible_y
            best_prev = prev_y

    best_prev.append(best_possible_y)
    return max_s, best_prev


def vert(x):
    score_table = [[0 for _ in range(len(unique_tags))] for _ in range(len(x))]
    best_table = [[0 for _ in range(len(unique_tags))] for _ in range(len(x))]
    for idx, tag in enumerate(unique_tags):
        score_table[0][idx] = emission_probability(x[0], tag)

    for word_idx, word in enumerate(x[1:]):
        for tag_idx, tag in enumerate(unique_tags):
            best_score = 0
            for old_tag in range(len(unique_tags)):
                local_score = score_table[word_idx][old_tag] * transition_matrix[old_tag][tag_idx] * emission_probability(word, tag)
                if best_score < local_score:
                    score_table[word_idx + 1][tag_idx] = local_score
                    best_table[word_idx + 1][tag_idx] = old_tag
                    best_score = local_score

    current_best = np.argmax(best_table[len(x)-1])
    out = [unique_tags[current_best]]
    for i in range(len(x)-1, 0, -1):
        current_best = best_table[i][current_best]
        out.append(unique_tags[current_best])
    out.reverse()
    return out


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

for sentence in sentences[0:3]:
    print([i[0] for i in sentence])
    print([i[1] for i in sentence])
    print(vert([i[0] for i in sentence]))

num_correct = 0
for sentence in sentences:
    out = vert([i[0] for i in sentence])
    num_correct += len([i for idx, i in enumerate(out) if i == sentence[idx][1]])


print(num_correct / len(test_tagged_words))
