from typing import List, Any

import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

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


# transition_matrix = []
# for tag1 in unique_tags:
#     local = []
#     for tag2 in unique_tags:
#         local.append(transition_probability(tag1, tag2))
#     transition_matrix.append(local)
#
# print(f"Question D: T is {len(unique_tags)}")
# for i in range(3):
#     print(f"Question D: row {i} is {[round(i, 3) for i in transition_matrix[i]]}")


print(emission_probability("Reliance", "VERB"))
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
