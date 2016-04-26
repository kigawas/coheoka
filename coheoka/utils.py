# -*- coding: utf-8 -*-
'''
Preprocessing utilities
'''

from random import shuffle, sample

from nltk import sent_tokenize
from scipy.stats import kendalltau as tau


def shuffle_sents(text, times):
    sents = sent_tokenize(text)
    res = []
    for i in range(times):
        shuffle(sents)
        res.append(' '.join(sents))
    return res


def remove_sents(text, times, remove_number=1):
    sents = sent_tokenize(text)
    res = []
    for i in range(times):
        res.append(' '.join(sample(sents, len(sents) - remove_number)))
    return res


def add_sents(text, times, added_text, add_number=1):
    sents = sent_tokenize(text)
    sents.append(added_text)
    res = []
    for i in range(times):
        shuffle(sents)
        res.append(' '.join(sents))
    return res


def tau_score_of_sentents(sent1_tokens, sent2_tokens):
    assert len(sent1_tokens) == len(sent2_tokens)
    t = tau(sent1_tokens, sent2_tokens)[0]
    if t <= 0.33:
        return -1
    elif t > 0.33 and t <= 0.66:
        return 0
    else:
        return 1
