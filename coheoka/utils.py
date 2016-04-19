# -*- coding: utf-8 -*-
'''
Preprocessing utilities
'''
from nltk import sent_tokenize
from random import shuffle, sample


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
