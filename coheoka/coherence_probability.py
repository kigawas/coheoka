# -*- coding: utf-8 -*-
'''
Coherence probability based on entity grid
Reference: Lapata, M., & Barzilay, R. (2005, July).
    Automatic evaluation of text coherence: Models and representations.
    In IJCAI (Vol. 5, pp. 1085-1090).
'''
from __future__ import print_function, division
from math import log

import numpy as np
from nltk import sent_tokenize

from entity_grid import EntityGrid


class CoherenceProbability(object):
    def __init__(self, text, coref=True):
        self._eg = EntityGrid(text).resolve_coreference(
        ) if coref else EntityGrid(text)
        self._coher_prob = self._coherence_prob()

    @property
    def grid(self):
        return self._eg.grid

    @property
    def coherence_prob(self):
        return self._coher_prob

    def _get_column_prob(self, col):
        column = self._eg.grid[col].tolist()
        sent_len = len(column)
        assert sent_len > 1
        transition_count = {}
        for tran in zip(column[1:], column[:-1]):
            transition_count[tran] = transition_count.get(tran, 0) + 1
        probs = []
        for i, role in enumerate(column):
            if i == 0:
                probs.append(log(column.count(column[0]) / sent_len))
            else:
                tran_cnt = transition_count[(column[i], column[i - 1])]
                ent_cnt = column.count(column[i - 1])
                probs.append(log(tran_cnt / ent_cnt))
        assert all([p <= 0.0 for p in probs])
        return sum(probs) / len(probs)

    def _coherence_prob(self):
        res = []
        for col in self._eg.grid.columns:
            res.append(self._get_column_prob(col))
        return sum(res) / len(res)


class ProbabilityVector(object):
    def __init__(self, corpus):
        self._corpus = corpus
        self._probs = None

    @property
    def corpus(self):
        return self._corpus

    @property
    def probs(self):
        if self._probs:
            return self._probs
        else:
            raise ValueError('Please call `make_prob` first')

    @property
    def mean(self):
        return np.mean(self.probs)

    @property
    def std(self):
        return np.std(self.probs)

    @property
    def var(self):
        return np.var(self.probs)

    def evaluate_coherence(self, text):
        p = CoherenceProbability(text)
        res = p.coherence_prob - self.mean
        return p.coherence_prob, res

    def make_probs(self):
        res = []
        for text in self.corpus:
            try:
                p = CoherenceProbability(text).coherence_prob
                res.append(p)
            except:
                print(text)
        self._probs = res
        return self


if __name__ == '__main__':
    T = 'I have a friend called Bob. He loves playing basketball. I also love playing basketball. We play basketball together sometimes.'  # NOQA

    e = CoherenceProbability(T)
    print(e.coherence_prob)
