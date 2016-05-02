# -*- coding: utf-8 -*-
'''
Coherence probability based on entity grid
'''
from __future__ import print_function, division
from math import log

import numpy as np

from entity_transition import EntityTransition
from entity_grid import EntityGrid
import utils


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
        self._probs = self._make_probs()

    @property
    def corpus(self):
        return self._corpus

    @property
    def probs(self):
        return self._probs

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
        return res

    def _make_probs(self):
        res = []
        for text in self.corpus:
            try:
                p = CoherenceProbability(text).coherence_prob
                res.append(p)
            except:
                print(text)
        return res


if __name__ == '__main__':
    from pprint import pprint
    T1 = '''
        The Justice Department is conducting an anti-trust trial against Microsoft Corp with evidence that the company is increasingly attempting to crush competitors.
        Microsoft is accused of trying to forcefully buy into markets where its own products are not competitive enough to unseat established brands.
        The case revolves around evidence of Microsoft aggressively pressuring Netscape into merging browser software.
        Microsoft claims its tactics are commonplace and good economically.
        The government may file a civil suit ruling that conspiracy to curb competition through collusion is a violation of the Sherman Act.
        Microsoft continues to show increased earnings despite the trial.
        '''

    T2 = 'I have a friend called Bob. He loves playing basketball. I also love playing basketball. We play basketball together sometimes.'
    T3 = 'I like apple juice. He also likes it. He also likes playing basketball.'
    T4 = '''A bank gets money from lenders, and pays interest. The bank then lends this money to borrowers. Banks allow borrowers and lenders of different sizes to meet.'''
    T5 = 'You should heed my advice, people should be using computers. Computers are an exelent way to comunicate.'

    T = T2
    e = CoherenceProbability(T)
    #pprint(e._eg.grid)
    pprint(e.coherence_prob)

    from pprint import pprint
    print(CoherenceProbability(T)._coherence_prob())
    print([('', CoherenceProbability(t)._coherence_prob())
           for t in utils.add_sents(T, 5, T5)])
    print([('', CoherenceProbability(t)._coherence_prob())
           for t in utils.remove_sents(T, 5)])
    print([('', CoherenceProbability(t)._coherence_prob())
           for t in utils.shuffle_sents(T, 5)])
    ct = [T1, T2, T3, T4, T5]
    pv = ProbabilityVector(ct)
    print(pv.probs, pv.mean, pv.std)
    print(pv.evaluate_coherence('I like apple juice. You should hear my advice. Computers are an exelent way to comunicate.'))
