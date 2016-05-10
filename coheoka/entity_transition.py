# -*- coding: utf-8 -*-
'''
Entity transition based on entity grid
Reference: Barzilay, R., & Lapata, M. (2008).
    Modeling local coherence: An entity-based approach.
    Computational Linguistics, 34(1), 1-34.
'''
from __future__ import print_function, division
import doctest
from itertools import product
from pprint import pprint

import pandas as pd

from entity_grid import EntityGrid, Constants


class EntityTransition(object):
    '''
    Local entity transition
    >>> eg = EntityGrid('I like apple juice. He also likes it.')
    >>> et = EntityTransition(eg.resolve_coreference())
    >>> et.transition_table.shape[1] <= 4
    True
    '''

    def __init__(self, eg, n=2):
        self._n = n
        self._grid = eg.grid
        self._transition_table = self._column_transitions(n)

    @property
    def n(self):
        '''Return transition order n'''
        return self._n

    @property
    def grid(self):
        '''Return entity grid'''
        return self._grid

    @property
    def transition_table(self):
        '''Return transition table'''
        return self._transition_table

    def make_new_transition_table(self, another_n=3):
        '''Generate a new transition table'''
        self._transition_table = self._column_transitions(another_n)
        self._n = another_n
        return self

    def all_prob(self):
        '''Calculate a feature vector using all transitions'''
        seq = [Constants.SUB, Constants.OBJ, Constants.OTHER, Constants.NOSHOW]
        probs = {}
        for pro in product(seq, repeat=self.n):
            probs[''.join(pro)] = self.prob(pro)
        return probs

    def prob(self, tran):
        '''Calculate probability of a transition'''
        import operator as op
        assert len(tran) == self.n
        tbl = self.transition_table
        freq, total = 0, op.mul(*tbl.shape)
        for _col in tbl.columns:
            col = tbl[_col]
            freq += col.tolist().count(tuple(tran))
        return freq / total

    def _column_transition(self, col, n):
        column = self.grid[col].tolist()
        if len(column) < n:
            # this is a trick to handle the case
            # where transition length is greater than the
            # number of sentences
            column_tran = [
                tuple(column + [Constants.NOSHOW] * (n - len(column)))
            ]
        else:
            column_tran, tran_len = [], len(column) - n + 1
            for i in range(tran_len):
                column_tran.append(tuple(column[i:i + n]))
        return column_tran

    def _column_transitions(self, n):
        transition_table = {}
        for col in self.grid.columns:
            transition_table[col] = self._column_transition(col, n)
        return pd.DataFrame.from_dict(transition_table)


class TransitionMatrix(object):
    '''
    Transition matrix
    >>> tm = TransitionMatrix(['I like apple juice. He also likes it.'])
    >>> 'SS' == tm.tran_matrix.columns[0]
    True
    >>> 'SO' == tm.tran_matrix.columns[1]
    True
    '''

    def __init__(self, corpus, n=2, coref=True):
        self._corpus = corpus
        self._n = n
        self._tran_list = self._make_tran_list(coref, n)
        self._tran_matrix = self._make_tran_matrix()

    @property
    def corpus(self):
        '''Retrun the corpus'''
        return self._corpus

    @property
    def n(self):
        '''Return the order n of transitions'''
        return self._n

    @property
    def tran_list(self):
        '''Return transition list'''
        return self._tran_list

    @property
    def tran_matrix(self):
        '''Return sorted transition matrix ordered by S, O, X, -'''
        mat = self._tran_matrix
        seq = [Constants.SUB, Constants.OBJ, Constants.OTHER, Constants.NOSHOW]
        return mat[sorted(mat.columns,
                          key=lambda x: [seq.index(c) for c in x])]

    @property
    def all_transitions(self):
        '''Return all transition produce by {S, O, X, N}^n'''
        seq = [Constants.SUB, Constants.OBJ, Constants.OTHER, Constants.NOSHOW]
        return [''.join(t) for t in product(seq, repeat=self.n)]

    def _make_tran_list(self, coref, n):
        tran_list = []
        new_corpus = []
        for i, doc in enumerate(self.corpus):
            try:
                if coref:
                    eg = EntityGrid(doc).resolve_coreference()
                else:
                    eg = EntityGrid(doc)
                tran_list.append(EntityTransition(eg, n))
                new_corpus.append(doc)
            except (UnicodeError, TypeError) as e:
                print(doc)
                print('Error detected at {}: {}'.format(i, e))
        self._corpus = new_corpus

        return tran_list

    def _make_tran_matrix(self):
        mat = {}
        for tran in self.all_transitions:
            mat[tran] = [t.all_prob()[tran] for t in self.tran_list]
        return pd.DataFrame.from_dict(mat)


def test_et(text, n=2):
    pprint(text)
    eg = EntityGrid(text)
    et = EntityTransition(eg, n)

    pprint(et.transition_table)
    pprint(et.all_prob())


def test_tm(*test, **kw):
    tm = TransitionMatrix(test, kw['n'])
    pprint(tm.tran_matrix)


if __name__ == '__main__':
    doctest.testmod()
