# -*- coding: utf-8 -*-
'''
Build entity grid using StanfordCoreNLP
'''
from __future__ import print_function, division
from itertools import product
from collections import defaultdict
import doctest
from pprint import pprint
import pandas as pd

from corenlp import StanfordCoreNLP


class CoreNLP(object):
    '''Connect CoreNLP server'''
    _NLP = StanfordCoreNLP('http://localhost:9000')
    _LOCAL_DEMO_PROP = {
        'annotators':
        'tokenize, ssplit, pos, lemma, ner, depparse, openie, coref',
        "openie.resolve_coref": "true",
        'outputFormat': 'json'
    }
    _ONLINE_DEMO_PROP = {
        "annotators": "tokenize,ssplit,pos,ner,depparse,openie,coref",
        "coref.md.type": "dep",
        "coref.mode": "statistical",
        'outputFormat': 'json'
    }

    @staticmethod
    def annotate(text):
        '''Get result from CoreNLP via JSON'''
        try:
            return CoreNLP.nlp().annotate(text,
                                          properties=CoreNLP._ONLINE_DEMO_PROP)
        except UnicodeError:
            pprint(text)

    @staticmethod
    def nlp():
        '''Return CoreNLP Server'''
        return CoreNLP._NLP


class Constants(object):
    '''Some constants'''
    REMOVE_ABBR = {'Inc.', 'Inc', 'Corp.', 'Corp'}
    _NOUNS = {'NN', 'NNS', 'NNP', 'NNPS', 'PRP'}
    # S O X
    _SUBJECTS = {'subj', 'nsubj', 'nsubjpass', 'csubj', 'csubjpass'}
    _OBJECTS = {'obj', 'iobj', 'dobj'}
    SUB, OBJ, OTHER, NOSHOW = 'S', 'O', 'X', '-'

    @staticmethod
    def noun_tags():
        """Get noun POS tags"""
        return Constants._NOUNS

    @staticmethod
    def get_role(dep):
        """Indentify an entity's grammatical role"""
        if dep in Constants._SUBJECTS:
            return Constants.SUB
        elif dep in Constants._OBJECTS:
            return Constants.OBJ
        else:
            return Constants.OTHER


class EntityGrid(object):
    '''
    Entity grid
    >>> eg = EntityGrid('My friend is Bob. He loves playing basketball.')
    >>> 'friend' in eg.grid.columns and 'he' in eg.grid.columns
    True
    >>> 'he' not in eg.resolve_coreference().grid.columns
    True
    '''

    def __init__(self, text):
        self.text = ' '.join([token
                              for token in text.split(' ')
                              if token not in Constants.REMOVE_ABBR])
        self._data = CoreNLP.annotate(self.text)
        self._sentences = self._data['sentences']

        self._depens = [s['basic-dependencies'] for s in self._sentences]
        self._entity_tokens = [
            [t for t in s['tokens']
             if t['pos'] in Constants.noun_tags()] for s in self._sentences
        ]

        self._noun2lemma = self._set_up_noun2lemma()
        self._grid = self._set_up_grid()

    @property
    def grid(self):
        """Entity grid"""
        return self._grid

    @property
    def nouns(self):
        """All nouns in text"""
        return self._noun2lemma.keys()

    @property
    def lemmas(self):
        """All lemmas in text"""
        return self._noun2lemma.values()

    def noun2lemma(self, noun):
        """Convert a noun to its lemma"""
        return self._noun2lemma[noun] if noun in self.nouns else None

    def _set_up_noun2lemma(self):
        noun2lemma = {}
        for token in self._entity_tokens:
            for ety in token:
                noun2lemma[ety['word']] = ety['lemma']
        return noun2lemma

    def _set_up_grid(self):
        depens, entities, noun2lemma = self._depens, self._entity_tokens, self._noun2lemma
        assert len(depens) == len(entities)
        grid = defaultdict(
            lambda: [Constants.NOSHOW for i in range(len(depens))])

        for i, (dep, ety) in enumerate(zip(depens, entities)):
            nouns = [e['word'] for e in ety]
            try:
                [d['dependentGloss'] for d in dep]
            except KeyError:
                pprint(dep)
                pprint(i)
                pprint(self.text)
            nouns_dp = [
                d
                for d in dep
                if d['dependentGloss'] in nouns and d['dep'] != 'compound'
            ]

            for n_dp in nouns_dp:
                grid[noun2lemma[n_dp['dependentGloss']]][i] = \
                    Constants.get_role(n_dp['dep']) # yapf: disable
        return pd.DataFrame.from_dict(grid)

    def _map_phrase_to_entity(self, phrase):
        '''e.g. my friend => friend, friend in grid
        my friend is Bob => friend, friend and Bob in grid, choose former
        '''
        nouns = [w for w in phrase.split(' ') if w in self.nouns]
        lemmas = [self.noun2lemma(w)
                  for w in nouns if self.noun2lemma(w) in self.grid.columns]
        #pprint(lemmas)
        return lemmas[0] if lemmas != [] else None

    def _add_column(self, _c1, _c2):
        '''Add grid[c2] to grid[c1]'''

        assert len(self.grid[_c1]) == len(self.grid[_c2])

        assert _c1 != _c2
        col1, col2 = self.grid[_c1], self.grid[_c2]
        for i, _col1 in enumerate(col1):
            if _col1 == Constants.NOSHOW:
                col1[i] = col2[i]
        self.grid.pop(_c2)
        return _c1

    def _add_columns(self, _c1, *c):
        '''Add columns of grid to the first'''
        reduce(self._add_column, [_c1] + list(c))

    def resolve_coreference(self):
        '''Resolve coreference by merging columns in grid'''
        is_rep = 'isRepresentativeMention'

        for chain in [chains
                      for chains in self._data['corefs'].values()
                      if len(chains) > 1]:
            core_entity, other_entities = None, []
            for cor in chain:
                word = self._map_phrase_to_entity(cor['text'])
                if word is not None and word not in other_entities:
                    if cor[is_rep]:
                        core_entity = word
                    elif word != core_entity:
                        other_entities.append(word)
                    else:
                        pass

            if core_entity is not None and other_entities != []:
                self._add_columns(core_entity, *other_entities)
        return self


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

    def make_new_transition_table(self, n=3):
        '''Generate a new transition table'''
        self._transition_table = self._column_transitions(n)
        self._n = n
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
    pprint(eg.grid)
    pprint(eg.resolve_coreference().grid)
    et = EntityTransition(eg, n)

    pprint(et.transition_table)
    pprint(et.all_prob())


def test_tm(*test, **kw):
    tm = TransitionMatrix(test, kw['n'])
    pprint(tm.tran_matrix)
    global df
    df = tm.tran_matrix


if __name__ == '__main__':
    doctest.testmod()
    S, O, X, N = Constants.SUB, Constants.OBJ, Constants.OTHER, Constants.NOSHOW
    T1 = '''
            The Justice Department is conducting an anti-trust trial against Microsoft Corp with evidence that the company is increasingly attempting to crush competitors.
            Microsoft is accused of trying to forcefully buy into markets where its own products are not competitive enough to unseat established brands.
            The case revolves around evidence of Microsoft aggressively pressuring Netscape into merging browser software.
            Microsoft claims its tactics are commonplace and good economically.
            The government may file a civil suit ruling that conspiracy to curb competition through collusion is a violation of the Sherman Act.
            Microsoft continues to show increased earnings despite the trial.
            '''

    #    T1 = 'My friend is Bob. He loves playing basketball. And he also is good at tennis.'

    T2 = 'I have a friend called Bob. He loves playing basketball. I also love playing basketball. We play basketball together sometimes.'
    T3 = 'I like apple juice. He also likes it.'
    T4 = 'The Justice Department is conducting an anti-trust trial against\
              Microsoft Corp with evidence that the company is increasingly attempting to crush competitors.'

    test_et(T2)  #)
#    test_tm(T1*10,T2,n=2)
