# -*- coding: utf-8 -*-
'''
Build entity grid using StanfordCoreNLP
Reference: Barzilay, R., & Lapata, M. (2008).
    Modeling local coherence: An entity-based approach.
    Computational Linguistics, 34(1), 1-34.
'''
from __future__ import print_function, division
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
        depens, entities, noun2lemma = self._depens, self._entity_tokens,\
            self._noun2lemma
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
                    Constants.get_role(n_dp['dep'])  # yapf: disable
        return pd.DataFrame.from_dict(grid)

    def _map_phrase_to_entity(self, phrase):
        '''e.g. my friend => friend, friend in grid
        my friend is Bob => friend, friend and Bob in grid, choose former
        '''
        nouns = [w for w in phrase.split(' ') if w in self.nouns]
        lemmas = [self.noun2lemma(w)
                  for w in nouns if self.noun2lemma(w) in self.grid.columns]
        # pprint(lemmas)
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


if __name__ == '__main__':
    doctest.testmod()
