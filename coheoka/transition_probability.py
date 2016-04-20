# -*- coding: utf-8 -*-
'''
Entity transition probability
'''
from __future__ import print_function, division
from math import log

from entity_transition import EntityTransition
from entity_grid import EntityGrid
import utils


class EntityProbability(object):
    def __init__(self, text, coref=True):
        self._eg = EntityGrid(text).resolve_coreference(
        ) if coref else EntityGrid(text)
        self._et = EntityTransition(self._eg)
        self._table = self._et.transition_table
        self._probs = self._et.all_prob()
        self._prob_vector = self._coherence_prob()

    @property
    def entity_transition(self):
        return self._et

    @property
    def probs(self):
        return self._probs

    def _get_column_prob(self, col):
        trans = [log(self._probs[''.join(itm)])
                 for itm in self._table[col].tolist()]
        return sum(trans) / len(trans)

    def _coherence_prob(self):
        res = []
        for col in self._table.columns:
            res.append(self._get_column_prob(col))
        return sum(res) / len(res)


if __name__ == '__main__':
    T1 = '''
        The Justice Department is conducting an anti-trust trial against Microsoft Corp with evidence that the company is increasingly attempting to crush competitors.
        Microsoft is accused of trying to forcefully buy into markets where its own products are not competitive enough to unseat established brands.
        The case revolves around evidence of Microsoft aggressively pressuring Netscape into merging browser software.
        Microsoft claims its tactics are commonplace and good economically.
        The government may file a civil suit ruling that conspiracy to curb competition through collusion is a violation of the Sherman Act.
        Microsoft continues to show increased earnings despite the trial.
        '''

    T2 = 'I have a friend called Bob. He loves playing basketball. I also love playing basketball. We play basketball together sometimes.'
    T3 = 'I like apple juice. He also likes it.'
    T4 = 'The Justice Department is conducting an anti-trust trial against\
              Microsoft Corp with evidence that the company is increasingly attempting to crush competitors.'

    from pprint import pprint
    print(EntityProbability(T1)._coherence_prob())
    pprint([(t, EntityProbability(t)._coherence_prob())
            for t in utils.add_sents(T1, 5, T2)])
    pprint([(t, EntityProbability(t)._coherence_prob())
            for t in utils.remove_sents(T1, 5)])
