# -*- coding: utf-8 -*-
"""
Evaluator based on transition matrix
"""
from __future__ import print_function, division
from sklearn import svm
from nltk import sent_tokenize
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import classification_report
from pprint import pprint

from entity_grid import TransitionMatrix


class Evaluator(object):
    def __init__(self,
                 corpus,
                 shuffle_times=4,
                 origin_label=1,
                 shuffle_label_func=lambda x, y: 0):
        self._corpus = corpus
        self._origin_matrix = self._label_origin_corpus(origin_label)
        self._shuffled_matrix = self._label_shuffled_corpus(shuffle_times,
                                                            shuffle_label_func)
        self._matrix = np.concatenate((self._origin_matrix,
                                       self._shuffled_matrix))
        self._X = None
        self._y = None
        self._clf = None

    @property
    def corpus(self):
        return self._corpus

    @property
    def matrix(self):
        return self._matrix

    @property
    def X(self):
        if self._X is not None:
            return self._X
        else:
            raise AttributeError(
                'Not generated. Please call `evaluate` first.')

    @property
    def y(self):
        if self._y is not None:
            return self._y
        else:
            raise AttributeError(
                'Not generated. Please call `evaluate` first.')

    @property
    def clf(self):
        return self._clf

    def _label_origin_corpus(self, label):
        res = []
        for text in self.corpus:
            res.append((text, label))
        return res

    def _label_shuffled_corpus(self, times, label_func):
        return sum(
            [self._shuffle_text(text, times, label_func)
             for text in self.corpus], [])

    def _shuffle_text(self, text, times, label_func):
        from random import shuffle
        origin_sents = sent_tokenize(text)
        sents = sent_tokenize(text)
        res = []
        for i in range(times):
            shuffle(sents)
            label = label_func(sents, origin_sents)
            res.append((' '.join(sents), label))
        return res

    def evaluate(self, clf=svm.SVC, cv=2):
        #        np.random.shuffle(self.matrix)
        self._X = TransitionMatrix([c for c in self.matrix[:, 0]]).tran_matrix
        self._y = self.matrix[:, 1].astype(int)
        self._clf = clf()
        X_train, X_test, y_train, y_test = self.X, self.X, self.y, self.y
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            self.X,
            self.y,
            test_size=0.4)
        self.clf.fit(X_train, y_train)
        #        print(y_test)
        #        print(self.clf.predict(X_test))

        print(classification_report(y_test, self.clf.predict(X_test)))
        return self.clf.score(X_test, y_test)


def test(*text):
    e = Evaluator(text, 4)
    #    pprint(e.matrix)

    pprint(e.evaluate())
#    pprint(e.X )
#    pprint(e.y)

if __name__ == '__main__':

    T1 = 'My friend is Bob. He loves playing basketball. And he also is good at tennis.'

    T2 = 'I have a friend called Bob. He loves playing basketball. I also love playing basketball. We play basketball together sometimes.'
    T3 = 'I like apple juice. He also likes it.'
    T4 = 'The Justice Department is conducting an anti-trust trial against\
              Microsoft Corp with evidence that the company is increasingly attempting to crush competitors.'

    T1 = '''
        The Justice Department is conducting an anti-trust trial against Microsoft Corp with evidence that the company is increasingly attempting to crush competitors.
        Microsoft is accused of trying to forcefully buy into markets where its own products are not competitive enough to unseat established brands.
        The case revolves around evidence of Microsoft aggressively pressuring Netscape into merging browser software.
        Microsoft claims its tactics are commonplace and good economically.
        The government may file a civil suit ruling that conspiracy to curb competition through collusion is a violation of the Sherman Act.
        Microsoft continues to show increased earnings despite the trial.
        '''

    #    test(T1,T1)
    #    test(T1)
    e = Evaluator(ctxt)
#    e.evaluate()
