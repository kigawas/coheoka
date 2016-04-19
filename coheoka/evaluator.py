# -*- coding: utf-8 -*-
"""
Evaluator based on transition matrix
"""
from __future__ import print_function, division
from nltk import sent_tokenize

from sklearn import cross_validation, svm
from pprint import pprint
import numpy as np
from scipy.stats import kendalltau as tau

from entity_transition import TransitionMatrix
from ranking import transform_pairwise


class Evaluator(object):
    def __init__(self,
                 corpus,
                 shuffle_times=20,
                 origin_label=1,
                 shuffle_label_func=lambda x, y: -1):
        self._corpus = corpus
        self._origin_matrix = self._label_origin_corpus(origin_label)
        self._shuffled_matrix = self._label_shuffled_corpus(shuffle_times,
                                                            shuffle_label_func)
        self._matrix = np.concatenate((self._origin_matrix,
                                       self._shuffled_matrix))
        self._X, self._y, self._clf = None, None, None

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
                'Not generated. Please call `make_data_and_clf` first.')

    @property
    def y(self):
        if self._y is not None:
            return self._y
        else:
            raise AttributeError(
                'Not generated. Please call `make_data_and_clf` first.')

    @property
    def clf(self):
        if self._clf is not None:
            return self._clf
        else:
            raise AttributeError(
                'Not generated. Please call `make_data_and_clf` first.')

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
            res.append((' '.join(sents[:]), label))
        return res

    def make_data_and_clf(self, clf=svm.LinearSVC):
        if self._X is None:
            self._X = TransitionMatrix([c for c in self.matrix[:, 0]
                                        ]).tran_matrix.as_matrix()
            self._y = self.matrix[:, 1].astype(int)
            self._clf = clf
        else:
            pass
        return self

    def predict(self, clf, X):
        return np.dot(X, clf.coef_.ravel())

    def get_ranking_order(self, clf, X):
        return np.argsort(clf.predict(X))

    def evaluate_tau(self, test_size=0.3):
        X, y = transform_pairwise(self.X, self.y)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X,
            y,
            test_size=test_size)
        c = self.clf()
        c.fit(X_train, y_train)
        return tau(self.predict(c, X_test), y_test)

    def evaluate_accuracy(self, test_size=0.3):
        X, y = transform_pairwise(self.X, self.y)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X,
            y,
            test_size=test_size)
        c = self.clf()
        c.fit(X_train, y_train)
        return c.score(X_test, y_test)


def test(*text):
    e = Evaluator(text).make_data_and_clf()
    pprint([e.evaluate_accuracy() for i in range(5)])
    pprint([e.evaluate_tau()[0] for i in range(5)])


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

    T2 = '''
        Meier was born in Amberg. Before "GNTM" she studied mathematics.
        While shopping at a mall, she was invited by a model scout to a casting for "GNTM". Out of 16,421 girls in the casting, she was chosen among 14 other girls to be on the TV show.
        During the show she won a role alongside Heidi Klum in a TV commercial for McDonald's.
        In the last episode (the finale), she won the show and became "Germany's Next Topmodel".
        After "Germany's Next Topmodel" Meier was in many magazines around the world such as "Vogue" (Taiwan), "Madame Figaro" (Russia) and "L'Officiel" (France) and worked for many brands such as "Pantene".
        In her private life, Meier is in a steady relationship since 2003.'''

    test(*[T1, T2, T3, T4])
    e = Evaluator(ct)
    e.make_data_and_clf()
    pprint(e.evaluate_accuracy())
