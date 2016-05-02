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
from utils import tau_score_of_sentents


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
        self._X, self._y, self._clf, self._fitted_clf = None, None, None, None

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

    @property
    def fitted_clf(self):
        if self._fitted_clf is not None:
            return self._fitted_clf
        else:
            raise AttributeError('Not generated. Please call `fit` first.')

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
        assert len(origin_sents) > 1
        sents = sent_tokenize(text)
        res = []
        for i in range(times):
            shuffle(sents)
            label = label_func(sents, origin_sents)
            res.append((' '.join(sents[:-1]), label))
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

    def fit(self):
        X, y = transform_pairwise(self.X, self.y)
        self._fitted_clf = self.clf().fit(X, y)
        return self

    def evaluate_coherence(self, text):
        origin_len = len(self._origin_matrix)
        x = TransitionMatrix([text]).tran_matrix.as_matrix()
        y = [-1]
        ref_x = self.X[:origin_len]
        ref_y = [1] * origin_len
        X = np.concatenate((x, ref_x))
#        y = np.concatenate((y, ref_y))
#        X, y = transform_pairwise(X, y)
        return self.predict(self.fitted_clf,x)


def test(*text):
    global e
    e = Evaluator(text).make_data_and_clf()
    pprint([e.evaluate_accuracy() for i in range(5)])
    pprint([e.evaluate_tau()[0] for i in range(5)])
    #text = '''A railfan is a person who likes railways and trains.  You should hear my advice. Computers are an exelent way to comunicate.'''
    #text = 'The Justice Department is conducting an '
    t = 'My friend is Bob. He loves playing basketball. And he also is good at tennis.'
    e.fit()
    print(e.evaluate_coherence(t))


if __name__ == '__main__':

    T1 = 'My friend is Bob. He loves playing basketball. And he also is good at tennis.'

    T2 = 'I have a friend called Bob. He loves playing basketball. I also love playing basketball. We play basketball together sometimes.'
    T3 = 'I like apple juice. He also likes it. And he almost drinks apple juice every day.'
    T4 = '''An invention is a new thing that someone has made. The computer was an invention when it was first made. We say when it was "invented". New things that are made or created are called inventions. The car is an invention that everyone knows.
Ideas are also called inventions. Writers can invent characters, and then invent a story about them. Inventions are made by inventors.
Inventing.'''

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
    #e = Evaluator(w20)
    #e = Evaluator(w20, shuffle_label_func=tau_score_of_sentents)
    #e.make_data_and_clf()
    #    pprint(e.evaluate_accuracy())
    #    pprint(e.evaluate_accuracy())
    #    pprint(e.evaluate_accuracy())
