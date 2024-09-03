import numpy as np
import pandas as pd

import re


class MultinomialNaiveBayesCLF:
    def __init__(self, alpha: int = 1):
        self.n = None
        self.spam_words = []
        self.mail_words = []
        self.probs_spam = {}
        self.probs_mail = {}
        self.spam_len = None
        self.mail_len = None
        self.alpha = alpha

    def fit(self, x: pd.DataFrame, y: pd.Series, coll_name: str) -> None:
        """
        :param coll_name: column with messages
        """
        spam_indexes = y[y == 1].index
        mail_indexes = y[y == 0].index

        for row in x.iloc[spam_indexes][coll_name]:
            self.spam_words += self.prep_message(row)
        for row in x.iloc[mail_indexes][coll_name]:
            self.mail_words += self.prep_message(row)

        self.spam_len = len(self.spam_words)
        unique, counts = np.unique(self.spam_words, return_counts=True)
        self.spam_words = dict(zip(unique, counts))
        self.mail_len = len(self.mail_words)
        unique, counts = np.unique(self.mail_words, return_counts=True)
        self.mail_words = dict(zip(unique, counts))
        self.n = self.spam_len + self.mail_len

        for word in np.unique(list(self.spam_words.keys()) + list(self.mail_words.keys())):
            prob1 = (self.spam_words[word] + self.alpha) if word in self.spam_words else self.alpha
            prob2 = (self.mail_words[word] + self.alpha) if word in self.mail_words else self.alpha
            self.probs_spam[word] = prob1 / (self.spam_len + self.alpha * self.n)
            self.probs_mail[word] = prob2 / (self.mail_len + self.alpha * self.n)

    def predict(self, x: pd.DataFrame, coll_name: str) -> np.array:
        classes = []
        for mail in x[coll_name]:
            mail = self.prep_message(mail)
            classes.append(self.pdf(mail))
        return np.array(classes)

    def pdf(self, message: list[str]) -> int:
        pos_cl, neg_cl = self.spam_len / self.n, self.mail_len / self.n
        for word in message:
            if word in self.probs_spam:
                pos_cl *= self.probs_spam[word]
            if word in self.probs_mail:
                neg_cl *= self.probs_mail[word]
        return 1 if pos_cl >= neg_cl else 0

    @staticmethod
    def prep_message(message: str) -> list[str]:
        split_message = lambda message: message.split(' ')
        drop_sym = lambda word: re.sub('[^a-zA-z]', '', word)

        return [drop_sym(word) for word in split_message(message) if drop_sym(word)]
