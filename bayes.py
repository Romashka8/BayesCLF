from interface import *

import numpy as np

import re


# ----------------------------------------------------------------------------------------------------------------------------

class BayesGaussian(BayesInterface):

	def __init__(self):

		"""
		Вариант для работы с непрерывными признаками, которые имеют нормальное(Гауссовское) распределение.
		Для Гауссовского Байеса задаются априорные вероятности классов,
		а также массив средних и стандартных отклонений.
		"""

		self.priors = None
		self.x_mean = None
		self.x_std = None

	def fit(self, x, y):

		"""
		Реализуется вычисление априорных вероятностей,
		а также вычисление среднего и стандартного отклонения.
		"""

		# Подсчет объектов в каждом из классов.
		classes, cls_count = np.unique(y, return_counts=True)
		n_classes = len(classes)

		# Вычисление априорных вероятностей.
		self.priors = cls_count / len(y)

		# Вычислим средее и стандартное отклонение матрицы признаков в зависимости от класса
		self.x_mean = np.array([np.mean(x[y == cl], axis=0) for cl in range(n_classes)])
		self.x_std = np.array([np.std(x[y == cl], axis=0) for cl in range(n_classes)])

	def predict(self, x):

		"""
		Вычисляем апостериорные вероятности как произведение
		априорных вероятностей классов и распределения тестовых признаков.
		Классы с максимальной апостериорной вероятностью будут итоговым прогнозом.
		"""

		pdfs = np.array([self.pdf(X, self.x_mean, self.x_std) for X in x])
		posteriors = self.priors * np.prod(pdfs, axis=2) # укороченная формула Байеса

		return np.argmax(posteriors, axis=1)

	@staticmethod
	def pdf(x, mean, std):

		"""
		Вычисление плотности вероятностного распределения признаков,
		согласно распределению Гаусса(нормальное распределение).
		"""

		return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# ----------------------------------------------------------------------------------------------------------------------------

class BayesMultinominal(BayesInterface):

	"""
	Вариант классификатора для работы с дискретными признаками,
	которые имеют мультиномиальное распределение.
	Такая вариация чаще всего используется в задачах классификации писем на спам.
	Реализуем классификатор для фильтрации спама.
	"""

	def __init__(self, alpha=1):
		
		"""
		Задаем необходимы для подсчета априорных вероятностей параметры.
		"""

		self.n = None
		self.alpha = alpha

		self.spam_words = []
		self.mail_words = []

		self.probs_spam = {}
		self.probs_mail = {}

		self.spam_len = None
		self.mail_len = None

	def fit(self, x, y, coll_name):

		"""
		Имплементация подсчета апостериорных вероятностей.
		x: pd.DataFrame
		y: pd.Series
		"""

		spam_indexes = y[y == 1].index
		mail_indexes = y[y == 0].index

		for row in x.iloc[spam_indexes][coll_name]: self.spam_words += self.prep_message(row)

		for row in x.iloc[mail_indexes][coll_name]: self.mail_words += self.prep_message(row)

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

	def predict(self):

		"""
		Реализация predict-а.
		"""

		classes = []

		for mail in x[coll_name]:

			mail = self.prep_message(mail)
			classes.append(self.pdf(mail))

		return np.array(classes)

	def pdf(self, message):

		"""
		Плотность вероятностного распределения.
		"""

		pos_cl, neg_cl = self.spam_len / self.n, self.mail_len / self.n

		for word in message:

			if word in self.probs_spam: pos_cl *= self.probs_spam[word]

			if word in self.probs_mail: neg_cl *= self.probs_mail[word]

		return 1 if pos_cl >= neg_cl else 0

	@staticmethod
	def prep_message(message):

		"""
		Предобработка сообщений.
		"""

		split_messages = lambda message: message.split(' ')
		drop_sym = lambda word: re.sub('[^a-zA-z]', '', word)

		return [drop_sym(word) for word in split_messages(message) if drop_sym(word)]

# ----------------------------------------------------------------------------------------------------------------------------
