from interface import *

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------------

class BayesGaussian(BayesInterface):

	def __init__(self):

		"""
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

		return np.agmax(posteriors, axis=1)

	@staticmethod
	def pdf(x, mean, std):

		"""
		Вычисление плотности вероятностного распределения признаков,
		согласно распределению Гаусса(нормальное распределение).
		"""

		return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# ----------------------------------------------------------------------------------------------------------------------------
