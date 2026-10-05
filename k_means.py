import numpy as np
import matplotlib.pyplot as plt
import random

class KMeans:
	def __init__(self, points_list, number_of_classes, epsilon, max_iterations):
		self.POINTS_LIST = np.array(points_list)
		self.EPSILON = epsilon
		self.NUMBER_OF_CLASSES = number_of_classes
		self.MAX_ITERATIONS = max_iterations
		self.DATA_2D = np.loadtxt("classif_data/gmm2d.asc")
		self.DATA_3D = np.loadtxt("classif_data/gmm3d.asc")
		self.centers = np.empty((number_of_classes, self.POINTS_LIST.shape[1]))
		self.COLORS = [ (random.random(), random.random(), random.random())
			            for _ in range(self.NUMBER_OF_CLASSES) ]
		self.labels = []


	def compute(self):
		number_of_points = len(self.POINTS_LIST)

		# Sélection aléatoire des centres pour l'initialisation
		indices = np.random.choice(number_of_points, self.NUMBER_OF_CLASSES, replace=False)
		self.centers = self.POINTS_LIST[indices]

		# Première itération
		distances = np.sum((self.POINTS_LIST[:, np.newaxis] - self.centers) ** 2, axis=2)
		self.labels = np.argmin(distances, axis=1)

		self.centers = np.array(self.centers)
		self.labels = np.array(self.labels)

		self.initial_labels = self.labels.copy()
		self.initial_centers = self.centers.copy()

		# Boucle jusqu'à convergence
		# Critère de convergence : Si les barycentres à l'étape i - 1  sont suffisamment proches de ceux à l'étape i
		iterations = 0
		while True and iterations < self.MAX_ITERATIONS:
			old_centers = self.centers.copy()
			for i in range(self.NUMBER_OF_CLASSES):
				mask = (self.labels == i)
				if np.any(mask):
					self.centers[i] = np.mean(self.POINTS_LIST[mask], axis=0)

			distances = np.sum((self.POINTS_LIST[:, np.newaxis] - self.centers) ** 2, axis=2)
			self.labels = np.argmin(distances, axis=1)

			if np.all(np.abs(self.centers - old_centers) < self.EPSILON):
				break
			iterations += 1


	def plot_2d(self):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
		# Itération 1
		ax1.scatter(self.POINTS_LIST[:, 0], self.POINTS_LIST[:, 1],
                    c=[self.COLORS[label] for label in self.initial_labels],
	                marker="x", s=10)
		ax1.scatter(self.initial_centers[:, 0], self.initial_centers[:, 1],
	                c="red", marker="o", s=30)
		ax1.set_title("Initialisation")
		ax1.axis('equal')

	    # Convergence
		ax2.scatter(self.POINTS_LIST[:, 0], self.POINTS_LIST[:, 1],
	                c=[self.COLORS[label] for label in self.labels],
	                marker="x", s=10)
		ax2.scatter(self.centers[:, 0], self.centers[:, 1],
	                c="red", marker="o", s=30)
		ax2.set_title("Convergence")
		ax2.axis('equal')

		plt.tight_layout()
		plt.show()


	def plot_3d(self):
		fig = plt.figure(figsize=(14, 6))
		# Itération 1
		ax1 = fig.add_subplot(1, 2, 1, projection='3d')
		ax1.scatter(self.POINTS_LIST[:, 0], self.POINTS_LIST[:, 1], self.POINTS_LIST[:, 2],
                    c=[self.COLORS[label] for label in self.initial_labels],
                    marker="x", s=10)
		ax1.scatter(self.initial_centers[:, 0], self.initial_centers[:, 1], self.initial_centers[:, 2],
			        c="red", marker="o", s=40)
		ax1.set_title("Initialisation")

		# Convergence
		ax2 = fig.add_subplot(1, 2, 2, projection='3d')
		ax2.scatter(self.POINTS_LIST[:, 0], self.POINTS_LIST[:, 1], self.POINTS_LIST[:, 2],
                    c=[self.COLORS[label] for label in self.labels],
                    marker="x", s=10)
		ax2.scatter(self.centers[:, 0], self.centers[:, 1], self.centers[:, 2],
                    c="red", marker="o", s=40)
		ax2.set_title("Convergence")

		plt.tight_layout()
		plt.show()
