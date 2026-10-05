import numpy as np
import matplotlib.pyplot as plt
import random

class KMeans:
	def __init__(self, points_list, number_of_classes):
		self.points_list = np.array(points_list)
		self.number_of_classes = number_of_classes
		self.DATA_2D = np.loadtxt("classif_data/gmm2d.asc")
		self.DATA_3D = np.loadtxt("classif_data/gmm3d.asc")
		self.centers = []
		self.centers_indices = []
		self.colors = []
		self.labels = []


	def compute(self):
		number_of_points = len(self.points_list)

		# Sélection aléatoire des centres pour l'initialisation
		for _ in range(self.number_of_classes):
			random_index = random.randint(0, number_of_points - 1)
			while random_index in self.centers_indices:
				random_index = random.randint(0, number_of_points - 1)
			self.centers_indices.append(random_index)
			self.centers.append(self.points_list[random_index])

		self.colors = [ (random.random(), random.random(), random.random())
			            for _ in range(self.number_of_classes) ]

		# Première itération
		distances = np.linalg.norm(self.points_list[:, np.newaxis] - self.centers, axis=2)
		self.labels = np.argmin(distances, axis=1)

		self.centers = np.array(self.centers)
		self.labels = np.array(self.labels)

		# Boucle jusqu'à convergence
		# Critère de convergence : Si les barycentres à l'étape i - 1  sont suffisamment proches de ceux à l'étape i
		EPSILON = 0.0001
		while True:
			old_centers = self.centers.copy()
			for i in range(self.number_of_classes):
				mask = (self.labels == i)
				if np.any(mask):
					self.centers[i] = np.mean(self.points_list[mask], axis=0)

			distances = np.linalg.norm(self.points_list[:, np.newaxis] - self.centers, axis=2)
			self.labels = np.argmin(distances, axis=1)

			if np.all(np.abs(self.centers - old_centers) < EPSILON):
				break

	def plot_2d(self):
		plt.scatter(self.DATA_2D[:, 0], self.DATA_2D[:, 1],
                    c=[self.colors[label] for label in self.labels],
                    marker="x", s=10)

		plt.scatter(self.DATA_2D[self.centers_indices, 0],
   				    self.DATA_2D[self.centers_indices, 1],
        			c="red", marker="o", s=30)

		plt.axis('equal')
		plt.show()


	def plot_3d(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.DATA_3D[:, 0], self.DATA_3D[:, 1], self.DATA_3D[:, 2],
	               c=[self.colors[label] for label in self.labels],
	               marker="x", s=10)
		ax.scatter(self.DATA_3D[self.centers_indices, 0],
	               self.DATA_3D[self.centers_indices, 1],
	               self.DATA_3D[self.centers_indices, 2],
	               c="red", marker="o", s=40)
		plt.show()
