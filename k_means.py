import numpy as np
import matplotlib.pyplot as plt
import random

class KMeans:
	def __init__(self, points_list, number_of_classes):
		self.points_list = np.array(points_list)
		self.number_of_classes = number_of_classes
		self.DATA_2D = np.loadtxt("classif_data/gmm2d.asc")
		self.DATA_3D = np.loadtxt("classif_data/gmm3d.asc")
		self.centers = np.empty((number_of_classes, self.points_list.shape[1]))
		self.colors = []
		self.labels = []


	def compute(self):
		number_of_points = len(self.points_list)
		picked = set()

		# Sélection aléatoire des centres pour l'initialisation
		for i in range(self.number_of_classes):
			while True:
				random_index = random.randint(0, number_of_points - 1)
				if random_index not in picked:
					break
			self.centers[i] = self.points_list[random_index]

		self.colors = [ (random.random(), random.random(), random.random())
			            for _ in range(self.number_of_classes) ]

		# Première itération
		distances = np.linalg.norm(self.points_list[:, np.newaxis] - self.centers, axis=2)
		self.labels = np.argmin(distances, axis=1)

		self.centers = np.array(self.centers)
		self.labels = np.array(self.labels)

		self.initial_labels = self.labels.copy()
		self.initial_centers = self.centers.copy()

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
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
		# Itération 1
		ax1.scatter(self.points_list[:, 0], self.points_list[:, 1],
                    c=[self.colors[label] for label in self.initial_labels],
	                marker="x", s=10)
		ax1.scatter(self.initial_centers[:, 0], self.initial_centers[:, 1],
	                c="red", marker="o", s=30)
		ax1.set_title("Initialisation")
		ax1.axis('equal')

	    # Convergence
		ax2.scatter(self.points_list[:, 0], self.points_list[:, 1],
	                c=[self.colors[label] for label in self.labels],
	                marker="x", s=10)
		ax2.scatter(self.centers[:, 0], self.centers[:, 1],
	                c="red", marker="o", s=30)
		ax2.set_title("Convergence")
		ax2.axis('equal')

		plt.tight_layout()
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
