import numpy as np
import matplotlib.pyplot as plt
import random

class KMeans:
	def __init__(self):
		self.DATA_2D = np.loadtxt("classif_data/gmm2d.asc")
		self.DATA_3D = np.loadtxt("classif_data/gmm3d.asc")
		self.centers_indices = []
		self.colors = []
		self.labels = []

	def compute(self, points_list, number_of_classes):
		number_of_points = len(points_list)

		for _ in range(number_of_classes):
			random_index = random.randint(0, number_of_points)
			while random_index in self.centers_indices:
				random_index = random.randint(0, number_of_points)
			self.centers_indices.append(random_index)

		print("centers_indices:", self.centers_indices)
		self.colors = [(random.random(), random.random(), random.random()) for _ in range(number_of_classes)]
		print("colors:", self.colors)

		for i in range(number_of_points):
			min_label = 0
			min_distance = np.linalg.norm(points_list[i] - points_list[self.centers_indices[min_label]]) ** 2
			for j in range(number_of_classes):
				distance = np.linalg.norm(points_list[i] - points_list[self.centers_indices[j]]) ** 2
				if distance < min_distance:
					min_distance = distance
					min_label = j
			self.labels.append(min_label)

	def plot_2d(self):
		plt.scatter(self.DATA_2D[:, 0], self.DATA_2D[:, 1],
                    c=[self.colors[label] for label in self.labels],
                    marker="x", s=5)

		plt.scatter(self.DATA_2D[self.centers_indices, 0],
   				    self.DATA_2D[self.centers_indices, 1],
        			c="red", marker="o", s=30)

		plt.axis('equal')
		plt.show()

	def plot_3d(self):
		pass
