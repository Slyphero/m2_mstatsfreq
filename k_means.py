import numpy as np
import matplotlib.pyplot as plt
import random

class KMeans:
	def __init__(self):
		self.DATA_2D = np.loadtxt("classif_data/gmm2d.asc")
		self.DATA_3D = np.loadtxt("classif_data/gmm3d.asc")
		self.centers_indices = []

	def compute(self, points_list, number_of_classes):
		number_of_points = len(points_list)

		for i in range(number_of_classes):
			random_index = random.randint(0, number_of_points)
			while random_index in self.centers_indices:
				random_index = random.randint(0, number_of_points)
			self.centers_indices.append(random_index)

		print("centers_indices:", self.centers_indices)



	def plot_2d(self):
		plt.plot(self.DATA_2D[:, 0], self.DATA_2D[:, 1], "x")
		plt.axis('equal')
		plt.show()

	def plot_3d(self):
		pass
