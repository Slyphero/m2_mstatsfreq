import numpy as np 
import matplotlib.pyplot as plt 

class KMeans:
	def __init__(self):
		self.DATA_2D = np.loadtxt("classif_data/gmm2d.asc")
		self.DATA_3D = np.loadtxt("classif_data/gmm3d.asc")


	def plot_2d(self):
		plt.plot(self.DATA_2D[:, 0], self.DATA_2D[:, 1], "x")
		plt.axis('equal')
		plt.show()

	def plot_3d(self):
		pass

		

