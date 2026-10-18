from k_means import *

import time

if __name__ == "__main__":
	means = KMeans(points_list=np.loadtxt("classif_data/gmm3d.asc"),
				   number_of_classes=5,
				   epsilon=0.0005,
				   max_iterations=100)

	means.compute()
	means.plot_3d()
