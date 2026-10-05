from efros_leung import *
from k_means import *

import time

if __name__ == "__main__":
	"""
	start = time.time()
	synthesis = EfrosLeung(image_path       = "textures_data/text2.png",
						   synthesis_length = 64,
						   patch_size       = 9,
						   epsilon          = 0.1,
						   seed_size        = 15)
	synthesis.initialize_synthesized_image()
	synthesis.fill_synthesized_image_partial(300)
	end = time.time()
	print(f"Elapsed time : {end - start:.2f} seconds")
	synthesis.plot_images()
	"""

	means = KMeans(points_list=np.loadtxt("classif_data/gmm3d.asc"),
				   number_of_classes=5,
				   epsilon=0.0005,
				   max_iterations=100)

	means.compute()
	means.plot_3d()
