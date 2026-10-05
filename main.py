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

	means = KMeans()
	means.compute(means.DATA_3D, 5)
	means.plot_3d()
