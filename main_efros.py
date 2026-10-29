from efros_leung import *

import time

if __name__ == "__main__":
	start = time.time()
	synthesis = EfrosLeung(image_path       = "textures_data/text5.png",
						   synthesis_length = 128,
						   patch_size       = 21,
						   epsilon          = 0.6,
						   seed_size        = 11)
	synthesis.initialize_synthesized_image()
	synthesis.fill_synthesized_image()
	end = time.time()
	print(f"Elapsed time : {end - start:.2f} seconds")
	synthesis.plot_images()
