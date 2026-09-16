from efros_leung import *

import time 

if __name__ == "__main__":
	start = time.time()
	synthesis = EfrosLeung("textures_data/text0.png", 64, 9, 0.1, 31)
	synthesis.initialize_synthesized_image()
	synthesis.fill_synthesized_image_partial(1000)
	end = time.time()
	print(f"Elapsed time : {end - start:.2f} seconds")
	synthesis.plot_image()
	

