from efros_leung import *

if __name__ == "__main__":
	synthesis = EfrosLeung("textures_data/text0.png", 256, 7, 0.1)

	synthesis.initialize_synthesized_image()
	synthesis.plot_image()