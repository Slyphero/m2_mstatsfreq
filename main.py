from efros_leung import *

if __name__ == "__main__":
	synthesis = EfrosLeung("textures_data/text0.png", 256, 9, 0.1, 3)
	synthesis.initialize_synthesized_image()
	synthesis.fill_synthesized_image()
	print(synthesis.patch_mask)

	synthesis.plot_image()

