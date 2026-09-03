from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint

class EfrosLeung:
	def __init__(self, image_path, synthesis_length, patch_size, epsilon):
		self.SYNTHESIS_LENGTH = synthesis_length
		self.PATCH_SIZE = patch_size
		self.EPSILON = epsilon

		self.TEXTURE_ARRAY = np.array(Image.open(image_path))
		self.TEXTURE_WIDTH, self.TEXTURE_HEIGHT, _ = np.shape(self.TEXTURE_ARRAY)

		# -1 pour ne pas confondre les pixels vides avec les éventuels pixels noirs de la texture
		self.final_image_array = np.array(
			Image.new("RGB", [self.SYNTHESIS_LENGTH, self.SYNTHESIS_LENGTH]),
			dtype=np.int16
		) - 1

	def initialize_synthesized_image(self):
		seed_x = randint(0, self.SYNTHESIS_LENGTH - self.TEXTURE_WIDTH - 1)
		seed_y = randint(0, self.SYNTHESIS_LENGTH - self.TEXTURE_HEIGHT - 1)

		# Ajout du sample texture dans l'image finale
		for i in range(seed_x, seed_x + self.TEXTURE_WIDTH):
			for j in range(seed_y, seed_y + self.TEXTURE_HEIGHT):
				self.final_image_array[i, j] = self.TEXTURE_ARRAY[i - seed_x, j - seed_y]

	def fill_synthesized_image(self):
		while not np.any(self.final_image_array == -1):
			best_x, best_y = 0, 0

			for i in range(0, self.SYNTHESIS_LENGTH):
				for j in range(0, self.SYNTHESIS_LENGTH):
					neighbors_count = 0

					for k in range(i - self.PATCH_SIZE, i + self.PATCH_SIZE):
						for l in range(j - self.PATCH_SIZE, j + self.PATCH_SIZE):
							continue

			continue


	def plot_image(self):
		plt.title("Image synthétisée")
		plt.imshow(self.final_image_array)
		plt.show()