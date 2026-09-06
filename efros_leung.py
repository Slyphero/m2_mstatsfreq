from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint

class EfrosLeung:
	def __init__(self, image_path, synthesis_length, patch_size, epsilon, seed_size):
		# Initialisation des paramètres
		self.SYNTHESIS_LENGTH = synthesis_length
		self.PATCH_SIZE = patch_size
		self.EPSILON = epsilon
		self.SEED_SIZE = seed_size
		self.SEED = np.array(Image.new("RGB", [self.SEED_SIZE, self.SEED_SIZE]), dtype=np.int16) - 1

		self.TEXTURE_ARRAY = np.array(Image.open(image_path))
		self.TEXTURE_HEIGHT, self.TEXTURE_WIDTH, _ = np.shape(self.TEXTURE_ARRAY)


	def __sample_example_texture(self):
		seed_y = randint(0, self.TEXTURE_HEIGHT - self.SEED_SIZE)
		seed_x = randint(0, self.TEXTURE_WIDTH - self.SEED_SIZE)

		for i in range(0, self.SEED_SIZE):
			for j in range(0, self.SEED_SIZE):
				self.SEED[i, j] = self.TEXTURE_ARRAY[seed_y + i, seed_x + j]
		

	def initialize_synthesized_image(self):
		self.__sample_example_texture()

		# -1 pour ne pas confondre les pixels vides avec les éventuels pixels noirs de la texture
		self.final_image_array = np.array(
			Image.new("RGB", [self.SYNTHESIS_LENGTH, self.SYNTHESIS_LENGTH]),
			dtype=np.int16
		) - 1

		image_center_x = (self.SYNTHESIS_LENGTH - 1) // 2
		image_center_y = (self.SYNTHESIS_LENGTH - 1) // 2

		for i in range(0, self.SEED_SIZE):
			for j in range(0, self.SEED_SIZE): 
				self.final_image_array[i + image_center_y - self.SEED_SIZE // 2, 
						               j + image_center_x - self.SEED_SIZE // 2] = self.SEED[i, j]


	def __count_direct_filled_neighbors(self, x, y):
		neighbors = 0
		for i in range(y - 1, y + 2):
			for j in range(x - 1, x + 2):
				if i == y and j == x:
					continue 

				if i >= 0 and j >= 0 and i < self.SYNTHESIS_LENGTH and j < self.SYNTHESIS_LENGTH:
					if self.final_image_array[i, j, 0] != - 1:
						neighbors += 1
		return neighbors


	def __fill_candidates_pixels(self):
		candidates_pixels = []

		for y in range(0, self.SYNTHESIS_LENGTH):
			for x in range(0, self.SYNTHESIS_LENGTH): 
				if self.__count_direct_filled_neighbors(x, y) > 0 and self.final_image_array[y, x, 0] == -1:
					candidates_pixels.append((y, x))
		return candidates_pixels
	

	def fill_synthesized_image(self):
		print(self.__fill_candidates_pixels())
		# while np.any(self.final_image_array == -1):
		# 	best_x, best_y = 0, 0

		# 	for i in range(0, self.SYNTHESIS_LENGTH):
		# 		for j in range(0, self.SYNTHESIS_LENGTH):
		# 			neighbors_count = 0

		# 			for k in range(i - self.PATCH_SIZE, i + self.PATCH_SIZE):
		# 				for l in range(j - self.PATCH_SIZE, j + self.PATCH_SIZE):
		# 					continue

		# 	continue


	def plot_image(self):
		plt.title("Image synthétisée")
		plt.imshow(self.final_image_array)
		plt.show()