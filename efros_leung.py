from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint

class EfrosLeung:
	def __init__(self, image_path, synthesis_length, patch_size, epsilon, seed_size):
		# Initialisation des paramètres
		self.SYNTHESIS_LENGTH = synthesis_length
		self.EPSILON = epsilon
		
		self.SEED_SIZE = seed_size
		self.SEED = np.array(Image.new("RGB", [self.SEED_SIZE, self.SEED_SIZE]), dtype=np.int16) - 1

		self.PATCH_SIZE = patch_size

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


	def __count_direct_filled_neighbors(self, y, x):
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
				if self.__count_direct_filled_neighbors(y, x) > 0 and self.final_image_array[y, x, 0] == -1:
					candidates_pixels.append((y, x))
		return candidates_pixels


	def __count_filled_neighbors_patch(self, y_candidate, x_candidate):
		neighbors = 0
		for y in range(y_candidate - self.PATCH_SIZE // 2, y_candidate + self.PATCH_SIZE // 2 + 1):
			for x in range(x_candidate - self.PATCH_SIZE // 2, x_candidate + self.PATCH_SIZE // 2 + 1):
				if y == y_candidate and x == x_candidate:
					continue 

				if y >= 0 and x >= 0 and y < self.SYNTHESIS_LENGTH and x < self.SYNTHESIS_LENGTH:
					if self.final_image_array[y, x, 0] != -1:
						neighbors += 1
		return neighbors


	def __set_patch_and_mask(self, y_best, x_best):
		self.patch = np.array(Image.new("RGB", [self.PATCH_SIZE, self.PATCH_SIZE]), dtype=np.int16) - 1
		self.patch_mask = np.zeros([self.PATCH_SIZE, self.PATCH_SIZE], dtype=np.int16)

		for y in range(0, self.PATCH_SIZE):
			for x in range(0, self.PATCH_SIZE):
				offset_y = y_best - self.PATCH_SIZE // 2 + y
				offset_x = x_best - self.PATCH_SIZE // 2 + x

				is_offset_valid = (offset_y >= 0 and offset_x >= 0 and 
					   offset_y < self.SYNTHESIS_LENGTH and offset_x < self.SYNTHESIS_LENGTH)

				if is_offset_valid and self.final_image_array[offset_y, offset_x, 0] != -1:
					self.patch[y, x] = self.final_image_array[offset_y, offset_x]
					self.patch_mask[y, x] = 1


	def __find_candidate_coordinates(self):
		ssd_array = np.array([])

		for y in range(0, self.TEXTURE_HEIGHT):
			for x in range(0, self.TEXTURE_WIDTH):
				pass


	def fill_synthesized_image(self):
		candidates = self.__fill_candidates_pixels()

		neighbors_array = []
		for candidate in candidates:
			neighbors_array.append(self.__count_filled_neighbors_patch(candidate[0], candidate[1]))

		best_index = np.argmax(neighbors_array)
		self.__set_patch_and_mask(candidates[best_index][0], candidates[best_index][1])



	def plot_image(self):
		plt.title("Image synthétisée")
		plt.imshow(self.final_image_array)
		plt.show()