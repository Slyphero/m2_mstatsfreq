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

		self.SEED = self.TEXTURE_ARRAY[seed_y : seed_y + self.SEED_SIZE,
								       seed_x : seed_x + self.SEED_SIZE]
		

	def initialize_synthesized_image(self):
		self.__sample_example_texture()

		# -1 pour ne pas confondre les pixels vides avec les éventuels pixels noirs de la texture
		self.final_image_array = np.array(
			Image.new("RGB", [self.SYNTHESIS_LENGTH, self.SYNTHESIS_LENGTH]),
			dtype=np.int16
		) - 1

		image_center_x = (self.SYNTHESIS_LENGTH - 1) // 2
		image_center_y = (self.SYNTHESIS_LENGTH - 1) // 2

		start_x = image_center_x - self.SEED_SIZE // 2
		start_y = image_center_y - self.SEED_SIZE // 2

		self.final_image_array[start_y : start_y + self.SEED_SIZE,
						 	   start_x : start_x + self.SEED_SIZE] = self.SEED
				

	def __fill_candidates_pixels(self):
		candidates_pixels = []

		for y in range(0, self.SYNTHESIS_LENGTH):
			for x in range(0, self.SYNTHESIS_LENGTH): 
				if self.final_image_array[y, x, 0] == -1:
					if self.__count_filled_neighbors_patch(y, x) > 0:
						candidates_pixels.append((y, x))
		return candidates_pixels


	def __count_filled_neighbors_patch(self, y_candidate, x_candidate):
		half = self.PATCH_SIZE // 2

		patch_area = self.final_image_array[max(0, y_candidate - half) : 
									        	min(self.SYNTHESIS_LENGTH, y_candidate + half + 1),
											max(0, x_candidate - half) : 
												min(self.SYNTHESIS_LENGTH, x_candidate + half + 1)]

		neighbors = np.sum(patch_area[:, :, 0] != -1)
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
		half = self.PATCH_SIZE // 2
		y_range = self.TEXTURE_HEIGHT - self.PATCH_SIZE + 1
		x_range = self.TEXTURE_WIDTH - self.PATCH_SIZE + 1
		
		ssd_matrix = np.full((self.TEXTURE_HEIGHT, self.TEXTURE_WIDTH), np.inf)

		mask_3d = self.patch_mask[:, :, np.newaxis]
		patch_3d = self.patch.astype(np.int32)

		for y in range(0, y_range):
			for x in range(0, x_range):
				bloc_source = self.TEXTURE_ARRAY[y : y + self.PATCH_SIZE, x : x + self.PATCH_SIZE].astype(np.int32)				
				diff_masquee = (bloc_source - patch_3d) * mask_3d
				ssd_matrix[y + half, x + half] = np.sum(diff_masquee ** 2)

		ssd_min = np.min(ssd_matrix)
		threshold = ssd_min * (1 + self.EPSILON)

		matching_candidates = np.argwhere(ssd_matrix <= threshold)
		choice = randint(0, len(matching_candidates) - 1)
		chosen_y, chosen_x = matching_candidates[choice]

		return (chosen_y, chosen_x)
	

	def fill_synthesized_image(self):
		candidates = self.__fill_candidates_pixels()

		while len(candidates) > 0:
			neighbors_array = []
			for candidate in candidates:
				neighbors_array.append(self.__count_filled_neighbors_patch(candidate[0], candidate[1]))

			best_index = np.argmax(neighbors_array)
			y_best, x_best = candidates[best_index][0], candidates[best_index][1]

			candidates.pop(best_index)
			self.__set_patch_and_mask(y_best, x_best)

			chosen_y, chosen_x = self.__find_candidate_coordinates()
			self.final_image_array[y_best, x_best] = self.TEXTURE_ARRAY[chosen_y, chosen_x]

			for i in range(y_best - 1, y_best + 2):
				for j in range (x_best - 1, x_best + 2):
					if i == y_best and j == x_best:
						continue
					if 0 <= i and i < self.SYNTHESIS_LENGTH and 0 <= j and j < self.SYNTHESIS_LENGTH:
						if self.final_image_array[i, j, 0] == -1 and (i, j) not in candidates:
							candidates.append((i, j))

	def fill_synthesized_image_partial(self, iterations_count):
		candidates = self.__fill_candidates_pixels()
		count = 1
		while len(candidates) > 0 and count <= iterations_count:
			neighbors_array = []
			for candidate in candidates:
				neighbors_array.append(self.__count_filled_neighbors_patch(candidate[0], candidate[1]))

			best_index = np.argmax(neighbors_array)
			y_best, x_best = candidates[best_index][0], candidates[best_index][1]

			candidates.pop(best_index)
			self.__set_patch_and_mask(y_best, x_best)

			chosen_y, chosen_x = self.__find_candidate_coordinates()
			self.final_image_array[y_best, x_best] = self.TEXTURE_ARRAY[chosen_y, chosen_x]

			for i in range(y_best - 1, y_best + 2):
				for j in range (x_best - 1, x_best + 2):
					if i == y_best and j == x_best:
						continue
					if 0 <= i and i < self.SYNTHESIS_LENGTH and 0 <= j and j < self.SYNTHESIS_LENGTH:
						if self.final_image_array[i, j, 0] == -1 and (i, j) not in candidates:
							candidates.append((i, j))

			count += 1

	def plot_image(self):
		plt.title("Image synthétisée")
		plt.imshow(self.final_image_array.astype(np.uint8))
		plt.show()