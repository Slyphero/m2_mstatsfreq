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