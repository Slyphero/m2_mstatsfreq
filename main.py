from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint

# Tester plusieurs valeurs de epsilon entre 0.05, 0.1, 0.2, 0.5
epsilon = 0.1

texture_img = Image.open("textures_data/text0.png")
texture_array = np.array(texture_img)
texture_width, texture_height, _ = np.shape(texture_array)

print("Sample width:", texture_width)
print("Sample height:", texture_height)

texture_text = Image.fromarray(np.uint8(texture_array))

synthesized_length = 256

# Création du moule de l'image finale synthétisée

final_img = Image.new("RGB", [synthesized_length, synthesized_length])
final_img_array = np.array(final_img)

# Générer position aléatoire pour l'emplacement de départ du sample texture 
# dans l'image finale synthétisée
seed_x = randint(0, synthesized_length - texture_width - 1)
seed_y = randint(0, synthesized_length - texture_height - 1)

# Ajout du sample texture dans l'image finale

for i in range(seed_x, seed_x + texture_width):
	for j in range(seed_y, seed_y + texture_height):
		final_img_array[i, j] = texture_array[i - seed_x, j - seed_y]

print("seed x", seed_x)
print("seed y", seed_y)

plt.figure(0)
plt.imshow(texture_array)

plt.figure(1)
plt.imshow(final_img_array)
plt.show()