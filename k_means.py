from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import random




class KMeans:
    def __init__(self, points_list, number_of_classes, epsilon, max_iterations):
        """
        Constructeur de la classe KMeans

        :param points_list : La liste de tous les points à classifier
        :param number_of_classes : Le nombre de classes total pour les différents
            groupes de points
        :param epsilon : Seuil de tolérance pour le critère de convergence
        :param max_iterations : Nombre maximum d'itérations de l'algorithme des k-moyennes
            au cas où le critère de convergence n'est jamais satisfait, mesure de sécurité
            pour éviter une boucle infinie.
        """
        self.POINTS_LIST = np.array(points_list)
        self.EPSILON = epsilon
        self.NUMBER_OF_CLASSES = number_of_classes
        self.MAX_ITERATIONS = max_iterations
        self.centers = np.empty((number_of_classes, self.POINTS_LIST.shape[1]))
        self.COLORS = [
            (random.random(), random.random(), random.random())
            for _ in range(self.NUMBER_OF_CLASSES)
        ]
        self.labels = []


    def compute(self):
        """
        Fonction principale du calcul des k-moyennes
        """
        number_of_points = len(self.POINTS_LIST)

        # Sélection aléatoire des centres pour l'initialisation
        indices = np.random.choice(
            number_of_points, self.NUMBER_OF_CLASSES, replace=False
        )
        self.centers = self.POINTS_LIST[indices]

        # Première itération
        distances = np.sum(
            (self.POINTS_LIST[:, np.newaxis] - self.centers) ** 2, axis=2
        )
        self.labels = np.argmin(distances, axis=1)

        self.centers = np.array(self.centers)
        self.labels = np.array(self.labels)

        self.initial_labels = self.labels.copy()
        self.initial_centers = self.centers.copy()

        # Boucle jusqu'à convergence
        # Critère de convergence : Si les barycentres à l'étape i - 1  sont suffisamment proches de ceux à l'étape i
        iterations = 0
        while True and iterations < self.MAX_ITERATIONS:
            old_centers = self.centers.copy()
            for i in range(self.NUMBER_OF_CLASSES):
                mask = self.labels == i
                if np.any(mask):
                    self.centers[i] = np.mean(self.POINTS_LIST[mask], axis=0)

            distances = np.sum(
                (self.POINTS_LIST[:, np.newaxis] - self.centers) ** 2, axis=2
            )
            self.labels = np.argmin(distances, axis=1)

            # Vérification du critère de convergence
            if np.all(np.abs(self.centers - old_centers) < self.EPSILON):
                break
            iterations += 1

    def predict(self, points):
        points = np.array(points)
        d = np.sum((points[:, np.newaxis] - self.centers) ** 2, axis=2)
        return np.argmin(d, axis=1)

    def map_clusters_to_classes(self, true_labels):
        mapping = {}
        for k in range(self.NUMBER_OF_CLASSES):
            members = true_labels[self.labels == k]
            mapping[k] = np.bincount(members).argmax() if len(members) else -1
        return mapping

    def plot_2d(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Itération 1
        ax1.scatter(
            self.POINTS_LIST[:, 0],
            self.POINTS_LIST[:, 1],
            c=[self.COLORS[label] for label in self.initial_labels],
            marker="x",
            s=10,
        )
        ax1.scatter(
            self.initial_centers[:, 0],
            self.initial_centers[:, 1],
            c="red",
            marker="o",
            s=30,
        )
        ax1.set_title("Initialisation")
        ax1.axis("equal")

        # Convergence
        ax2.scatter(
            self.POINTS_LIST[:, 0],
            self.POINTS_LIST[:, 1],
            c=[self.COLORS[label] for label in self.labels],
            marker="x",
            s=10,
        )
        ax2.scatter(self.centers[:, 0], self.centers[:, 1], c="red", marker="o", s=30)
        ax2.set_title("Convergence")
        ax2.axis("equal")

        plt.tight_layout()
        plt.show()

    def plot_3d(self):
        fig = plt.figure(figsize=(14, 6))
        # Itération 1
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(
            self.POINTS_LIST[:, 0],
            self.POINTS_LIST[:, 1],
            self.POINTS_LIST[:, 2],
            c=[self.COLORS[label] for label in self.initial_labels],
            marker="x",
            s=10,
        )
        ax1.scatter(
            self.initial_centers[:, 0],
            self.initial_centers[:, 1],
            self.initial_centers[:, 2],
            c="red",
            marker="o",
            s=40,
        )
        ax1.set_title("Initialisation")

        # Convergence
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(
            self.POINTS_LIST[:, 0],
            self.POINTS_LIST[:, 1],
            self.POINTS_LIST[:, 2],
            c=[self.COLORS[label] for label in self.labels],
            marker="x",
            s=10,
        )
        ax2.scatter(
            self.centers[:, 0],
            self.centers[:, 1],
            self.centers[:, 2],
            c="red",
            marker="o",
            s=40,
        )
        ax2.set_title("Convergence")

        plt.tight_layout()
        plt.show()


def divide_in_patches(image_path, patch_size):
    """
    Fonction de découpage de l'image passée en paramètre en patches
    de taille patch_size * patch_size

    :param image_path : Chemin de l'image qu'on souhaite découper
    :param patch_size : Taille de chaque patch
    """
    image = np.array(Image.open(image_path).convert("L"))
    height, width = image.shape
    patches = []
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patches.append(image[y : y + patch_size, x : x + patch_size])
    return patches


def gabor(filter_size, orientation, localization, frequency):
    """
    :param filter_size taille du noyau
    :param orientation (theta)
    :param localization (sigma)
    :param frequency (lambda)
    """
    radius = filter_size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]

    xr = x * np.cos(orientation) + y * np.sin(orientation)
    yr = x * np.sin(orientation) - y * np.cos(orientation)

    G = np.exp(-(xr**2 + yr**2) / (2 * localization**2)) * np.cos(
        2 * np.pi * frequency * xr / localization
    )

    return G


def convolve(patch, kernel):
    """
    :param patch patch à convoluer
    :param kernel noyau de convolution
    """
    height, width = patch.shape
    kernel_size = kernel.shape[0]
    radius = kernel_size // 2

    # Si jamais le patch dépasse des bords, on réutilse les pixels des bords
    padded = np.pad(patch, radius, mode="reflect")

    out = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            zone = padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            out[y, x] = np.sum(zone * kernel)

    return out


def describe(patch, filters):
    patch = patch.astype(float)
    features = []
    for kernel in filters:
        response = np.abs(convolve(patch, kernel))
        features += [response.mean(), response.std()]
    return np.array(features)
