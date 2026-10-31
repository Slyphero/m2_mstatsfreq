from k_means import *


import numpy as np
import matplotlib.pyplot as plt
import time

PATCH_SIZE = 32


FILTERS = [
    gabor(15, orientation=theta, localization=4, frequency=lam)
    for lam in (0.25, 0.5, 1.0)
    for theta in np.arange(4) * np.pi / 4
]

# On remarque que la classification manque de précision lorsqu'on l'applique sur des textures non homogènes
TEXTURES = [
    # Textures plus homogènes
    "classif_data/D1.gif",
    "classif_data/D3.gif",
    "classif_data/D4.gif",

    # Textures moins homogènes
    "classif_data/D2.gif",
    "classif_data/D30.gif",
    "classif_data/D45.gif"
]
number_of_textures = len(TEXTURES)


if __name__ == "__main__":
    points_DO = []
    labels_DO = []
    points_DT = []
    labels_DT = []

    for num, path in enumerate(TEXTURES):
        patches = divide_in_patches(path, PATCH_SIZE)
        np.random.shuffle(patches)
        half = len(patches) // 2

        for p in patches[:half]:
            points_DO.append(describe(p, FILTERS))
            labels_DO.append(num)

        for p in patches[half:]:
            points_DT.append(describe(p, FILTERS))
            labels_DT.append(num)

    points_DO = np.array(points_DO)
    points_DT = np.array(points_DT)
    labels_DO = np.array(labels_DO)
    labels_DT = np.array(labels_DT)

    mu = points_DO.mean(axis=0)
    sd = points_DO.std(axis=0) + 1e-9

    points_DO = (points_DO - mu) / sd
    points_DT = (points_DT - mu) / sd

    means = KMeans(
        points_list=points_DO,
        number_of_classes=number_of_textures,
        epsilon=0.0005,
        max_iterations=100,
    )

    means.compute()

    mapping = means.map_clusters_to_classes(labels_DO)

    def rate(points, true_labels):
        pred = np.array([mapping[k] for k in means.predict(points)])
        return np.mean(pred == true_labels)

    print("DO :", rate(points_DO, labels_DO))
    print("DT :", rate(points_DT, labels_DT))

    pred = np.array([mapping[k] for k in means.predict(points_DT)])
    conf = np.zeros((number_of_textures, number_of_textures), dtype=int)
    for valid, p in zip(labels_DT, pred):
        conf[valid, p] += 1

    def plot_confusion(conf):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(conf, cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(number_of_textures))
        ax.set_yticks(range(number_of_textures))
        ax.set_xticklabels([x.replace("classif_data/", "") for x in TEXTURES])
        ax.set_yticklabels([x.replace("classif_data/", "") for x in TEXTURES])
        ax.set_xlabel("Texture prédite")
        ax.set_ylabel("Vraie texture")
        ax.set_title("Matrice de confusion (DT)")

        threshold = conf.max() / 2
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                color = "white" if conf[i, j] > threshold else "black"
                ax.text(j, i, conf[i, j], ha="center", va="center", color=color)

        plt.tight_layout()

    plot_confusion(conf)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, kernel in zip(axes.ravel(), FILTERS):
        ax.imshow(kernel, cmap="gray")
        ax.axis("off")
    plt.show()
