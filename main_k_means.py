from k_means import *


import numpy as np
import time

PATCH_SIZE = 32


FILTERS = [
    gabor(15, orientation=theta, localization=4, frequency=lam)
    for lam in (0.25, 0.5, 1.0)
    for theta in np.arange(4) * np.pi / 4
]


TEXTURES = ["classif_data/D1.gif", "classif_data/D2.gif", "classif_data/D3.gif"]

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
        number_of_classes=len(TEXTURES),
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

    patch = patches[0]
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    axes[0, 0].imshow(patch, cmap="gray")
    axes[0, 0].set_title("patch")
    for ax, kernel in zip(axes.ravel()[1:], FILTERS):
        ax.imshow(np.abs(convolve(patch.astype(float), kernel)), cmap="gray")
        ax.axis("off")
    plt.show()
