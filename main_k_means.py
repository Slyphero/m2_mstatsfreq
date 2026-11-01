from k_means import *
from efros_leung import EfrosLeung

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
    "classif_data/D4.gif",
    # Textures moins homogènes
    "classif_data/D2.gif",
    "classif_data/D45.gif",
]
number_of_textures = len(TEXTURES)


if __name__ == "__main__":
    points_DO = []
    labels_DO = []
    points_DT = []
    labels_DT = []

    # Permettant de conserver les patches bruts de DT pour la synthèse
    raw_DT = {}

    for num, path in enumerate(TEXTURES):
        patches = divide_in_patches(path, PATCH_SIZE)
        np.random.shuffle(patches)
        half = len(patches) // 2

        raw_DT[num] = patches[half:]

        # Séparation des patches en deux moitiés, données entraînement et données tests
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

    # moyenne et écart type
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

    def classify_array(image, true_label):
        patches = divide_array_in_patches(image, PATCH_SIZE)
        pts = (np.array([describe(p, FILTERS) for p in patches]) - mu) / sd
        pred = np.array([mapping[k] for k in means.predict(pts)])
        return np.sum(pred == true_label), len(pred)

    SYNTH_SIZE = 64
    N_IMAGES = 3

    print(f"{'Texture':25s} normal bruit flou")
    for num in range(number_of_textures):
        ok, total = 0, 0
        ok_n, total_n = 0, 0
        ok_b, total_b = 0, 0

        # On cumule plusieurs images par texture pour éviter de se baser sur un trop faible nombre de patch
        for k in range(N_IMAGES):
            # Puisque les patches sont de taille 32x32, la mosaïque génère des images 96x96 comme
            # entrée pour Efros Leung qui synthétise dans ce contexte des images 64x64, compromis
            # entre vitesse d'exécution et résolution / taille de la synthèse
            source = mosaic(raw_DT[num][k * 9 : (k + 1) * 9])
            synth = EfrosLeung(None, SYNTH_SIZE, 9, 0.1, 15, image_array=source)
            synth.initialize_synthesized_image()
            synth.fill_synthesized_image()
            gray = synth.get_gray_image()
            Image.fromarray(gray).save(f"IMAGES_MSTATS/K_MEANS/synth_{num}_{k}.png")

            o, t = classify_array(gray, num)
            ok += o
            total += t

            o, t = classify_array(add_noise(gray, 20), num)
            ok_n += o
            total_n += t

            o, t = classify_array(blur(gray, 3), num)
            ok_b += o
            total_b += t

        print(
            f"{TEXTURES[num]:25s} {ok/total:.2f}  {ok_n/total_n:.2f}  {ok_b/total_b:.2f}"
        )

    def rate_degraded(degrade, param):
        """
        degrade : fonction de dégradation
        param : paramètre de dégradation
        """
        ok, total = 0, 0
        for num in range(number_of_textures):
            # Limitation à 20 patches par texture, itérations coûteuses
            for p in raw_DT[num][:20]:
                q = degrade(p, param)
                pt = (describe(q, FILTERS) - mu) / sd
                pred = mapping[means.predict(pt[np.newaxis, :])[0]]
                ok += pred == num
                total += 1
        return ok / total

    print("Sigma | Taux réussite bruité")
    for s in (0, 5, 10, 20, 40):
        print(f"bruit {s} {rate_degraded(add_noise, s):.2f}")

    print("Taille noyau | Taux réussite flouté")
    for k in (1, 3, 5, 7):
        print(f"flou {k} {rate_degraded(blur, k):.2f}")

    pred = np.array([mapping[k] for k in means.predict(points_DT)])
    conf = np.zeros((number_of_textures, number_of_textures), dtype=int)
    for valid, p in zip(labels_DT, pred):
        conf[valid, p] += 1

    def plot_confusion(conf):
        """
        Fonction permettant d'afficher la matrice de confusion de manière graphique
        la diagonale principale représente le nombre de bonnes prédictions
        """
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

    # Afficher les filtres utilisés
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, kernel in zip(axes.ravel(), FILTERS):
        ax.imshow(kernel, cmap="gray")
        ax.axis("off")
    plt.show()
