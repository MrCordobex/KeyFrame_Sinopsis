from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def MultiDimensionalScaling(distancias, n_comp=2,plot=False):
    print('Aplicando MDS...')
    # Aplicamos MDS para reducir la dimensionalidad a 2D
    mds = MDS(n_components=n_comp, max_iter=1000, eps=1e-2, n_init=1, dissimilarity="precomputed", n_jobs=-1,random_state=42)
    embedding_2d = mds.fit_transform(distancias)
    print('MDS completado.')
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=100, c='blue', alpha=0.5)
        plt.title("MDS 2D")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid()
        plt.show()

    return embedding_2d

def FastMultiDimensionalScaling(distancias, n_comp=2,neigbors=1,plot=False):
    distancias = distancias.astype('float32')
    # --- 1. Parámetros ---
    # --- 2. Selección aleatoria de landmarks ---
    n_samples = distancias.shape[0]   
    landmark_inds = np.arange(n_samples)[::2]
    m = landmark_inds.size                          # número de landmarks
    k = neigbors                              # vecinos para interpolación KNN
    # --- 3. Submatriz de distancias landmarks-landmarks ---
    D_ll = distancias[np.ix_(landmark_inds, landmark_inds)]

    # --- 4. MDS clásico sobre landmarks ---
    mds_land = MDS(n_components=n_comp, max_iter=1000, eps=1e-2, n_init=1, dissimilarity="precomputed", n_jobs=-1,random_state=42)
    X_land = mds_land.fit_transform(D_ll)   # shape (m, 2)

    # --- 5. Índices de los no‑landmarks y distancias a landmarks ---
    all_inds = np.arange(n_samples)
    non_landmark_inds = np.setdiff1d(all_inds, landmark_inds)
    D_nl = distancias[np.ix_(non_landmark_inds, landmark_inds)]  # shape (n_samples - m, m)

    # --- 6. Interpolación mediante vecinos más cercanos ---
    # Para cada punto no‑landmark, localizar sus k landmarks más cercanos:
    neighbors_idx = np.argsort(D_nl, axis=1)[:, :k]              # índices en [0..m)
    neighbors_dists = np.take_along_axis(D_nl, neighbors_idx, axis=1)

    # Pesos inversos a la distancia (evitando división por cero)
    eps = 1e-8
    weights = 1.0 / (neighbors_dists + eps)
    weights /= weights.sum(axis=1, keepdims=True)               # normalizar

    # Construir embedding para no-landmarks
    X_non = np.zeros((non_landmark_inds.size, 2), dtype=float)
    for i in range(non_landmark_inds.size):
        lm_idx = neighbors_idx[i]       # landmarks más cercanos de este punto
        X_non[i] = weights[i] @ X_land[lm_idx]

    # --- 7. Unir embeddings en orden original ---
    embedding_2d_2 = np.zeros((n_samples, 2), dtype=float)
    # landmarks en sus posiciones
    embedding_2d_2[landmark_inds] = X_land
    # no‑landmarks en sus posiciones
    embedding_2d_2[non_landmark_inds] = X_non
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(embedding_2d_2[:, 0], embedding_2d_2[:, 1], s=100, c='blue', alpha=0.5)
        plt.title("MDS 2D")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid()
        plt.show()

    return embedding_2d_2