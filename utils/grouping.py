from utils.clustering import KMEANS,HDBSCAN
from utils.cosenoRBF import calcular_similitud_rbf, calcular_matriz_similitud_coseno,extraer_embeddings_de_carpeta,cargar_imagen
from utils.dimesionalreduction import MultiDimensionalScaling,FastMultiDimensionalScaling
import pandas as pd
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def grouping(nombre_carpeta,similitud,reducction,clustering,n_clusters=2,plot_similitud=False,plot_reduction=False,plot_clustering=False,plot_images=False):

    "Extraemos los embeddings de las imagenes"
    print("Extrayendo embeddings de las imagenes...")
    carpeta_imagenes = nombre_carpeta
    nombres, embeddings = extraer_embeddings_de_carpeta(carpeta_imagenes)


    "Calculamos la matriz de similitud"
    print("Calculando la matriz de similitud...")
    if similitud=='cosine':
        matriz_similitud = calcular_matriz_similitud_coseno(embeddings,plot=plot_similitud)
    elif similitud=='rbf':
        matriz_similitud = calcular_similitud_rbf(embeddings,gamma=2,plot=plot_similitud)
    else:
        raise ValueError("Similitud no soportada. Usa 'cosine' o 'rbf'.")
    distancias = 1-matriz_similitud**2
    
    
    
    "Reducción de la dimensionalidad"
    print("Aplicando reducción de la dimensionalidad...")
    if reducction=='MDS':
        embedding_2d=MultiDimensionalScaling(distancias,n_comp=2,plot=plot_reduction)
    elif reducction=='FastMDS':
        embedding_2d=FastMultiDimensionalScaling(distancias,n_comp=2,neigbors=1,plot=plot_reduction)
    else:
        raise ValueError("Reducción no soportada. Usa 'MDS' o 'FastMDS'.")
    
    
    "Clustering"
    print("Aplicando clustering...")
    if clustering=='KMEANS':
        labels = KMEANS(embedding_2d,n_clusters,plot=plot_clustering)
    elif clustering=='HDBSCAN':
        labels = HDBSCAN(embedding_2d,min_cluster_size=30, min_samples=15,plot=plot_clustering)
    else:
        raise ValueError("Clustering no soportado. Usa 'KMEANS' o 'HDBSCAN'.")

    "Enseñamos las imagenes"
    if plot_images:
        # Creamos un diccionario para almacenar los índices de las imágenes de cada clúster
        clusters = {}
        for lbl in np.unique(labels):
            if lbl == -1:
                continue   # opcional: si quieres saltarte el ruido
            indices = [i for i, l in enumerate(labels) if l == lbl]
            clusters[lbl] = indices

        # Diccionario para almacenar el índice representativo de cada clúster
        representative_indices = {}

        for cluster, indices in clusters.items():
            # Extraemos las coordenadas 2D de las imágenes en este clúster
            puntos_cluster = embedding_2d[indices]
            # Calculamos el centroide del clúster (promedio de los puntos)
            centroide = np.mean(puntos_cluster, axis=0)
            # Calculamos la distancia euclidiana de cada punto del clúster al centroide
            dists = np.linalg.norm(puntos_cluster - centroide, axis=1)
            # Índice relativo (dentro de "indices") del punto más cercano al centroide
            idx_min_rel = np.argmin(dists)
            # Índice global en nuestras listas
            idx_representativo = indices[idx_min_rel]
            representative_indices[cluster] = idx_representativo

            # Imprime por consola la foto representativa y los frames del clúster
            frames_cluster = [nombres[i] for i in indices]
            print(f"\nCluster {cluster}:")
            print(f"  Foto representativa: {nombres[idx_representativo]}")
            print(f"  Frames que pertenecen al clúster: {frames_cluster}")

        # ------------------
        # Graficar las imágenes representativas
        # ------------------
        n_rep = len(representative_indices)
        # Elegimos una distribución en grid: calculamos filas y columnas
        cols = int(math.ceil(np.sqrt(n_rep)))
        rows = int(math.ceil(n_rep / cols))

        plt.figure(figsize=(cols*3, rows*3))
        for i, (cluster, rep_index) in enumerate(sorted(representative_indices.items())):
            # Construir la ruta completa a la imagen
            img_path = os.path.join(carpeta_imagenes, nombres[rep_index])
            img = cargar_imagen(img_path)
            
            ax = plt.subplot(rows, cols, i+1)
            ax.imshow(img)
            ax.set_title(f"Cluster {cluster}\n{nombres[rep_index]}", fontsize=10)
            ax.axis("off")
        plt.suptitle("Imagenes Representativas por Cluster", fontsize=14)
        plt.tight_layout()
        plt.show()

    "Guardamos los resultados en un dataframe"
    df = pd.DataFrame({
    'frame_id': nombres,
    'label':    labels,
    'embedding_2d_x': embedding_2d[:, 0],
    'embedding_2d_y': embedding_2d[:, 1]
    })
    return df

