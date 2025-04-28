import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from sklearn.cluster import KMeans

def KMEANS(embedding_2d, n_clusters,plot=False):
    print('Aplicando K-Means...')
      # El número de clústeres deseados
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding_2d)
    print('K-Means completado.')
    if plot:
        # Usamos un colormap discreto (aquí Set1 es cualitativo y tiene colores contrastados)
        custom_cmap = plt.cm.get_cmap("tab20", n_clusters)


        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                            c=labels, cmap=custom_cmap, s=80, alpha=0.8)
        plt.title("MDS + K-Means", fontsize=14)
        plt.xlabel("Dimensión 1", fontsize=12)
        plt.ylabel("Dimensión 2", fontsize=12)
        plt.grid(True)

        # En lugar de una barra de color, generamos una leyenda personalizada
        unique_labels = np.unique(labels)
        handles = []
        for ul in unique_labels:
            # Obtenemos el color asociado al cluster
            color = custom_cmap(ul / (n_clusters - 1))
            handle = plt.Line2D([], [], marker="o", linestyle="",
                                markersize=10, markerfacecolor=color,
                                label=f"Cluster {ul}")
            handles.append(handle)

        plt.legend(handles=handles, title="Clusters", loc="best", fontsize=10)
        plt.show()
    return labels



def HDBSCAN(embedding_2d,min_cluster_size=30, min_samples=15,plot=False):
    # Supongamos que embedding_2d es tu array (n_muestras x 2) con las coordenadas ya calculadas
    print("Aplicando HDBSCAN...")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(embedding_2d)
    print("HDBSCAN completado.")
    if plot:
        # Obtenemos las etiquetas únicas
        unique_labels = np.unique(labels)
        # Calculamos el número de clusters excluyendo el ruido (-1)
        n_clusters = len(unique_labels[unique_labels != -1])

        # Creamos el colormap para los clusters
        custom_cmap = plt.cm.get_cmap("tab20", n_clusters)

        # Creamos un diccionario para mapear etiquetas a colores
        color_mapping = {}
        for label in unique_labels:
            if label == -1:
                color_mapping[label] = "black"
            else:
                # Aquí usamos un mapeo discreto: suponiendo que los clusters se numeran de 0 a n_clusters-1
                # Si los labels no son consecutivos, podemos usar un índice generado a partir de la lista ordenada
                idx = np.where(np.sort(unique_labels[unique_labels != -1]) == label)[0][0]
                color_mapping[label] = custom_cmap(idx / (n_clusters - 1) if n_clusters > 1 else 0)

        # Creamos una lista con el color correspondiente para cada punto
        point_colors = [color_mapping[label] for label in labels]

        plt.figure(figsize=(8, 6))
        plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=point_colors,
            s=80,
            alpha=0.8
        )
        plt.title("MDS + HDBSCAN", fontsize=14)
        plt.xlabel("Dimensión 1", fontsize=12)
        plt.ylabel("Dimensión 2", fontsize=12)
        plt.grid(True)

        # Creamos la leyenda manualmente
        handles = []
        for label in np.sort(unique_labels):
            if label == -1:
                label_text = "Ruido/Outlier"
            else:
                label_text = f"Cluster {label}"
            
            handle = plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                                markerfacecolor=color_mapping[label],
                                label=label_text)
            handles.append(handle)

        plt.legend(handles=handles, title="Clusters", loc="best", fontsize=10)
        plt.show()
    return labels