import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist


def cargar_imagen(ruta_imagen):
    return Image.open(ruta_imagen).convert('RGB')

def extraer_embeddings_de_carpeta(carpeta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 4. Transformación y modelo
    transformacion = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std =[0.229,0.224,0.225])
    ])
    modelo = models.resnet101(pretrained=True)
    modelo.fc = torch.nn.Identity()
    modelo.eval().to(device)
    embeddings, nombres = [], []
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")]
    for archivo in tqdm(archivos, desc="Procesando imágenes"):
        img = cargar_imagen(os.path.join(carpeta, archivo))
        t = transformacion(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = modelo(t).squeeze(0).cpu()
        embeddings.append(emb)
        nombres.append(archivo)
    return nombres, embeddings

# 2. Nueva función de similitud: kernel RBF sobre coseno
def calcular_similitud_rbf(embeddings,gamma=1.0,plot=False):
    """
    embeddings: list de torch.Tensor shape=(d,)
    gamma: float, ancho del kernel
    Devuelve: matriz (n, n) con similitudes en (0,1]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Apilar y normalizar
    X = torch.stack(embeddings).to(device)        # (n, d)
    X_norm = F.normalize(X, p=2, dim=1)           # filas con norma 1
    
    # Gramiano coseno
    C = X_norm @ X_norm.T                         # (n, n) en [-1,1]
    
    # RBF sobre la distancia cuadrada: 2 - 2*C
    # K_ij = exp(-gamma * ||xi-xj||^2) = exp(-2*gamma*(1-Cij))
    sim_rbf = torch.exp(-2.0 * gamma * (1.0 - C))
    sim_rbf = sim_rbf.cpu().numpy()           # (n, n) en [0,1]
    if plot:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(sim_rbf, cmap='viridis')
        plt.colorbar(im)
        plt.xticks([]); plt.yticks([])
        plt.title("Mapa de Calor de Similitud RBF")
        plt.tight_layout()
        plt.show()
    return sim_rbf

def calcular_matriz_similitud_coseno(embeddings,plot=False):
    """
    Calcula la matriz de similitud del coseno para todos los embeddings.
    """
    # Convertir todos los embeddings a un tensor para facilitar cálculos
    embeddings_tensor = torch.stack(embeddings)
    # Normalizamos los embeddings para que la similitud se base en la dirección
    embeddings_normalizados = F.normalize(embeddings_tensor, p=2, dim=1)
    
    # Multiplicación matricial para obtener la matriz de similitud del coseno
    similitud = embeddings_normalizados @ embeddings_normalizados.T
    matriz = similitud.cpu().numpy()
    if plot:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(matriz, cmap='viridis')
        plt.colorbar(im)
        plt.xticks([]); plt.yticks([])
        plt.title("Mapa de Calor de Similitud")
        plt.tight_layout()
        plt.show()
    return matriz
    
