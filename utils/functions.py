import numpy as np
from itertools import groupby
from operator import itemgetter

def drop_small_islands(labels, percent_threshold=0.05):
    labels = np.array(labels)
    mask = np.ones_like(labels, dtype=bool)
    
    # Encuentra los grupos consecutivos
    groups = []
    for k, g in groupby(enumerate(labels), key=itemgetter(1)):
        g = list(g)
        start = g[0][0]
        end = g[-1][0]
        groups.append((start, end, k))
    
    for i in range(1, len(groups) - 1):
        prev_start, prev_end, _ = groups[i - 1]
        curr_start, curr_end, _ = groups[i]
        next_start, next_end, _ = groups[i + 1]
        
        len_prev = prev_end - prev_start + 1
        len_next = next_end - next_start + 1
        len_curr = curr_end - curr_start + 1
        min_len_side = min(len_prev, len_next)
        
        # Si el grupo actual es significativamente más pequeño que sus vecinos
        if len_curr < percent_threshold * min_len_side:
            mask[curr_start:curr_end+1] = False  # marcar para eliminar

    return mask

def clean_noise(df,island=True, noise_threshold=0.1):
    """
    Elimina el ruido de un DataFrame basado en un umbral de ruido.
    
    Args:
        df (pd.DataFrame): DataFrame con una columna 'label'.
        noise_threshold (float): Umbral para considerar un grupo como ruido.
        
    Returns:
        pd.DataFrame: DataFrame sin ruido.
    """
    df_filtrado = df[df['label'] != -1]
    if island:
        # Eliminar grupos de ruido
        mask = drop_small_islands(df_filtrado['label'], percent_threshold=noise_threshold)
        df_cleaned = df_filtrado[mask].reset_index(drop=True)
    else:
        df_cleaned = df_filtrado.copy()
    
    return df_cleaned

