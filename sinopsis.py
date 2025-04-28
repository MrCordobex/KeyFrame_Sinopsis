import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2
import os
import re
import shutil

def time_series(df_final, plot_series=False, window=21):
    # 1) Copiar DataFrame y crear pivot
    df = df_final.copy()
    df['frame_idx'] = range(len(df))
    pivot = df.pivot(index='frame_idx', columns='label', values='frame_score').fillna(0)

    # 2) Suavizado de media móvil
    smoothed = pivot.rolling(window=window, center=True, min_periods=1).mean()

    # 3) Enmascarar fuera de soporte
    mask = pivot > 0
    smoothed_masked = smoothed.where(mask, 0)
    original_masked = pivot.where(mask, 0)

    # 4) Detección de clusters fuertes (gap máximo)
    max_scores    = smoothed_masked.max()
    sorted_scores = max_scores.sort_values()
    diffs         = sorted_scores.diff().iloc[1:]
    gap_idx       = diffs.values.argmax()
    low_val       = sorted_scores.iloc[gap_idx]
    high_val      = sorted_scores.iloc[gap_idx + 1]
    threshold     = (low_val + high_val) / 2

    keep_clusters = sorted_scores[sorted_scores > threshold].index.tolist()
    drop_clusters = sorted_scores[sorted_scores <= threshold].index.tolist()

    print(f"Umbral automático = {threshold:.3f}")
    print("Mantengo clusters:", keep_clusters)
    print("Descarto clusters:", drop_clusters)

    # Parámetros de find_peaks
    alpha, beta, dist = 0.5, 0.3, 20
    half_window = window // 2

    # 5) Plot opcional con detección de picos reales en original
    if plot_series:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cluster_list = keep_clusters + drop_clusters
        color_map = {cl: colors[i % len(colors)] for i, cl in enumerate(cluster_list)}

        for cl in keep_clusters:
            series_s = smoothed_masked[cl].values
            series_o = original_masked[cl].values
            idx      = smoothed_masked.index
            M        = series_s.max()
            height_thr     = alpha * M
            prominence_thr = beta  * M

            # Picos en suavizado
            peaks_s, _ = find_peaks(series_s, height=height_thr,
                                     prominence=prominence_thr, distance=dist)
            peaks_s = peaks_s[series_s[peaks_s] >= threshold]

            # Fallback: si no hay picos suavizados, usar máximo de la suavizada para referencia
            if len(peaks_s) == 0:
                peaks_s = np.array([series_s.argmax()])
                print(f"No se detectaron picos suavizados para cluster {cl}. Usando máximo en índice {peaks_s[0]}.")

            # Para cada pico suavizado, obtener pico verdadero en original
            true_peaks = []
            for p in peaks_s:
                start = max(0, p - half_window)
                end   = min(len(series_o), p + half_window + 1)
                window_vals = series_o[start:end]
                local_idx   = start + window_vals.argmax()
                true_peaks.append(local_idx)
            true_peaks = np.array(true_peaks)

            # Plot
            ax.plot(idx, series_s, '-', color=color_map[cl], label=f'Cluster {cl}')
            ax.plot(idx[peaks_s], series_s[peaks_s], 'o', color=color_map[cl], markersize=6)

        for cl in drop_clusters:
            series_s = smoothed_masked[cl].values
            idx      = smoothed_masked.index
            ax.plot(idx, series_s, '--', color=color_map[cl], label=f'Cluster {cl} (descartado)')

        ax.axhline(threshold, color='black', linestyle='-', linewidth=1,
                   label=f'Umbral = {threshold:.3f}')
        ax.set_xlabel('Frame index (orden de aparición)')
        ax.set_ylabel('Frame score (media móvil)')
        ax.set_title('Series temporales suavizadas con picos reales en original')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    return df, threshold, keep_clusters, drop_clusters, smoothed_masked, original_masked, alpha, beta, dist, half_window


def sinopsis(df_final, yolo_folder, output_folder, plot_series=False, plot_frames=True):
    # 1) Obtener series y clusters
    df, threshold, keep_clusters, drop_clusters, smoothed_masked, \
        original_masked, alpha, beta, dist, half_window = \
        time_series(df_final, plot_series)

    frames = []

    # 2) Selección de frames por cluster usando picos reales
    for cl in keep_clusters:
        series_s = smoothed_masked[cl].values
        series_o = original_masked[cl].values
        M = series_s.max()
        height_thr = alpha * M
        prominence_thr = beta * M

        peaks_s, _ = find_peaks(series_s, height=height_thr,
                                 prominence=prominence_thr, distance=dist)
        peaks_s = peaks_s[series_s[peaks_s] >= threshold]

        if len(peaks_s) == 0:
            peaks_s = np.array([series_s.argmax()])
            print(f"No se detectaron picos suavizados para cluster {cl}. Usando máximo en índice {peaks_s[0]}.")

        # True peaks en original
        true_peaks = []
        for p in peaks_s:
            start = max(0, p - half_window)
            end   = min(len(series_o), p + half_window + 1)
            window_vals = series_o[start:end]
            local_idx   = start + window_vals.argmax()
            true_peaks.append(local_idx)
        true_peaks = np.array(true_peaks)

        # Mostrar y guardar frames
        if plot_frames:
            cols = min(len(true_peaks), 5)
            rows = (len(true_peaks) + cols - 1) // cols
            plt.figure(figsize=(4 * cols, 4 * rows))

        for i, frame_idx in enumerate(true_peaks):
            frame_id = df.loc[df['frame_idx'] == frame_idx, 'frame_id'].iat[0]
            frames.append(frame_id)

            if plot_frames:
                img_path = os.path.join(yolo_folder, frame_id)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                ax = plt.subplot(rows, cols, i + 1)
                ax.imshow(img)
                ax.set_title(f"{frame_id}")
                ax.axis('off')

        if plot_frames:
            plt.suptitle(f"Cluster {cl} – {len(true_peaks)} frames seleccionados", fontsize=16)
            plt.tight_layout()
            plt.show()

    # 3) Preparar video resumen
    numero_frames  = 30
    carpeta_origen = yolo_folder
    destino        = output_folder
    os.makedirs(destino, exist_ok=True)

    def extract_number(name):
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else -1

    all_frames = sorted(df['frame_id'].tolist(), key=extract_number)

    # 4) Reconstruir picos por cluster con picos reales
    peaks_by_cluster = {}
    for cl in keep_clusters:
        series_s = smoothed_masked[cl].values
        series_o = original_masked[cl].values
        peaks_s, _ = find_peaks(series_s, height=alpha * M,
                                 prominence=beta * M, distance=dist)
        peaks_s = peaks_s[series_s[peaks_s] >= threshold]
        if len(peaks_s) == 0:
            peaks_s = np.array([series_s.argmax()])
        # True peaks
        true_peaks = []
        for p in peaks_s:
            start = max(0, p - half_window)
            end   = min(len(series_o), p + half_window + 1)
            window_vals = series_o[start:end]
            local_idx   = start + window_vals.argmax()
            true_peaks.append(local_idx)
        peaks_by_cluster[cl] = np.array(true_peaks)

    # 5) Copiar frames para video resumen
    selected_frames = []
    for peaks in peaks_by_cluster.values():
        for idx in peaks:
            fid = df.loc[df['frame_idx'] == idx, 'frame_id'].iat[0]
            pos = all_frames.index(fid)
            start = max(0, pos - numero_frames)
            end   = min(len(all_frames), pos + numero_frames + 1)
            selected_frames.extend(all_frames[start:end])

    selected_frames = list(dict.fromkeys(selected_frames))
    selected_frames.sort(key=extract_number)

    for fn in selected_frames:
        src = os.path.join(carpeta_origen, fn)
        dst = os.path.join(destino, fn)
        if os.path.exists(src):
            shutil.copy(src, dst)

    print(f"✅ Copiados {len(selected_frames)} frames (±{numero_frames}) en '{destino}'")
    return frames
