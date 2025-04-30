import os
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

def yolo_prediction(df_cleaned,model_path,input,output,filter,conf_threshold=0.2):
    # ---------- Configuración ----------
    MODEL_PATH    = model_path
    input_folder  = input
    output_folder = output
    os.makedirs(output_folder, exist_ok=True)

    classes_filter = filter
    conf_threshold = conf_threshold

    # Tu DataFrame inicial con frame_id y label
    df = df_cleaned.copy()  # debe tener columnas ['frame_id','label']

    # Carga el modelo
    model = YOLO(MODEL_PATH)

    metrics = []
    # ---------- Bucle de inferencia ----------
    for image_name in tqdm(df['frame_id'], desc="Procesando frames"):
        img_path = os.path.join(input_folder, image_name)
        results  = model(img_path, classes=classes_filter,conf=conf_threshold, verbose=False)
        r        = results[0]

        # 1) Guardar imagen con predicción
        pred_img = r.plot(labels=True, boxes=True, masks=True)
        cv2.imwrite(os.path.join(output_folder, image_name), pred_img)

        # 2) Extraer métricas
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        n_dets    = len(confs)
        score_max = float(confs.max()) if n_dets > 0 else 0.0

        classes = r.boxes.cls.cpu().numpy().astype(int) if n_dets > 0 else np.array([])
        n_classes = len(np.unique(classes))

        if hasattr(r, 'masks') and r.masks is not None:
            masks      = r.masks.data.cpu().numpy()
            area_total = float(masks.sum())
        else:
            boxes      = r.boxes.xyxy.cpu().numpy() if n_dets > 0 else np.zeros((0,4))
            areas      = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
            area_total = float(areas.sum())

        metrics.append({
            'frame_id':   image_name,
            'n_dets':     n_dets,
            'n_classes':  n_classes,
            'score_max':  score_max,
            'area_total': area_total
        })

    # ---------- Unir métricas al DataFrame original ----------
    df_metrics = pd.DataFrame(metrics)
    df_final   = df.merge(df_metrics, on='frame_id', how='left')

    # ---------- Cálculo de frame_score ----------
    # 1) Lista de métricas a normalizar
    features = ['n_dets', 'n_classes', 'score_max', 'area_total']

    # 2) Normalizar cada métrica a [0,1] usando min-max global
    for f in features:
        min_v = df_final[f].min()-0.0001  # Evitar división por cero
        max_v = df_final[f].max()
        df_final[f + '_norm'] = (df_final[f] - min_v) / (max_v - min_v)

    # 3) Definir pesos (deben sumar 1)
    weights = {
        'n_dets_norm':     0.1,
        'n_classes_norm':  0.2,
        'score_max_norm':  0.65,
        'area_total_norm': 0.05
    }

    # 4) Calcular frame_score como suma ponderada
    df_final['frame_score'] = sum(df_final[f] * w for f, w in weights.items())

    # ---------- Listo! ----------
    print("✅ Procesado completado.")
    print(f"📁 Guardado en: {output_folder}")
    return df_final
