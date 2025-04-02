# detect_fight.py

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import config

# Carrega o modelo YOLO Pose
pose_model = YOLO(config.YOLO_POSE_MODEL_PATH)


def detect_keypoints(frame):
    """
    Detecta os keypoints utilizando o YOLO Pose.
    Retorna os 34 valores (x e y para cada um dos 17 pontos) da primeira pessoa detectada.
    Caso não haja detecção, retorna uma lista com 34 zeros.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model(frame_rgb, verbose=False)
    all_detections = []
    if results and len(results) > 0:
        detected = results[0]
        if detected.keypoints is not None and len(detected.keypoints) > 0:
            for person in detected.keypoints:
                kp_array = person.xy.cpu().numpy()
                if kp_array.shape[0] == 1:
                    kp_array = kp_array[0]
                coords = []
                for point in kp_array:
                    coords.extend([float(point[0]), float(point[1])])
                if len(coords) < 34:
                    coords.extend([0] * (34 - len(coords)))
                all_detections.append(coords)
    if not all_detections:
        return [0] * 34
    return all_detections[0]  # Utiliza apenas a primeira detecção


# Importa a classe do modelo (pode ser importada do train_model ou definida aqui)
from train_model import FightDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FightDetector(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS,
                      config.OUTPUT_SIZE, model_type=config.MODEL_TYPE)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()


def detect_fight_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo:", video_path)
        return

    sliding_window = []
    frame_number = 0
    fight_detected = False
    fight_start_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # Extrai keypoints do frame
        keypoints = detect_keypoints(frame)
        sliding_window.append(keypoints)

        # Mantém o tamanho da janela igual a config.TIMESTEPS
        if len(sliding_window) > config.TIMESTEPS:
            sliding_window.pop(0)

        if len(sliding_window) == config.TIMESTEPS:
            # Prepara a sequência para o modelo
            sequence = np.array(sliding_window, dtype=np.float32)  # Shape: (TIMESTEPS, INPUT_SIZE)
            sequence = torch.tensor(sequence).unsqueeze(0).to(device)  # Shape: (1, TIMESTEPS, INPUT_SIZE)

            with torch.no_grad():
                output = model(sequence)  # Saída: (1, TIMESTEPS)
                # Considera a previsão do último frame da sequência
                pred = torch.sigmoid(output)[0, -1].item()
                if pred > config.DETECTION_THRESHOLD and not fight_detected:
                    fight_detected = True
                    fight_start_frame = frame_number
                    print(f"Luta detectada iniciando no frame: {fight_start_frame}")
                    # Se preferir, pode interromper a análise após a primeira detecção
                    # break

    if not fight_detected:
        print("Nenhuma luta detectada no vídeo.")

    cap.release()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python detect_fight.py <caminho_para_video>")
    else:
        video_path = sys.argv[1]
        detect_fight_in_video(video_path)
