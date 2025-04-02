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
    Retorna uma lista de listas, onde cada sublista possui 34 valores (x e y de 17 keypoints).
    Caso não haja detecção, retorna lista vazia.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model(frame_rgb, verbose=False)
    all_detections = []

    if results and len(results) > 0:
        detected = results[0]
        if detected.keypoints is not None and len(detected.keypoints) > 0:
            for person in detected.keypoints:
                kp_array = person.xy.cpu().numpy()
                # Se tiver shape (1,17,2), remove a dimensão extra
                if kp_array.shape[0] == 1:
                    kp_array = kp_array[0]
                coords = []
                for point in kp_array:
                    coords.extend([float(point[0]), float(point[1])])
                # Garante que tenha 34 valores
                if len(coords) < 34:
                    coords.extend([0] * (34 - len(coords)))
                all_detections.append(coords)

    return all_detections  # pode ser lista vazia se não detectou ninguém


def draw_keypoints(frame, all_person_keypoints):
    """
    Desenha os keypoints de todas as pessoas no frame.
    all_person_keypoints: lista de listas (cada sublista com 34 valores)
    """
    for person_kp in all_person_keypoints:
        for i in range(0, len(person_kp), 2):
            x = int(person_kp[i])
            y = int(person_kp[i + 1])
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    return frame


# Importa a classe do modelo
from train_model import FightDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FightDetector(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS,
                      config.OUTPUT_SIZE, model_type=config.MODEL_TYPE)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()


def detect_and_display(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo:", video_path)
        return

    sliding_window = []
    frame_number = 0
    fight_detected = False
    fight_start_frame = None
    fight_status = "Normal"
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # Extrai os keypoints do frame
        all_keypoints = detect_keypoints(frame)
        if all_keypoints:
            keypoints = all_keypoints[0]  # Seleciona a primeira pessoa detectada
        else:
            keypoints = [0] * 34
        sliding_window.append(keypoints)
        if len(sliding_window) > config.TIMESTEPS:
            sliding_window.pop(0)

        # Se a janela estiver completa, faça a predição
        if len(sliding_window) == config.TIMESTEPS:
            sequence = np.array(sliding_window, dtype=np.float32)
            sequence = torch.tensor(sequence).unsqueeze(0).to(device)  # [1, TIMESTEPS, INPUT_SIZE]
            with torch.no_grad():
                output = model(sequence)
                pred_prob = torch.sigmoid(output)[0, -1].item()
                if pred_prob > config.DETECTION_THRESHOLD:
                    fight_status = "Luta"
                    if not fight_detected:
                        fight_detected = True
                        fight_start_frame = frame_number
                else:
                    fight_status = "Normal"

        # Desenha os keypoints de todas as pessoas no frame (para visualização)
        frame = draw_keypoints(frame, all_keypoints)
        # Sobrepõe as informações de status e frame no vídeo
        cv2.putText(frame, f"Status: {fight_status}", (20, 30), font, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {frame_number}", (20, 60), font, 1, (255, 0, 0), 2)
        if fight_detected:
            cv2.putText(frame, f"Luta iniciada no frame: {fight_start_frame}", (20, 90), font, 1, (0, 255, 255), 2)

        # Exibe o frame com as anotações
        cv2.imshow("Deteccao de Luta", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python detect_fight.py <caminho_para_video>")
    else:
        video_path = sys.argv[1]
        detect_and_display(video_path)
