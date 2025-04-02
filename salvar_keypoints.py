import cv2
import numpy as np
from ultralytics import YOLO

# Carregar o modelo YOLO Pose
model = YOLO("yolov8n-pose.pt")

# Caminho do vídeo
video_path = "video.mp4"

keypoints_list = []

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar YOLO Pose
    results = model(frame)

    # Obter keypoints do primeiro frame detectado
    for result in results:
        if result.keypoints is not None:
            keypoints_list.append(result.keypoints.cpu().numpy())

cap.release()

# Converter para um array numpy e salvar
keypoints_array = np.array(keypoints_list)
np.save("keypoints.npy", keypoints_array)

print("Keypoints salvos com sucesso!")
