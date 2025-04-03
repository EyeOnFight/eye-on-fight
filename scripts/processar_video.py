import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Diretórios
VIDEO_FOLDER = "../data/split/"  # Agora usamos as pastas split
OUTPUT_FOLDER = "../data/keypoints/"

# Criar pastas de saída para cada conjunto
splits = ["train", "val", "test"]
categories = ["fight", "no_fight"]  # Classes

for split in splits:
    for category in categories:
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, category), exist_ok=True)

# Carregar o modelo YOLO Pose
model = YOLO("yolov8n-pose.pt")

# Processar cada conjunto de vídeos
for split in splits:
    for category in categories:
        label = 1 if category == "fight" else 0  # Define a classe
        video_path = os.path.abspath(os.path.join(VIDEO_FOLDER, split, category))
        output_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, split, category))

        # Verificar se a pasta existe antes de tentar listar os arquivos
        if not os.path.exists(video_path):
            print(f"🚨 Erro: Diretório não encontrado: {video_path}")
            continue

        for video_file in os.listdir(video_path):
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue

            output_csv = os.path.join(output_path, f"{video_file.split('.')[0]}.csv")

            # Se o arquivo já existir, pula a extração para esse vídeo
            if os.path.exists(output_csv):
                print(f"⏩ Arquivo já existente, pulando: {output_csv}")
                continue

            full_video_path = os.path.join(video_path, video_file)
            cap = cv2.VideoCapture(full_video_path)

            data = []  # Lista para armazenar keypoints do vídeo

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)

                for result in results:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    
                    if len(keypoints) > 0:
                        keypoints_flat = keypoints.flatten().tolist()
                        data.append([label] + keypoints_flat)  # Coloca a label como primeira coluna

            cap.release()

            if data:
                # Encontrar o maior número de keypoints
                max_keypoints = max(len(kp) - 1 for kp in data)
                
                # Padronizar todas as linhas para o mesmo número de keypoints
                data_padded = [kp + [0] * (max_keypoints - (len(kp) - 1)) for kp in data]
                
                # Criar DataFrame com colunas dinâmicas
                columns = ["label"] + [f"k{i}" for i in range(max_keypoints)]
                df = pd.DataFrame(data_padded, columns=columns)
                
                df.to_csv(output_csv, index=False)
                
                print(f"✅ Keypoints extraídos e salvos: {output_csv} (Label: {label})")
            else:
                print(f"⚠️ Nenhum keypoint detectado para {video_file}")

print("🔥 Extração de keypoints concluída!")
