import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Diretórios
VIDEO_FOLDER = "../data/split/"  # Agora usamos as pastas split
OUTPUT_FOLDER = "../data/keypoints/"

# Configuração da taxa de quadros desejada (exemplo: 10 FPS)
TARGET_FPS = 10

# Criar pastas de saída para cada conjunto
splits = ["train", "val", "test"]
categories = ["fight", "no_fight"]  # Classes

for split in splits:
    for category in categories:
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, category), exist_ok=True)

# Carregar o modelo YOLO Pose atualizado
model = YOLO("yolov8s-pose.pt")

# Processar cada conjunto de vídeos
for split in splits:
    for category in categories:
        label = 1 if category == "fight" else 0  # Define a classe
        video_path = os.path.abspath(os.path.join(VIDEO_FOLDER, split, category))
        output_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, split, category))

        if not os.path.exists(video_path):
            print(f"🚨 Erro: Diretório não encontrado: {video_path}")
            continue

        for video_file in os.listdir(video_path):
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue

            output_csv = os.path.join(output_path, f"{video_file.split('.')[0]}.csv")

            if os.path.exists(output_csv):
                print(f"⏩ Arquivo já existente, pulando: {output_csv}")
                continue

            full_video_path = os.path.join(video_path, video_file)
            cap = cv2.VideoCapture(full_video_path)

            # Obter FPS do vídeo
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = max(1, int(original_fps / TARGET_FPS))  # Pular frames para atingir TARGET_FPS
            
            data = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:  # Processa apenas os frames necessários
                    results = model(frame)
                    
                    for result in results:
                        # Verifica se há detecções no frame antes de acessar keypoints
                        if not result.boxes or result.keypoints is None or result.keypoints.xy is None:
                            print(f"⚠️ Nenhuma detecção no frame {frame_count}, inserindo keypoints vazios.")
                            keypoints_data = np.zeros((2, 17, 2))  # Garante que sempre temos 2 pessoas
                        else:
                            keypoints_data = result.keypoints.xy.cpu().numpy()

                            # Verifica se existe o atributo 'conf' antes de acessá-lo
                            if hasattr(result.keypoints, 'conf') and result.keypoints.conf is not None:
                                confidences = result.keypoints.conf.cpu().numpy()
                            else:
                                confidences = np.zeros((keypoints_data.shape[0],))  # Se não houver confiança, assume 0
                            
                            # Selecionar as duas pessoas mais importantes com maior confiança
                            if len(keypoints_data) > 1:
                                sorted_indices = np.argsort(-confidences.mean(axis=1))[:2]  # Pegamos os 2 primeiros
                                keypoints_data = keypoints_data[sorted_indices]
                            
                            # Garantir que temos exatamente 2 pessoas com 17 keypoints cada
                            num_detected = keypoints_data.shape[0]
                            print(f"🔍 Frame {frame_count}: {num_detected} pessoa(s) detectada(s), shape={keypoints_data.shape}")

                            if num_detected < 2:
                                print(f"⚠️ Apenas {num_detected} pessoa(s) detectada(s) no frame {frame_count}")

                                if num_detected == 0:
                                    keypoints_data = np.zeros((2, 17, 2))  # Se nenhuma pessoa detectada, insere zeros
                                else:
                                    padding = np.zeros((1, 17, 2))  # Se apenas 1 pessoa, adiciona padding
                                    print(f"🔧 Padding criado com shape={padding.shape}")
                                    keypoints_data = np.vstack([keypoints_data, padding]) if keypoints_data.size else padding
                            
                        # Flatten e garantir que tenha 68 colunas (34 keypoints * 2 coordenadas)
                        keypoints_flat = keypoints_data.flatten().tolist()
                        data.append([label] + keypoints_flat)
                
                frame_count += 1

            cap.release()

            if data:
                num_keypoints = len(data[0]) - 1  # Exclui o label
                columns = ["label"] + [f"k{i}" for i in range(num_keypoints)]
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(output_csv, index=False)
                print(f"✅ Keypoints extraídos e salvos: {output_csv} (Label: {label})")
            else:
                print(f"⚠️ Nenhum keypoint detectado para {video_file}")

print("🔥 Extração de keypoints concluída!")
