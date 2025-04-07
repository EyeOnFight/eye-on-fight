import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU
import streamlit as st  # Usado para cache

# ============================
# Customização para carregar o modelo RNN
# ============================
class CustomGRU(GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove 'time_major' se existir
        super(CustomGRU, self).__init__(*args, **kwargs)

# Caminho e parâmetros do modelo RNN
MODEL_PATH = 'models/rnn_fight_detectionV3.h5'
MAX_TIMESTEPS = 9000  # Deve ser consistente com o modelo treinado
NUM_FEATURES = 68     # 2 pessoas x 17 keypoints x 2 coordenadas

# Carrega o modelo RNN utilizando a classe customizada para GRU
rnn_model = load_model(MODEL_PATH, custom_objects={'GRU': CustomGRU})

# ============================
# Função para carregar o modelo YOLO de forma cacheada
# ============================
@st.cache_resource
def get_pose_model():
    from ultralytics import YOLO
    # O parâmetro 'verbose=False' tenta reduzir os logs do YOLO, se suportado.
    return YOLO("yolov8n-pose.pt", verbose=False)

# ============================
# Função para converter o vídeo (10 fps e resolução 640x480)
# ============================
def convert_video(input_path, output_path, fps=10, resolution=(640, 480)):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo.")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Redimensiona o frame para a resolução desejada
        frame_resized = cv2.resize(frame, resolution)
        out.write(frame_resized)
    
    cap.release()
    out.release()
    return output_path

# ============================
# Função para extrair keypoints do vídeo usando YOLO Pose
# ============================
def extract_keypoints(video_path, target_fps=10):
    """
    Extrai os keypoints de cada frame do vídeo processado.
    Garante que para cada frame haja exatamente 2 pessoas com 17 keypoints (cada keypoint com 2 coordenadas),
    resultando em um vetor com 68 valores.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo para extração de keypoints.")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / target_fps))
    features_list = []
    frame_count = 0

    # Carrega o modelo YOLO de forma cacheada
    pose_model = get_pose_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            results = pose_model(frame, verbose=False)
            
            for result in results:
                # Se houver detecções válidas:
                if result.boxes and result.keypoints is not None and result.keypoints.xy is not None:
                    keypoints_data = result.keypoints.xy.cpu().numpy()
                    # Se forem detectadas mais de duas pessoas, seleciona as duas de maior confiança
                    if keypoints_data.shape[0] > 2:
                        if hasattr(result.keypoints, 'conf') and result.keypoints.conf is not None:
                            confidences = result.keypoints.conf.cpu().numpy()
                            sorted_indices = np.argsort(-confidences.mean(axis=1))[:2]
                            keypoints_data = keypoints_data[sorted_indices]
                        else:
                            keypoints_data = keypoints_data[:2]
                    # Se for detectada apenas uma pessoa, adiciona padding com zeros
                    elif keypoints_data.shape[0] == 1:
                        keypoints_data = np.vstack([keypoints_data, np.zeros((1, 17, 2))])
                else:
                    # Caso não haja detecções, usa zeros para 2 pessoas
                    keypoints_data = np.zeros((2, 17, 2))
                
                # Achata a matriz para um vetor (2 x 17 x 2 = 68 valores)
                keypoints_flat = keypoints_data.flatten()
                features_list.append(keypoints_flat)
        frame_count += 1

    cap.release()
    features_array = np.array(features_list)
    return features_array

# ============================
# Função para realizar padding na sequência de frames
# ============================
def pad_sequence(features, max_timesteps=MAX_TIMESTEPS, num_features=NUM_FEATURES):
    padded = np.zeros((max_timesteps, num_features))
    length = min(len(features), max_timesteps)
    padded[:length, :] = features[:length]
    return padded

# ============================
# Função que integra todas as etapas e retorna a probabilidade de luta
# ============================
def predict_fight(video_path):
    """
    Processa o vídeo (conversão, extração de keypoints, padding e normalização)
    e retorna a probabilidade de ocorrer uma luta.
    """
    # Converte o vídeo para 10 fps e resolução 640x480
    converted_video_path = "temp_converted.mp4"
    convert_video(video_path, converted_video_path, fps=10, resolution=(640,480))
    
    # Extrai os keypoints do vídeo convertido
    features = extract_keypoints(converted_video_path, target_fps=10)
    
    # Verifica se a extração obteve os 68 features por frame
    if features.shape[1] != NUM_FEATURES:
        raise ValueError(f"Extração de features falhou: esperado {NUM_FEATURES} features, mas obteve {features.shape[1]}")
    
    # Realiza o padding para garantir um input fixo para o modelo
    padded_features = pad_sequence(features, MAX_TIMESTEPS, NUM_FEATURES)
    
    # Normaliza as features (ajuste conforme o pré-processamento utilizado no treinamento)
    if np.max(padded_features) > 0:
        padded_features = padded_features / np.max(padded_features)
    
    # Expande as dimensões para criar um batch (1, timesteps, features)
    padded_features = np.expand_dims(padded_features, axis=0)
    
    # Faz a predição usando o modelo RNN
    y_pred_probs = rnn_model.predict(padded_features)
    
    # Supondo que o modelo retorne [não luta, luta]
    fight_probability = y_pred_probs[0][1] * 100  # Convertendo para porcentagem
    return fight_probability
