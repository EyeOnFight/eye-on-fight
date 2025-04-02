# config.py
# Configurações centralizadas para o projeto de visão computacional de detecção de luta

# Caminho para o modelo YOLO Pose
YOLO_POSE_MODEL_PATH = "yolov8n-pose.pt"  # Ajuste conforme o seu ambiente

# Configurações de pré-processamento e vídeo
VIDEO_FRAME_RATE = 30  # Taxa de quadros a ser utilizada (pode ser alterada)

# Configurações do treinamento
TRAIN_CSV_DIR = "csv"  # Diretório onde os CSVs dos vídeos processados estão armazenados
MODEL_TYPE = "LSTM"    # Escolha entre "RNN", "GRU" ou "LSTM"
TIMESTEPS = 10         # Número de frames que compõem cada sequência temporal
INPUT_SIZE = 34        # Número de features (17 keypoints * 2 coordenadas)
HIDDEN_SIZE = 128      # Tamanho do estado oculto da rede recorrente
NUM_LAYERS = 2         # Número de camadas recorrentes
OUTPUT_SIZE = 1        # Saída para classificação binária (luta ou não)
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32
MODEL_SAVE_PATH = "models/fight_detector.pt"  # Caminho para salvar o modelo treinado

# Configurações de inferência
DETECTION_THRESHOLD = 0.5  # Limiar para classificar como luta
