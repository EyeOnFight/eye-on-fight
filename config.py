# Taxa de quadros desejada para o processamento (frames por segundo).
VIDEO_FRAME_RATE = 30

YOLO_POSE_MODEL_PATH = "yolov8n-pose.pt"

# Tipo de representação dos keypoints:
# "absolute" para posições absolutas ou "relative" para posições relativas.
KEYPOINTS_FEATURE_TYPE = "absolute"

# Opções: "RNN", "GRU" ou "LSTM".
RECURRENT_MODEL_TYPE = "GRU"

# Número de timesteps (recorrências) que o modelo irá considerar.
NUM_TIMESTEPS = 10  # Ajuste para avaliar diferentes quantidades de recurrences

# Tamanho do estado escondido (hidden state) da RNN.
HIDDEN_SIZE = 128  # Exemplo: 128; ajuste conforme necessário

# ===============================
# Configurações de Treinamento e Diretórios
# ===============================
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50

# Diretórios de dados e salvamento de modelos
DATASET_DIR = "dataset"
CSV_DIR = "csv"
MODEL_SAVE_PATH = "models/model.pth"

THRESHOLD = 0.5  # Defina um threshold para considerar que há briga, por exemplo, 0.5
VIDEO_PATH= "videos_teste/Fighting025_x264.mp4"  # Caminho para o vídeo de teste
