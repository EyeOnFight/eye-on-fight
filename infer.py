# infer.py
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn as nn
import time

# Importa as configurações definidas em config.py
from config import VIDEO_PATH, MODEL_SAVE_PATH, NUM_TIMESTEPS, HIDDEN_SIZE, RECURRENT_MODEL_TYPE, THRESHOLD, \
    YOLO_POSE_MODEL_PATH

# Configurações do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Definição da classe do modelo recorrente
class RecurrentModel(nn.Module):
    def __init__(self, input_size=34, hidden_size=HIDDEN_SIZE, output_size=1, model_type=RECURRENT_MODEL_TYPE):
        super(RecurrentModel, self).__init__()

        self.hidden_size = hidden_size
        self.model_type = model_type

        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        output, _ = self.rnn(x)

        # Pega apenas a saída do último timestep
        output = output[:, -1, :]

        # Passa pela camada fully connected
        output = self.fc(output)
        return output


# Carrega o modelo treinado
model = RecurrentModel()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

# O resto da classe RecurrentModel permanece igual...

# Carrega o modelo YOLO Pose para extração dos keypoints
pose_model = YOLO(YOLO_POSE_MODEL_PATH)

# Definição de conexões entre keypoints para desenho do esqueleto
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Cabeça
    (5, 6),  # Ombros
    (5, 7), (7, 9), (6, 8), (8, 10),  # Braços
    (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Pernas
]


def process_video(video_path):
    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return [], [], np.array([]), []

    frames = []
    keypoints_array = []
    seq_frame_indices = []

    frame_count = 0

    # Leitura dos frames do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Armazena o frame original
        frames.append(frame)

        # Extração de keypoints usando YOLO
        results = pose_model(frame, verbose=False)

        # Inicializa array de keypoints para este frame (zerado)
        frame_keypoints = np.zeros(34)  # 17 keypoints * 2 (x, y)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # Obtém os keypoints de todas as pessoas detectadas: (N, 17, 2)
            all_keypoints_tensor = results[0].keypoints.xy
            all_keypoints = all_keypoints_tensor.cpu().numpy()  # Array com shape (N, 17, 2)

            # Opção 1: Agregar calculando a média dos keypoints de todas as pessoas
            aggregated_keypoints = np.mean(all_keypoints, axis=0)  # shape (17, 2)
            for i, kp in enumerate(aggregated_keypoints):
                frame_keypoints[i * 2] = kp[0]
                frame_keypoints[i * 2 + 1] = kp[1]

            # Opção 2 (opcional): Se desejar visualizar cada pessoa com cores diferentes,
            # você pode iterar sobre all_keypoints e chamar a função draw_keypoints para cada conjunto.

        keypoints_array.append(frame_keypoints)
        frame_count += 1

    cap.release()

    # Converte para numpy array
    keypoints_array = np.array(keypoints_array)

    # Cria sequências para o modelo recorrente
    sequences = []

    # Para cada posição possível de início de sequência
    for i in range(len(keypoints_array) - NUM_TIMESTEPS + 1):
        # Extrai uma sequência de NUM_TIMESTEPS frames consecutivos
        seq = keypoints_array[i:i + NUM_TIMESTEPS]
        sequences.append(seq)
        seq_frame_indices.append(i + NUM_TIMESTEPS - 1)  # Índice do último frame da sequência

    # Converte para numpy array com formato adequado para o modelo
    if sequences:
        sequences = np.array(sequences)
    else:
        sequences = np.array([])

    return frames, keypoints_array, sequences, seq_frame_indices


# Função para desenhar keypoints melhorada
def draw_keypoints(frame, keypoints):
    # Desenha pontos
    point_color = (0, 255, 0)  # Verde para os pontos
    point_radius = 4

    # Desenha linhas de conexão entre os keypoints
    line_color = (0, 255, 255)  # Amarelo para as linhas
    line_thickness = 2

    # Extrair coordenadas x, y dos keypoints
    coords = []
    for i in range(0, len(keypoints), 2):
        x, y = int(keypoints[i]), int(keypoints[i + 1])
        coords.append((x, y))
        if x != 0 or y != 0:
            cv2.circle(frame, (x, y), point_radius, point_color, -1)

    # Desenha conexões entre os keypoints
    for connection in POSE_CONNECTIONS:
        p1, p2 = connection
        if (coords[p1][0] != 0 or coords[p1][1] != 0) and (coords[p2][0] != 0 or coords[p2][1] != 0):
            cv2.line(frame, coords[p1], coords[p2], line_color, line_thickness)

    return frame


# Função de inferência melhorada
def run_inference(video_path, save_video=True):
    print(f"Processando o vídeo: {video_path}")
    frames, keypoints_array, sequences, seq_frame_indices = process_video(video_path)
    num_frames = len(frames)

    # Configuração para salvar o vídeo
    output_path = os.path.join(os.path.dirname(video_path),
                               f"resultado_{os.path.basename(video_path)}")

    if frames and save_video:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    else:
        out = None

    # Inicializa vetor de probabilidades
    frame_probs = np.zeros(num_frames)

    # Realiza predições para todas as sequências
    if sequences.shape[0] > 0:
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(sequences_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy().squeeze()

        # Associa cada predição ao frame correspondente
        for idx, frame_idx in enumerate(seq_frame_indices):
            frame_probs[frame_idx] = predictions[idx]

        # Suaviza as probabilidades para transições menos abruptas
        for i in range(num_frames):
            if frame_probs[i] == 0 and i > 0:
                frame_probs[i] = frame_probs[i - 1]

    paused = False
    frame_idx = 0

    while frame_idx < num_frames:
        if not paused:
            frame = frames[frame_idx].copy()
            prob = frame_probs[frame_idx]

            # Desenha barra de probabilidade
            bar_height = 20
            bar_width = int(prob * frame.shape[1])
            bar_color = (0, 0, 255) if prob > THRESHOLD else (0, 255, 0)

            # Fundo da barra (cinza)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_height), (70, 70, 70), -1)
            # Barra de probabilidade
            cv2.rectangle(frame, (0, 0), (bar_width, bar_height), bar_color, -1)

            # Texto com a probabilidade
            if prob > THRESHOLD:
                status_text = f"LUTA DETECTADA: {prob * 100:.1f}%"
                text_color = (0, 0, 255)  # Vermelho
                # Adiciona um overlay vermelho transparente
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            else:
                status_text = f"NORMAL: {prob * 100:.1f}%"
                text_color = (0, 255, 0)  # Verde

            # Desenha texto status
            cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, text_color, 2, cv2.LINE_AA)

            # Desenha keypoints
            keypoints = keypoints_array[frame_idx]
            frame = draw_keypoints(frame, keypoints)

            # Informações adicionais
            cv2.putText(frame, f"Frame: {frame_idx + 1}/{num_frames}",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            # Instruções de controle
            cv2.putText(frame, "Espaço: Pausar | Q: Sair",
                        (frame.shape[1] - 300, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Exibe o frame
            cv2.imshow("Detecção de Lutas", frame)

            # Salva o frame se necessário
            if out:
                out.write(frame)

            frame_idx += 1

        # Gerencia os controles de teclado
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # Barra de espaço
            paused = not paused
        elif key == ord('n') and paused:  # Próximo frame quando pausado
            frame_idx += 1
            if frame_idx >= num_frames:
                frame_idx = num_frames - 1
        elif key == ord('p') and paused:  # Frame anterior quando pausado
            frame_idx -= 1
            if frame_idx < 0:
                frame_idx = 0

    # Limpa recursos
    cv2.destroyAllWindows()
    if out:
        out.release()
        print(f"Vídeo salvo em: {output_path}")


# Função principal - sem alterações

if __name__ == "__main__":
    run_inference(VIDEO_PATH, save_video=True)
