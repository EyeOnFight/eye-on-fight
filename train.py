# train.py
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Importa as configurações definidas em config.py
from config import (
    NUM_TIMESTEPS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    CSV_DIR,
    RECURRENT_MODEL_TYPE,
    MODEL_SAVE_PATH,
    KEYPOINTS_FEATURE_TYPE,
)

# ============================
# Dataset para Sequências de Keypoints
# ============================
class PoseSequenceDataset(Dataset):
    def __init__(self, csv_dir, num_timesteps=10):
        """
        csv_dir: diretório contendo os arquivos CSV.
        num_timesteps: quantidade de frames consecutivos para formar uma sequência.
        """
        self.sequences = []  # Lista de arrays com shape (num_timesteps, 34)
        self.labels = []     # Lista de labels para cada sequência

        # Lista todos os arquivos CSV no diretório
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Ordena os frames para garantir a ordem temporal
            df = df.sort_values(by="frame")
            data = df.values  # Cada linha: [frame, person_id, 34 keypoints, label]
            # Extraímos apenas os 34 valores dos keypoints, ignorando 'frame' e 'person_id'
            features = data[:, 2:-1]
            labels = data[:, -1]  # Label associado a cada frame

            # Cria janelas deslizantes de num_timesteps
            num_frames = features.shape[0]
            if num_frames >= num_timesteps:
                for i in range(num_frames - num_timesteps + 1):
                    seq = features[i : i + num_timesteps]
                    # Neste exemplo, o label da sequência é definido como o label do último frame
                    label = labels[i + num_timesteps - 1]
                    self.sequences.append(seq)
                    self.labels.append(label)

        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)



# ============================
# Modelo Recorrente Configurável (RNN, GRU ou LSTM)
# ============================
class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type="LSTM"):
        """
        input_size: dimensão de entrada (neste caso, 34, correspondendo aos keypoints).
        hidden_size: tamanho do estado escondido.
        rnn_type: tipo de rede recorrente ("RNN", "GRU" ou "LSTM").
        """
        super(RecurrentModel, self).__init__()
        self.rnn_type = rnn_type.upper()
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Tipo de RNN não suportado: {rnn_type}")

        # Camada final para classificar o evento (saída binária)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, num_timesteps, input_size)
        rnn_out, _ = self.rnn(x)
        # Utiliza a saída do último timestep
        out = rnn_out[:, -1, :]
        out = self.fc(out)
        return out

# ============================
# Loop de Treinamento
# ============================
def train():
    # Cria o dataset e o dataloader
    dataset = PoseSequenceDataset(csv_dir=CSV_DIR, num_timesteps=NUM_TIMESTEPS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Configuração do dispositivo (GPU se disponível)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instancia o modelo de acordo com a configuração
    model = RecurrentModel(input_size=34, hidden_size=HIDDEN_SIZE, rnn_type=RECURRENT_MODEL_TYPE)
    model.to(device)

    # Define o otimizador e a função de perda (BCEWithLogitsLoss é adequado para classificação binária)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Loop de treinamento
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device).unsqueeze(1)  # Ajusta a dimensão para (batch, 1)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sequences.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

    # Salva o modelo treinado
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modelo salvo em: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
