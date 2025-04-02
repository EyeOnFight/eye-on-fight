# train_model.py

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config


# Dataset customizado para criar sequências temporais a partir dos CSVs
class FightDataset(Dataset):
    def __init__(self, csv_dir, timesteps):
        self.sequences = []
        self.labels = []
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Considera apenas a pessoa 0 (primeira detecção) para simplificar
            df = df[df['person_id'] == 0].sort_values('frame')
            if df.empty:
                continue
            # Colunas dos keypoints e o label
            keypoint_cols = [f"x{i}" for i in range(1, 18)] + [f"y{i}" for i in range(1, 18)]
            data = df[keypoint_cols].values
            labels = df['label'].values
            # Cria sequências usando janela deslizante
            for i in range(len(data) - timesteps + 1):
                seq = data[i:i + timesteps]
                lbl_seq = labels[i:i + timesteps]
                self.sequences.append(seq)
                self.labels.append(lbl_seq)
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Modelo de rede neural recorrente configurável
class FightDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, model_type="LSTM"):
        super(FightDetector, self).__init__()
        self.model_type = model_type.upper()
        if self.model_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Modelo inválido. Escolha entre 'RNN', 'GRU' ou 'LSTM'.")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, timesteps, input_size]
        if self.model_type == "LSTM":
            out, (hn, cn) = self.rnn(x)
        else:
            out, hn = self.rnn(x)
        # Aplica camada linear em cada timestep
        out = self.fc(out)
        return out.squeeze(-1)  # Saída com shape: [batch, timesteps]


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FightDataset(config.TRAIN_CSV_DIR, config.TIMESTEPS)

    # Divisão simples para treinamento e validação (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = FightDetector(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS,
                          config.OUTPUT_SIZE, model_type=config.MODEL_TYPE)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * sequences.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {epoch_loss:.4f}")

    # Salva o modelo treinado
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("Modelo treinado salvo em:", config.MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()
