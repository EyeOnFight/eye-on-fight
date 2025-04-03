import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Carregar os dados
X_train = np.load("../data/datasets/X_train.npy", allow_pickle=True)
y_train = np.load("../data/datasets/y_train.npy")
X_val = np.load("../data/datasets/X_val.npy", allow_pickle=True)
y_val = np.load("../data/datasets/y_val.npy")

# Descobrir o maior número de timesteps e preencher os menores com zeros
max_timesteps = max([x.shape[0] for x in X_train])
num_features = X_train[0].shape[1]  # Número de características (keypoints)

# Função para padronizar o tamanho das sequências
def pad_sequences(data, max_timesteps, num_features):
    padded_data = np.zeros((len(data), max_timesteps, num_features))
    
    for i, sample in enumerate(data):
        print(f"Sample {i} shape: {sample.shape}")  # <-- Adiciona esse print para debug
        seq_len = min(sample.shape[0], max_timesteps)  
        feature_len = min(sample.shape[1], num_features)  
        padded_data[i, :seq_len, :feature_len] = sample[:seq_len, :feature_len]
    
    return padded_data

X_train_padded = pad_sequences(X_train, max_timesteps, num_features)
X_val_padded = pad_sequences(X_val, max_timesteps, num_features)

# Transformar rótulos em categóricos (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat = to_categorical(y_val, num_classes=2)

# Criar o modelo GRU
model = Sequential([
    Masking(mask_value=0.0, input_shape=(max_timesteps, num_features)),
    GRU(64, return_sequences=True),
    Dropout(0.3),
    GRU(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')  # Saída com duas classes (fight / no_fight)
])

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
epochs = 20
batch_size = 8
history = model.fit(
    X_train_padded, y_train_cat,
    validation_data=(X_val_padded, y_val_cat),
    epochs=epochs,
    batch_size=batch_size
)

# Salvar o modelo treinado
model.save("models/rnn_fight_detection.h5")

print("🔥 Modelo treinado e salvo com sucesso!")
