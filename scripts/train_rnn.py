import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pickle

# Parâmetros consistentes com o pré-processamento
max_timesteps = 9000  # Número máximo de timesteps (15 minutos a 10 FPS)
num_features = 68  # 34 keypoints por pessoa * 2 pessoas

# Carregar os dados
X_train = np.load("../data/datasets/X_train.npy", allow_pickle=True)
y_train = np.load("../data/datasets/y_train.npy")
X_val = np.load("../data/datasets/X_val.npy", allow_pickle=True)
y_val = np.load("../data/datasets/y_val.npy")

# Verificar os dados carregados
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

# Transformar rótulos em categóricos (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat = to_categorical(y_val, num_classes=2)

# Criar o modelo GRU
model = Sequential([
    Masking(mask_value=0.0, input_shape=(max_timesteps, num_features)),
    GRU(32, return_sequences=False),  
    Dropout(0.4),  
    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  
    Dense(2, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
epochs = 40
batch_size = 12
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=epochs,
    batch_size=batch_size
)

# Salvar o modelo treinado
model.save("models/rnn_fight_detection.h5")

# Salvar o histórico do treinamento
with open("models/training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("🔥 Modelo treinado e salvo com sucesso!")