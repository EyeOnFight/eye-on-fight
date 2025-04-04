import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

# Função para normalizar os dados
def normalize_data(data):
    return [(x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) for x in data]

# Carregar os dados
X_train = np.load("data/datasets/X_train.npy", allow_pickle=True)
y_train = np.load("data/datasets/y_train.npy")
X_val = np.load("data/datasets/X_val.npy", allow_pickle=True)
y_val = np.load("data/datasets/y_val.npy")

# Normalizar os dados
X_train = normalize_data(X_train)
X_val = normalize_data(X_val)

# Padronizar o número de features (colunas)
target_features = 204
max_timesteps = max([x.shape[0] for x in X_train])  # comprimento máximo

def pad_sequences(data, max_timesteps, num_features):
    padded_data = np.zeros((len(data), max_timesteps, num_features))
    for i, sample in enumerate(data):
        print(f"Sample {i} shape: {sample.shape}")
        seq_len = min(sample.shape[0], max_timesteps)
        feat_len = min(sample.shape[1], num_features)
        padded_data[i, :seq_len, :feat_len] = sample[:seq_len, :feat_len]
    return padded_data

X_train_padded = pad_sequences(X_train, max_timesteps, target_features)
X_val_padded = pad_sequences(X_val, max_timesteps, target_features)

# Transformar rótulos em one-hot encoding
y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat = to_categorical(y_val, num_classes=2)

# Criar modelo otimizado
model = Sequential([
    Input(shape=(max_timesteps, target_features)),
    Masking(mask_value=0.0),
    GRU(128, return_sequences=True),
    Dropout(0.5),
    GRU(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# Treinamento
history = model.fit(
    X_train_padded, y_train_cat,
    validation_data=(X_val_padded, y_val_cat),
    epochs=30,
    batch_size=16,
    callbacks=[early_stopping, lr_scheduler]
)

# Avaliação final
y_pred = model.predict(X_val_padded)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_val_cat, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=['no_fight', 'fight']))

# Salvar o modelo
model.save("models/rnn_fight_detection_optimized.keras")

print("✅ Modelo otimizado treinado e salvo com sucesso!")
