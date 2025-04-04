import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Diretórios dos arquivos CSV
DATA_FOLDER = "../data/keypoints/"
OUTPUT_FOLDER = "../data/normalized/"

# Criar pasta de saída
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Processar cada conjunto (train, val, test)
splits = ["train", "val", "test"]
categories = ["fight", "no_fight"]

# Coletar todos os dados de treino para ajustar o scaler global
scaler = MinMaxScaler()
train_features = []
train_labels = []

for category in categories:
    input_path = os.path.join(DATA_FOLDER, "train", category)

    for file in os.listdir(input_path):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(input_path, file)
        df = pd.read_csv(file_path)

        labels = df["label"]
        features = df.drop(columns=["label"])

        if features.empty:
            print(f"Aviso: O arquivo {file_path} está vazio e será ignorado.")
            continue

        train_features.append(features)
        train_labels.append(labels)

# Concatenar todos os dados de treino e ajustar o scaler global
X_train = pd.concat(train_features, ignore_index=True)
scaler.fit(X_train)

# Função para aplicar normalização com o scaler global
def normalize_and_save(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        labels = df["label"]
        features = df.drop(columns=["label"])

        if features.empty:
            print(f"Aviso: O arquivo {file_path} está vazio e será ignorado.")
            continue

        scaled_features = scaler.transform(features)
        df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
        df_scaled.insert(0, "label", labels.values)

        output_file_path = os.path.join(output_dir, file)
        df_scaled.to_csv(output_file_path, index=False)
        print(f"✅ Normalizado e salvo: {output_file_path}")

# Aplicar normalização para todos os splits e categorias
for split in splits:
    for category in categories:
        input_path = os.path.join(DATA_FOLDER, split, category)
        output_path = os.path.join(OUTPUT_FOLDER, split, category)
        normalize_and_save(input_path, output_path)

print("🔥 Normalização global concluída!")
