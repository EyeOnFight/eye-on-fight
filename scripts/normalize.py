from importlib.resources import as_file
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Diretórios dos arquivos CSV
DATA_FOLDER = "../data/keypoints/"
OUTPUT_FOLDER = "../data/normalized/"

# Criar pasta de saída
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Inicializar o Min-Max Scaler
scaler = MinMaxScaler()

# Processar cada conjunto (train, val, test)
splits = ["train", "val", "test"]
categories = ["fight", "no_fight"]

for split in splits:
    for category in categories:
        input_path = os.path.join(DATA_FOLDER, split, category)
        output_path = os.path.join(OUTPUT_FOLDER, split, category)
        os.makedirs(output_path, exist_ok=True)

        for file in os.listdir(input_path):
            if not file.endswith(".csv"):
                continue
    
            file_path = os.path.join(input_path, file)
            df = pd.read_csv(file_path)
            
            # Separar labels e features
            labels = df["label"]
            features = df.drop(columns=["label"])
            if features.empty:
                print(f"Aviso: O arquivo {as_file} está vazio e será ignorado.")
                continue  # Pula para o próximo arquivo
            
            
            # Aplicar Min-Max Scaling
            scaled_features = scaler.fit_transform(features)
            
            # Recriar DataFrame
            df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
            df_scaled.insert(0, "label", labels.values)
            
            # Salvar novo CSV
            output_file_path = os.path.join(output_path, file)
            df_scaled.to_csv(output_file_path, index=False)
            print(f"✅ Normalizado e salvo: {output_file_path}")

print("🔥 Normalização concluída!")
