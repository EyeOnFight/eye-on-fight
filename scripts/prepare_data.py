import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Diretório base dos CSVs normalizados
base_dir = "../data/normalized/"

# Estrutura das pastas
splits = ["train", "val", "test"]
classes = {"fight": 1, "no_fight": 0}

X = []  # Lista para armazenar os keypoints
y = []  # Lista para armazenar os rótulos

# Carregar os dados
for split in splits:
    for category, label in classes.items():
        folder_path = os.path.join(base_dir, split, category)
        
        if not os.path.exists(folder_path):
            print(f"Aviso: Diretório não encontrado: {folder_path}")
            continue
        
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path)
            
            # Separar rótulos e características
            features = df.iloc[:, 1:].values  # Demais colunas (valores numéricos)
            X.append(features)
            y.append(label)

# Converter listas para arrays numpy
X = np.array(X, dtype=object)  # Como os vídeos têm tamanhos diferentes, usamos dtype=object
y = np.array(y, dtype=int)

# Dividir os dados em treino (70%), validação (15%) e teste (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Conjuntos de dados preparados:")
print(f"Treino: {len(X_train)} vídeos")
print(f"Validação: {len(X_val)} vídeos")
print(f"Teste: {len(X_test)} vídeos")

# Criar diretório de saída
output_dir = "datasets/"
os.makedirs(output_dir, exist_ok=True)

# Salvar os datasets em arquivos numpy
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

print("🔥 Dados salvos como arrays numpy!")
