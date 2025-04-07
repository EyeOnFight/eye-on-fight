import os
import shutil
import random

# Diretórios de entrada e saída
INPUT_FOLDER = "../data/normalized/"
OUTPUT_FOLDER = "../data/mixed/"

# Criar pasta de saída
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Splits a serem processados
splits = ["train", "val", "test"]

for split in splits:
    input_folders = [
        os.path.join(INPUT_FOLDER, split, "fight"),
        os.path.join(INPUT_FOLDER, split, "no_fight"),
    ]
    output_folder = os.path.join(OUTPUT_FOLDER, split)
    os.makedirs(output_folder, exist_ok=True)

    # Coletar todos os arquivos das categorias
    all_files = []
    for folder in input_folders:
        if not os.path.exists(folder):
            print(f"Aviso: Diretório não encontrado: {folder}")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
        all_files.extend(files)

    # Embaralhar os arquivos
    random.shuffle(all_files)

    # Copiar os arquivos para a pasta de saída com nomes únicos
    for idx, file in enumerate(all_files):
        file_name = f"{idx:04d}_{os.path.basename(file)}"  # Adiciona um índice ao nome do arquivo
        shutil.copy(file, os.path.join(output_folder, file_name))

    print(f"✅ Arquivos misturados e salvos em: {output_folder}")

print("🔥 Mistura dos dados normalizados concluída!")