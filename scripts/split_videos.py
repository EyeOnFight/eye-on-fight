import os
import random
import shutil

# Diretórios de entrada e saída
VIDEO_FOLDER = "../data/videos/Normal"
OUTPUT_FOLDER = "../data/split/Normal"

# Criar pastas para treino, validação e teste
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(OUTPUT_FOLDER, split), exist_ok=True)

# Listar todos os vídeos
all_videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith((".mp4", ".avi", ".mov"))]

# Embaralhar os vídeos para evitar viés na divisão
random.shuffle(all_videos)

# Definir tamanhos dos conjuntos
num_videos = len(all_videos)
train_size = int(0.7 * num_videos)
val_size = int(0.15 * num_videos)

# Dividir os vídeos
train_videos = all_videos[:train_size]
val_videos = all_videos[train_size:train_size + val_size]
test_videos = all_videos[train_size + val_size:]

# Função para copiar vídeos para os diretórios corretos
def move_videos(video_list, split_name):
    for video in video_list:
        src_path = os.path.join(VIDEO_FOLDER, video)
        dest_path = os.path.join(OUTPUT_FOLDER, split_name, video)
        shutil.copy2(src_path, dest_path)  # Copia mantendo metadados

# Mover os vídeos para as respectivas pastas
move_videos(train_videos, "train")
move_videos(val_videos, "val")
move_videos(test_videos, "test")

print("Divisão concluída! ✅")
print(f"Treino: {len(train_videos)} vídeos")
print(f"Validação: {len(val_videos)} vídeos")
print(f"Teste: {len(test_videos)} vídeos")
