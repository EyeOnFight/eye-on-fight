from pathlib import Path
import os

class Config:
    # Obtém o diretório base do projeto
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Diretórios de dados
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VIDEO_DIR = os.path.join(DATA_DIR, "videos")
    FIGHT_VIDEO_DIR = os.path.join(VIDEO_DIR, "fight")
    NORMAL_VIDEO_DIR = os.path.join(VIDEO_DIR, "normal")
    
    # Diretório de saída
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
    KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "keypoints")
    
    # Arquivo de anotações
    ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations", "temporal_annotation.txt")
    
    # Modelo YOLOv8
    MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-pose.pt")
    
    @classmethod
    def criar_diretorios(cls):
        """Cria todos os diretórios necessários se não existirem"""
        diretorios = [
            cls.DATA_DIR,
            cls.VIDEO_DIR,
            cls.FIGHT_VIDEO_DIR,
            cls.NORMAL_VIDEO_DIR,
            cls.OUTPUT_DIR,
            cls.CSV_DIR,
            cls.KEYPOINTS_DIR
        ]
        
        for diretorio in diretorios:
            os.makedirs(diretorio, exist_ok=True)
            
    @classmethod
    def listar_videos(cls, tipo="all"):
        """
        Lista todos os vídeos disponíveis
        :param tipo: 'all', 'fight' ou 'normal'
        :return: lista de caminhos dos vídeos
        """
        videos = []
        if tipo in ["all", "fight"]:
            videos.extend(list(Path(cls.FIGHT_VIDEO_DIR).glob("*.mp4")))
        if tipo in ["all", "normal"]:
            videos.extend(list(Path(cls.NORMAL_VIDEO_DIR).glob("*.mp4")))
        return [str(video) for video in videos]

# Criar diretórios ao importar o módulo
Config.criar_diretorios()

if __name__ == "__main__":
    # Exemplo de uso
    print("=== Diretórios do Projeto ===")
    print(f"Base: {Config.BASE_DIR}")
    print(f"Vídeos: {Config.VIDEO_DIR}")
    print(f"Saída: {Config.OUTPUT_DIR}")
    
    print("\n=== Vídeos Disponíveis ===")
    print("\nVídeos de Briga:")
    for video in Config.listar_videos("fight"):
        print(f"- {video}")
        
    print("\nVídeos Normais:")
    for video in Config.listar_videos("normal"):
        print(f"- {video}")