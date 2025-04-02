from ultralytics import YOLO
import cv2

# Carregar o modelo YOLO Pose pré-treinado
model = YOLO("yolov8n-pose.pt")  # Versão pequena do modelo (mais rápida)

# Caminho do vídeo
video_path = "video.mp4"

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar YOLO Pose no frame
    results = model(frame)

    # Exibir os keypoints detectados
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Pose', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
