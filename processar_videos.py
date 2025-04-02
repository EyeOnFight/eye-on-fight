import cv2

# Caminho do vídeo
video_path = "video.mp4"

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  


    cv2.imshow('Frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
