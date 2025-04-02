import cv2
import pandas as pd
import os


def draw_keypoints(frame, keypoints):
    color = (0, 255, 0)  # verde
    radius = 3
    thickness = -1  # preenchido
    for i in range(0, len(keypoints), 2):
        x = int(keypoints[i])
        y = int(keypoints[i + 1])
        if x != 0 or y != 0:
            cv2.circle(frame, (x, y), radius, color, thickness)
    return frame


def visualize_csv(video_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    df = pd.read_csv(csv_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Obtém o número do frame atual
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Filtra todas as linhas correspondentes ao frame atual
        rows = df[df['frame'] == frame_num]

        # Se houver detecções para esse frame, itera sobre elas
        if not rows.empty:
            for index, row in rows.iterrows():
                # Extrai os keypoints: colunas de índice 2 até 35 (ignorando 'frame' e 'person_id')
                keypoints = row.iloc[2:36].tolist()
                frame = draw_keypoints(frame, keypoints)
                # Opcional: exibe o label e o person_id na imagem
                label = row['label']
                person_id = row['person_id']
                cv2.putText(frame, f"ID: {person_id} Label: {label}", (50, 50 + 30 * int(person_id)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Verificacao CSV", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Exemplo de uso:
if __name__ == "__main__":
    video_path = os.path.join("dataset", "Fighting042_x264.mp4")
    csv_path = os.path.join("csv", "Fighting042_x264.csv")
    visualize_csv(video_path, csv_path)
