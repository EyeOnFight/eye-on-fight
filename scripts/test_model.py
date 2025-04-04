import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ----------------------------
# Parâmetros
# ----------------------------
MAX_TIMESTEPS = 31642
NUM_FEATURES = 238  # Adaptado para o modelo treinado

# ----------------------------
# Função para extrair keypoints com MediaPipe Pose
# ----------------------------
def extract_keypoints_mediapipe(video_path, max_frames=MAX_TIMESTEPS):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)

    keypoints_list = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frame_keypoints = []
            for landmark in results.pose_landmarks.landmark[:33]:
                frame_keypoints.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])
            # Preenche com zeros extras até atingir 238 features
            while len(frame_keypoints) < NUM_FEATURES:
                frame_keypoints.append(0.0)

            keypoints_list.append(frame_keypoints)

        frame_count += 1

    cap.release()
    pose.close()
    return np.array(keypoints_list)

# ----------------------------
# Padding da sequência (mesmo do treino)
# ----------------------------
def pad_sequence(data, max_timesteps, num_features):
    padded = np.zeros((1, max_timesteps, num_features))
    seq_len = min(data.shape[0], max_timesteps)
    feat_len = min(data.shape[1], num_features)
    padded[0, :seq_len, :feat_len] = data[:seq_len, :feat_len]
    return padded

# ----------------------------
# Caminhos
# ----------------------------
video_path = "dataset/Normal_Videos_015_x264.mp4"
model_path = "models/past_models/rnn_fight_detection.h5"

# ----------------------------
# Início da inferência
# ----------------------------
print("📦 Carregando modelo treinado...")
model = load_model(model_path)

print("🎯 Extraindo keypoints com MediaPipe...")
keypoints = extract_keypoints_mediapipe(video_path)

if keypoints.shape[0] == 0:
    print("❌ Nenhum keypoint detectado no vídeo.")
    exit()

print(f"📀 Shape dos keypoints extraídos: {keypoints.shape}")
video_input = pad_sequence(keypoints, MAX_TIMESTEPS, NUM_FEATURES)

print("🤖 Fazendo predição...")
prediction = model.predict(video_input)[0]  # Extrai o array da predição

fight_prob = float(prediction[1]) * 100  # probabilidade da classe '1' (luta)
no_fight_prob = float(prediction[0]) * 100
predicted_class = int(np.argmax(prediction))

print("\n✅ Resultado da predição:")
print(f"Luta: {fight_prob:.2f}% | Sem luta: {no_fight_prob:.2f}%")
print("💥 LUTA detectada!" if predicted_class == 1 else "🕊️ Nenhuma luta detectada.")