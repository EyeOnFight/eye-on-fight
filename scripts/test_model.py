import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Caminhos
X_test_path = '../data/datasets/X_test.npy'
y_test_path = '../data/datasets/y_test.npy'
model_path = 'models/rnn_fight_detection.h5'

# Carregar os dados
X_test = np.load(X_test_path, allow_pickle=True)
y_test = np.load(y_test_path, allow_pickle=True)
model = load_model(model_path)

# Descobrir parâmetros
max_timesteps = 9000  # Deve ser consistente com o modelo treinado
num_features = 68  # Garantir que bate com o modelo treinado

print(f"Esperado pelo modelo: {num_features} features por timestep")

# Verificar se os dados estão vazios
if len(X_test) == 0 or len(y_test) == 0:
    raise ValueError("Os dados de teste estão vazios. Verifique os arquivos X_test.npy e y_test.npy.")

# Filtrar apenas as amostras com o número correto de features
valid_indices = [i for i, x in enumerate(X_test) if x.shape[1] == num_features]
X_test_filtered = [X_test[i] for i in valid_indices]
y_test_filtered = y_test[valid_indices]

if len(valid_indices) < len(X_test):
    print(f"Atenção: {len(X_test) - len(valid_indices)} amostras foram removidas por não terem {num_features} features.")

print(f"\nAmostras válidas com {num_features} features: {len(valid_indices)}")
print(f"Índices removidos: {[i for i in range(len(X_test)) if i not in valid_indices]}")

# Padronizar os dados
def pad_sequences_fixed(data, max_len, num_features):
    padded_data = np.zeros((len(data), max_len, num_features))
    for i, sample in enumerate(data):
        if sample.shape[1] != num_features:
            raise ValueError(f"Sample {i} has {sample.shape[1]} features, expected {num_features}")
        padded_data[i, :sample.shape[0], :sample.shape[1]] = sample
    return padded_data

X_test_padded = pad_sequences_fixed(X_test_filtered, max_timesteps, num_features)

# Normalizar os dados (se necessário)
X_test_padded = X_test_padded / np.max(X_test_padded)

# Fazer as previsões
y_pred_probs = model.predict(X_test_padded)
y_pred = np.argmax(y_pred_probs, axis=1)

# Avaliar
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test_filtered, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test_filtered, y_pred))
print("\nAcurácia:", accuracy_score(y_test_filtered, y_pred))

# Salvar os resultados
np.save("results/y_pred.npy", y_pred)
np.save("results/y_pred_probs.npy", y_pred_probs)

with open("results/classification_report.txt", "w") as f:
    f.write("Matriz de Confusão:\n")
    f.write(str(confusion_matrix(y_test_filtered, y_pred)))
    f.write("\n\nRelatório de Classificação:\n")
    f.write(classification_report(y_test_filtered, y_pred))
    f.write("\n\nAcurácia: " + str(accuracy_score(y_test_filtered, y_pred)))

print("🔥 Resultados salvos com sucesso!")