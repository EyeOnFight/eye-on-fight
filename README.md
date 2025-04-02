Perfeito! Vou montar um pipeline completo de classificação de eventos com YOLOPose e redes neurais recorrentes, usando os 50 vídeos de lutas e 50 vídeos normais, com foco na precisão e boas práticas.

Incluirei o uso de PyTorch, pré-processamento eficiente dos vídeos, extração de poses com YOLOPose, sequência temporal com RNN (LSTM ou GRU), além de tentar detectar os momentos de início/fim das lutas. Também considerarei o uso de recursos como optical flow ou atenção para melhorar a performance temporal do modelo.

Já começo a trabalhar nisso e te trago em breve o plano técnico completo com código sugerido e recomendações.

# Pipeline de Classificação de Lutas em Vídeo com YOLOPose e RNN

**Objetivo:** Desenvolver um pipeline robusto para classificar automaticamente vídeos como contendo **luta** ou **não-luta**. O método combina a **detecção de poses humanas** (usando YOLOPose) em cada frame e uma **rede neural recorrente (RNN)** para analisar a sequência temporal dessas poses. A seguir apresentamos um plano técnico detalhado, com explicações de cada etapa, trechos de código estruturado em PyTorch e recomendações para obter alta precisão, mantendo boas práticas de engenharia de software.  

Este pipeline baseia-se na ideia de que **padrões de movimento humano** capturados pelas poses corporais podem distinguir lutas de atividades normais. Abordagens similares na literatura demonstraram sucesso – por exemplo, combinar a estimação de pose (OpenPose) com LSTM alcançou **95% de acurácia** em detecção de lutas em vídeos de filmes ([GitHub - jpowellgz/FightDetectionPoseLSTM: Fight detection using Open Pose and Bi-LSTM](https://github.com/jpowellgz/FightDetectionPoseLSTM#:~:text=videos%20containing%20fights,proposed%20has%20a%20similar%20performance)). Optamos pelo YOLOPose devido à sua eficiência em estimar múltiplas poses humanas em uma única etapa ([](https://arxiv.org/pdf/2204.06806#:~:text=bounding%20boxes%20for%20multiple%20persons,since%20all%20persons%20are%20localized)), alcançando desempenho de estado-da-arte na tarefa de pose humana ([](https://arxiv.org/pdf/2204.06806#:~:text=achieves%20new%20state,Our%20training%20codes%20will%20be)). A implementação será em PyTorch, aproveitando a GPU NVIDIA GTX 1650 para acelerar tanto a detecção de poses quanto o treinamento do modelo RNN.

## 1. Pré-processamento dos Vídeos

Para preparar os dados, realizamos **amostragem de frames** de cada vídeo e normalizamos as imagens, garantindo consistência na entrada do modelo de pose. Também estruturamos o código de forma modular para facilitar manutenção e reutilização. As etapas de pré-processamento incluem:

- **Leitura e Decodificação:** Carregar cada vídeo a partir das pastas `luta/` e `normal/`. Podemos usar bibliotecas como OpenCV (`cv2.VideoCapture`) ou decord para iterar sobre os frames.
- **Amostragem de Frames:** Para reduzir redundância e carga computacional, extrair frames em intervalos regulares. Por exemplo, em um vídeo de ~30 FPS, podemos capturar 5 FPS (um frame a cada ~6 quadros) ou outro valor adequado. Assim, de um vídeo de 60s obteríamos ~300 frames. Isso preserva informação de movimento suficiente para detectar lutas, com menos frames para processar.
- **Redimensionamento:** Padronizar o tamanho dos frames. Se os vídeos já estão em 320x240, podemos manter essa resolução. Contudo, **modelos YOLO geralmente são treinados em resoluções maiores (como 640x640)**, então pode ser benéfico **redimensionar ou fazer padding** dos frames para essa resolução antes da detecção de pose, a fim de melhorar a acurácia da pose (mantendo a proporção para não distorcer pessoas).
- **Normalização de Pixel:** Converter os frames para um formato apropriado à rede YOLOPose – por exemplo, BGR para RGB (se necessário) e escala de valores de pixel [0,1] ou [0,255] conforme o modelo exige. A maioria dos modelos pré-treinados em PyTorch já lida com normalização internamente ou espera valores 0-255 normalizados internamente, então basta garantir o tipo correto (uint8 ou float tensor).
- **Estrutura de Dados:** Armazenar os frames amostrados de cada vídeo em uma lista ou array. Opcionalmente, salvar esses frames em disco (por exemplo, em um diretório temporário) para facilitar reuso sem precisar decodificar o vídeo novamente.

**Boas práticas:** encapsular essa lógica de pré-processamento em funções bem definidas. Por exemplo, podemos criar uma função `extract_frames(video_path, target_fps)` que retorna a lista de frames amostrados. Também podemos definir uma classe PyTorch `VideoDataset` para representar o dataset de vídeos, abstraindo leitura, amostragem e associação com o rótulo (luta ou não-luta). Abaixo um exemplo simplificado de extração de frames usando OpenCV:

```python
import cv2

def extract_frames(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(max(1, orig_fps // target_fps))  # pula frames para atingir FPS desejado
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # fim do vídeo
        if frame_idx % frame_interval == 0:
            frame_resized = cv2.resize(frame, (320, 240))  # redimensiona se necessário
            frames.append(frame_resized)
        frame_idx += 1
    cap.release()
    return frames

# Exemplo de uso:
video_frames = extract_frames("videos/luta/video1.mp4", target_fps=5)
print(f"Frames extra\u00eddos: {len(video_frames)}")
```

*Explicação:* No código acima, `frame_interval` determina a taxa de amostragem baseada no FPS original. Redimensionamos para 320x240 (adequado ao dataset dado) – ajuste esta parte se optar por aumentar a resolução para o modelo de pose. Ao final, teremos uma lista de numpy arrays (`frames`), cada um correspondendo a um frame selecionado.

## 2. Detecção de Poses (YOLOPose) por Frame

Com os frames amostrados, passamos para a **detecção de poses humanas em cada frame** usando YOLOPose. O YOLOPose é uma variação do algoritmo YOLO adaptado para estimar pontos-chave do corpo (esqueletos humanos) além de detecção de pessoas. Usar um modelo pré-treinado (por exemplo, treinado no COCO Keypoints) é essencial, dado o tamanho do nosso dataset, para obter boas detecções sem precisar treinar do zero. Escolhas possíveis incluem: **YOLOv7 Pose** (disponível no repositório oficial YOLOv7) ou **YOLOv8 Pose** (da Ultralytics). Ambos seguem a abordagem YOLO-Pose, que une detecção e estimação de pose em uma única passada de rede ([](https://arxiv.org/pdf/2204.06806#:~:text=bounding%20boxes%20for%20multiple%20persons,since%20all%20persons%20are%20localized)). 

**Configuração do modelo:** Utilizaremos PyTorch. No caso do YOLOv8 Pose, a Ultralytics fornece uma API simples. Por exemplo:

```python
from ultralytics import YOLO

# Carrega o modelo YOLOPose (a variante 'n' = nano ou 's' = small é leve o bastante para GTX 1650)
pose_model = YOLO("yolov8n-pose.pt")  # modelo pré-treinado de pose

# Realiza previsão em um frame (numpy array ou caminho de imagem)
frame = video_frames[0]  # um frame da etapa anterior
results = pose_model(frame)  # inferência de pose
```

Após a inferência, precisamos extrair os **keypoints** (pontos do esqueleto) detectados. Cada pessoa no frame terá um conjunto de coordenadas (x,y) para pontos como cabeça, ombros, cotovelos, joelhos, etc. Supondo que o modelo retorne os resultados em `results[0]`, podemos acessar os keypoints assim (utilizando a API Ultralytics YOLO):

```python
res = results[0]            # resultado para a imagem analisada
keypoints_data = res.keypoints  # tensor com coordenadas dos pontos (e possivelmente confidências)
```

Aqui, `keypoints_data` pode ser um tensor de dimensão `(N, K, 3)` onde N = número de pessoas detectadas no frame, K = número de pontos por pessoa (por ex., 17 pontos do COCO) e cada ponto tem 3 valores (x, y, confiança). Vamos então **formatar esses dados em um vetor fixo de características** representando o frame:

- Para cada frame, queremos uma representação vetorial de tamanho constante, independentemente de quantas pessoas estejam presentes. Como pressuposto, lutas envolvem **no máximo duas pessoas principais** interagindo. Portanto, podemos fixar em considerar até 2 pessoas por frame. Se o modelo detectar mais, ignoramos as adicionais (por exemplo, considerando as 2 de maior confiança ou área de bounding box). Se detectar menos (ex: apenas 1 pessoa), usamos um preenchimento para a segunda pessoa (por exemplo, zeros).
- Concatenamos os keypoints das até 2 pessoas. Cada pessoa fornece K pontos * 2 coordenadas = `2*K` valores (ignorando a confiança por simplicidade, ou poderíamos incluir também). Com K=17, isso dá 34 valores por pessoa. Para 2 pessoas: 68 valores por frame.
- Também normalizamos as coordenadas dos pontos em relação ao tamanho do frame (largura=320, altura=240), escalando x e y para [0,1]. Isso torna as características invariantes à escala do vídeo.
- Opcionalmente, poderíamos incluir características adicionais, como a confiança de cada ponto ou a bounding box da pessoa, mas inicialmente manteremos apenas as coordenadas normalizadas para simplificar.

Implementamos a formatação acima:

```python
import numpy as np

def frame_to_pose_features(frame, model):
    """Detecta poses no frame e retorna um vetor de caracter\u00edsticas fixo representando at\u00e9 2 poses."""
    results = model(frame)
    res = results[0]
    if res.keypoints is None:
        # Nenhuma pessoa detectada: retorna vetor zero
        return np.zeros(2 * 17 * 2, dtype=np.float32)
    # Converte tensor de keypoints para numpy
    keypoints = res.keypoints.numpy()  # shape: (N, 17, 3) se houver detec\u00e7\u00f5es
    # Ordena pessoas pela confian\u00e7a m\u00e9dia dos pontos (poderia ser pelo score da detec\u00e7\u00e3o)
    avg_conf = keypoints[:, :, 2].mean(axis=1)  # confian\u00e7a m\u00e9dia de cada pessoa
    order = np.argsort(avg_conf)[::-1]         # ordena do mais confiante para o menos
    keypoints = keypoints[order]
    # Garante 2 pessoas (pad com zeros se precisar)
    if keypoints.shape[0] < 2:
        # empilha uma pessoa "vazia" de 17 pontos (zeros)
        missing = np.zeros((1, keypoints.shape[1], keypoints.shape[2]))
        keypoints = np.concatenate([keypoints, missing], axis=0)
    # Seleciona apenas 2 pessoas
    keypoints = keypoints[:2, :, :]
    # Extrai coordenadas (ignora a terceira coluna de confian\u00e7a)
    coords = keypoints[:, :, :2]  # shape: (2, 17, 2)
    # Normaliza coordenadas x,y pelo tamanho do frame
    h, w = frame.shape[0], frame.shape[1]
    coords[:, :, 0] /= w  # x / largura
    coords[:, :, 1] /= h  # y / altura
    # Achata em um vetor 1D
    feature_vec = coords.flatten()  # shape resultante: 2*17*2 = 68
    return feature_vec.astype(np.float32)

# Exemplo de aplica\u00e7\u00e3o em todos os frames de um v\u00eddeo
video_features = []
for frame in video_frames:
    feat = frame_to_pose_features(frame, pose_model)
    video_features.append(feat)
video_features = np.stack(video_features)  # shape: (num_frames, 68)
```

No código acima, cuidamos de ordenar as detecções por confiança e garantir exatamente duas pessoas. O vetor final `feature_vec` contém `[x1,y1, x2,y2, ..., x17,y17]` da pessoa1, seguido dos mesmos 34 valores para pessoa2. Se uma pessoa faltava, esse segmento será zero, o que indica ausência de segunda pessoa naquele frame.

**Desempenho:** Para processar todos os frames dos 100 vídeos, o passo de pose é o mais custoso. Algumas dicas para otimizar no ambiente GTX 1650:

- **Batching de frames:** Em vez de inferir frame a frame, podemos passar um *batch* de frames para o modelo de uma vez, se houver memória. Por exemplo, `results = pose_model(list_of_frames)`, a API da Ultralytics permite uma lista/batch. Isso utiliza melhor a GPU paralelizando inferências. Ajuste o tamanho de lote conforme a VRAM (p.ex., batches de 16 frames).
- **Modelo leve:** Usar um modelo YOLOPose leve (como `yolov8n-pose.pt` ou YOLOv7-tiny-pose) para maior velocidade, visto que a GTX 1650 tem ~4GB VRAM. Modelos maiores aumentam a precisão, mas podem ser lentos ou não caber.
- **Processamento offline:** Uma vez extraídas as características de pose para cada frame, **salve os resultados** (por exemplo, em arquivos NumPy `.npy` ou pickle). Assim, não é necessário rodar o YOLOPose a cada vez que treinarmos ou ajustarmos o classificador – usamos os dados de pose pré-computados, economizando tempo durante experimentos.

## 3. Formação de Sequências Temporais de Poses

Com cada frame convertido em um vetor de características de pose, organizamos a sequência temporal para alimentar a RNN. Cada vídeo agora será representado por uma sequência de vetores dimensionados, e o rótulo associado (luta=1, não-luta=0). Aspectos importantes:

- **Estrutura da sequência:** Podemos representar a sequência como um tensor de forma `(T, D)`, onde `T` é o número de frames amostrados do vídeo e `D` = 68 (dimensão do vetor de pose por frame, conforme definido). No exemplo acima, `video_features` já é um array `(num_frames, 68)` para um vídeo.
- **Comprimento fixo ou variável:** Vídeos podem ter número de frames ligeiramente diferente dependendo da duração ou taxa de quadros. As RNNs em PyTorch podem lidar com sequências de comprimento variável (via `pack_padded_sequence`), mas uma abordagem simples é **padronizar** o comprimento:
  - Encontrar o número máximo de frames nos vídeos (digamos, ~300 no nosso caso) e então *padding* (preencher) sequências mais curtas com zeros até esse comprimento. Alternativamente, definir um comprimento fixo (por ex., 300) e truncar sequências mais longas ou preencher as mais curtas. Padding com vetor zero é adequado aqui (nenhuma pose = zeros).
  - Armazenar também uma máscara ou comprimento real para cada sequência, se quisermos evitar que o modelo considere os frames padding como informação (embora zeros de pose signifiquem "ninguém presente", o que por si só indica não-luta naquele instante).
- **Preparação do dataset para treinamento:** Podemos implementar a classe `VideoPoseDataset` que carrega as sequências de pose e rótulos. Essa classe pode ler os arquivos `.npy` salvos para cada vídeo (um arquivo por vídeo, contendo a sequência de poses) e retornar o tensor de sequência e o label. Utilizar o DataLoader do PyTorch para fornecer minibatches de sequências embaralhadas durante o treino.

Exemplo conceitual de montagem do dataset (assumindo que salvamos as features de cada vídeo em arquivos separados):

```python
import torch
from torch.utils.data import Dataset, DataLoader

class VideoPoseDataset(Dataset):
    def __init__(self, features_dir):
        self.samples = []  # lista de (feature_tensor, label)
        # Percorre subpastas luta/normal no diret\u00f3rio
        for label_dir in ["luta", "normal"]:
            label = 1 if label_dir == "luta" else 0
            dir_path = f"{features_dir}/{label_dir}"
            for file in os.listdir(dir_path):
                if file.endswith(".npy"):
                    seq = np.load(os.path.join(dir_path, file))  # carrega seq de pose
                    # Converte para tensor PyTorch
                    seq_tensor = torch.tensor(seq, dtype=torch.float32)
                    # Opcional: padding/truncamento para comprimento fixo, se necess\u00e1rio
                    # Exemplo: fixar 300 frames
                    T, D = seq_tensor.shape
                    if T < 300:
                        pad = torch.zeros(300 - T, D)
                        seq_tensor = torch.cat([seq_tensor, pad], dim=0)
                    else:
                        seq_tensor = seq_tensor[:300]
                    self.samples.append((seq_tensor, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# Instancia dataset e dataloader
train_dataset = VideoPoseDataset("features_treinamento")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

No código acima, preenchemos/truncamos as sequências para 300 frames. Em produção, uma estratégia mais elegante é calcular o comprimento máximo real e evitar truncar lutas longas; mas no nosso dataset cada vídeo tem ~1min, então 300 frames a 5fps é suficiente. Note que mantivemos o *batch_size* relativamente pequeno (8) para caber na GPU, dado que cada sequência é 300x68 e temos duas camadas (batch de sequências grandes consome memória).

**Observação:** Ao estruturar os dados assim, se preferirmos não fixar o comprimento, podemos usar `collate_fn` no DataLoader para empacotar sequências de comprimentos variados. Mas para simplicidade e compatibilidade geral, o padding fixo funciona bem.

## 4. Treinamento da Rede Neural Recorrente (LSTM/GRU)

Agora definimos e treinamos a rede recorrente que irá **classificar cada sequência de poses** como luta ou não-luta. Vamos optar por uma arquitetura LSTM (Long Short-Term Memory), mas uma GRU (Gated Recurrent Unit) poderia ser usada de forma semelhante – ambas capturam dependências temporais. A rede tomará a sequência de vetores de pose de um vídeo e produzirá uma predição binária. 

**Arquitetura proposta:**

- **Entrada:** sequências de tamanho T x D (por exemplo, 300 x 68) por vídeo.
- **Camada recorrente:** 1 ou 2 camadas LSTM com, digamos, 128 unidades ocultas cada. Usamos `batch_first=True` no PyTorch LSTM para aceitar input shape (batch, seq, features). Podemos também experimentar uma LSTM bidirecional (`bidirectional=True`) para que a rede considere a sequência em ambas direções temporais – útil se quisermos contexto do futuro, mas como estamos classificando o vídeo inteiro (que já está completo), o bidirecional pode melhorar a captura de padrões completos da sequência.
- **Camada fully-connected de saída:** pega o último estado oculto da LSTM (ou a concatenação dos últimos estados das direções, se bidirecional) e passa por uma camada linear para produzir um logit (um valor real). Este logit é então passado por uma função sigmóide para obter probabilidade de luta.
- **Função de perda:** usamos Binary Cross-Entropy (BCE) com logits (ou seja, `nn.BCEWithLogitsLoss` no PyTorch), apropriada para classificação binária.
- **Otimização:** um otimizador como Adam ou SGD. Adam geralmente converge mais rápido em redes recorrentes.
- **Épocas:** dado o dataset pequeno (100 vídeos), podemos treinar por, por exemplo, 50 epochs e monitorar a validação para evitar overfitting. Utilizar **Early Stopping** se a perda de validação não melhorar após certa quantidade de épocas.
- **Divisão de dados:** separar um conjunto de treino (por ex. 80 vídeos) e validação/teste (20 vídeos) estratificados entre lutas e não-lutas, para medir desempenho em dados não vistos. Poderíamos também usar validação cruzada k-fold devido ao dataset reduzido, mas isso multiplicaria o tempo de treinamento.

Abaixo definimos a classe do modelo e o laço de treinamento em PyTorch:

```python
import torch.nn as nn
import torch.optim as optim

class FightClassifier(nn.Module):
    def __init__(self, input_size=68, hidden_size=128, num_layers=1, bidirectional=False):
        super(FightClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                             bidirectional=bidirectional)
        # Dimens\u00e3o de saida da LSTM = hidden_size * (2 se bidirecional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)
    def forward(self, x):
        # x: tensor de shape (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)         # out: (batch, seq_len, hidden*direc)
        # Pega o \u00faltimo estado oculto da \u00faltima camada
        if self.bidirectional:
            # concatenar \u00faltimos estados das duas dire\u00e7\u00f5es
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)  # hn shape: (num_layers*direc, batch, hidden)
        else:
            last_hidden = hn[-1]  # shape: (batch, hidden)
        logit = self.fc(last_hidden)        # shape: (batch, 1)
        return logit

# Inicializa modelo, perda e otimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FightClassifier(input_size=68, hidden_size=128, num_layers=2, bidirectional=True).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

Aqui definimos uma LSTM de 2 camadas bidirecionais com 128 unidades (essas hiperparâmetros podem ser ajustados). O método `forward` retorna diretamente o *logit* bruto; usaremos `sigmoid` depois para obter probabilidade. Note que usamos o último estado oculto (`hn`) da LSTM em vez da saída `out` completo; isso é comum em classificação de sequência inteira, pois queremos a representação após ler todo o vídeo. Em caso bidirecional, `hn` terá 2x num_layers estados (um por direção); concatenamos as duas últimas (uma de cada direção) para obter um vetor combinado.

**Laço de treinamento:** 

```python
# Supondo que temos train_loader e val_loader definidos para treino e valida\u00e7\u00e3o
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for seq_batch, label_batch in train_loader:  # cada seq_batch: shape (batch, T, 68)
        seq_batch = seq_batch.to(device)
        label_batch = label_batch.to(device).float().unsqueeze(1)  # shape (batch, 1)
        optimizer.zero_grad()
        logits = model(seq_batch)               # forward
        loss = criterion(logits, label_batch)   # calcula perda BCE
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    
    # Avalia em valida\u00e7\u00e3o a cada \u00e9poca
    model.eval()
    correct = 0
    preds_list, targets_list = [], []
    with torch.no_grad():
        for seq_batch, label_batch in val_loader:
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device).float().unsqueeze(1)
            logits = model(seq_batch)
            # usa sigm\u00f3ide para probabilidade e converte para 0/1
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            correct += (preds == label_batch.long()).sum().item()
            preds_list.extend(probs.cpu().numpy().flatten().tolist())
            targets_list.extend(label_batch.cpu().numpy().flatten().tolist())
    val_acc = correct / len(val_loader.dataset)
    # (Poder\u00edamos calcular F1 e AUC aqui tamb\u00e9m – ver pr\u00f3ximo t\u00f3pico)
    print(f"Epoch {epoch+1}: Loss treino = {avg_loss:.4f}, Acc val = {val_acc:.3f}")
```

No loop acima, treinamos o modelo e depois calculamos a acurácia no conjunto de validação (convertendo logits em classes binárias). Também coletamos as probabilidades e rótulos verdadeiros em `preds_list` e `targets_list` para calcular outras métricas depois (como F1 e AUC). Note a conversão de `label_batch` para *float* e depois *unsqueeze(1)* para bater a dimensão do logit `(batch,1)` esperado pelo BCE. Usamos `.to(device)` para garantir que tanto dados quanto modelo estejam na GPU, tirando proveito da GTX 1650.

**Boas práticas de engenharia no treinamento:**

- Estruturar o código de treinamento possivelmente em uma função ou script separado, mantendo separação clara entre preparação de dados, definição de modelo e treinamento/validação.
- Usar logs (print ou logging) para acompanhar a perda e métricas por época. Isso ajuda a depurar e verificar a convergência.
- Salvar checkpoints do modelo (por ex., usando `torch.save`) para poder recuperar o melhor modelo encontrado (com maior acurácia em validação, por exemplo).
- Monitorar *overfitting*: se a acurácia de treinamento sobe mas a de validação piora, aplicar técnicas como Early Stopping ou regularização (dropout na LSTM, L2 penalty, etc.). No modelo acima, poderíamos adicionar `self.lstm = nn.LSTM(..., dropout=0.5)` para dropout entre camadas LSTM se usamos `num_layers > 1`.

## 5. Identificação de Frames de Luta (Detecção Temporal)

Além de classificar o vídeo globalmente, um recurso desejável é **identificar quando, dentro do vídeo, ocorrem as lutas** – ou seja, quais frames ou intervalos temporais correspondem a comportamento violento. Essa detecção temporal é opcional e desafiadora, pois nosso conjunto de dados não fornece rótulos frame-a-frame, apenas por vídeo. No entanto, podemos abordar de algumas maneiras:

**5.1. Saída por frame via RNN:** Uma abordagem é modificar a arquitetura para produzir uma decisão em cada time step em vez de apenas no final. Por exemplo, usar a saída `out` da LSTM para gerar probabilidades em cada frame, adicionando uma camada linear que mapeia o vetor oculto de cada frame para um escore de luta. Isso é similar a um modelo many-to-many (sequência para sequência) mas treinado com um rótulo global. Sem rótulos por frame, podemos **assumir que todos os frames de vídeos rotulados como luta contêm alguma luta**, o que não é estritamente verdade se a briga ocorre apenas em parte do vídeo, mas pode servir de aproximação. Uma técnica melhor é **utilizar atenção temporal**: o modelo foca automaticamente nos frames importantes para a decisão global de luta. Por exemplo, um mecanismo de atenção na saída da LSTM poderia ponderar quais frames contribuíram mais para classificar como luta, dando indícios de onde a ação ocorre.

**5.2. Pós-processamento com Janela Deslizante:** Outra solução prática, sem alterar o treinamento, é usar o modelo treinado para analisar segmentos menores do vídeo. Podemos dividir o vídeo em **janelas temporais** (ex: segmentos de 2 segundos = 10 frames) e classificar cada segmento com o mesmo modelo treinado. Assim, obtemos várias predições ao longo do vídeo e podemos marcar aquelas janelas cujo modelo indicar como luta. Em implementações em produção, este método é viável: rodamos a sequência de pose de, digamos, frames 0-9, 5-14, 10-19, etc (janelas sobrepostas para granulação fina) e coletamos as probabilidades.

Exemplo ilustrativo de detecção com janela deslizante (pseudocódigo):

```python
window_size = 50  # tamanho da janela em frames (ex: 10s se 5fps)
step = 25         # deslocamento da janela (sobreposi\u00e7\u00e3o de 50%)
video_seq = torch.tensor(video_features).unsqueeze(0).to(device)  # seq de um v\u00eddeo (1, T, 68)
model.eval()
frame_scores = np.zeros(len(video_features))
with torch.no_grad():
    for start in range(0, len(video_features) - window_size + 1, step):
        end = start + window_size
        window_seq = video_seq[:, start:end, :]  # shape (1, window_size, 68)
        logit = model(window_seq)
        prob = torch.sigmoid(logit).item()
        if prob >= 0.5:
            # marca frames da janela como luta (aumenta o score)
            frame_scores[start:end] += 1
# Frames com score alto indicam alta confian\u00e7a de luta naquele intervalo
suspected_fight_frames = np.where(frame_scores > 0)[0]
```

No exemplo acima, sempre que uma janela é classificada como luta, todos os frames dessa janela recebem um incremento de score. Ao final, frames com score > 0 seriam aqueles pertencentes a pelo menos uma janela detectada como luta. Podemos então agrupar frames consecutivos marcados para obter intervalos de tempo contínuos. Claro, este método depende da capacidade do classificador distinguir segmentos; se a briga ocupar só parte da janela, ainda assim a janela toda será marcada. Reduzir o tamanho da janela aumenta a resolução da detecção temporal, mas janelas muito pequenas podem não ter contexto suficiente para o modelo decidir.

**5.3. Considerações adicionais:** Uma vez identificado os frames/intervalos de luta estimados, poderíamos **destacar esses trechos** na saída do sistema (por exemplo, sinalizando o timestamp de início e fim da luta no vídeo). Para maior precisão, seria ideal ter rótulos temporais no conjunto de treino (frames anotados onde há luta) e treinar a RNN como um modelo de sequência rotulada (por exemplo, usando loss em cada frame). Sem isso, as abordagens acima tentam inferir indiretamente. Uma alternativa de aprendizado é **detecção de anomalia**: presumir que movimentos muito bruscos ou poses caóticas indiquem violência. Métricas como velocidade dos membros (derivadas dos keypoints entre frames) ou detecção de quedas poderiam complementar a identificação de frames violentos. Essas heurísticas, combinadas com a saída do modelo, podem refinar a localização temporal.

## 6. Avaliação e Métricas

Para avaliar o desempenho do pipeline, usaremos um conjunto de teste separado (por exemplo, 20% dos vídeos, ou via cross-validation para maior confiança dado o dataset pequeno). As principais **métricas de classificação binária** a reportar são:

- **Acurácia (Accuracy):** proporção de vídeos corretamente classificados (luta vs não-luta). Com dataset balanceado (50/50), a acurácia é informativa – esperamos um valor alto se o modelo funciona bem.
- **F1-Score:** a média harmônica de precisão e revocação para a classe "luta" (ou podemos computar para ambas classes e fazer média, mas geralmente foca-se na classe positiva). O F1 dá uma noção de equilíbrio entre falsos positivos e falsos negativos. É útil pois em alguns cenários de segurança preferiríamos minimizar falsos negativos (perder uma luta real) mesmo que signifique alguns falsos alarmes.
- **AUC (Area Under ROC Curve):** mede a qualidade do modelo em ordenar vídeos luta acima de não-luta independentemente de um limiar. Uma AUC alta (próxima de 1.0) indica que o modelo atribui probabilidades mais altas consistentemente para vídeos de luta. Calculamos a AUC usando as probabilidades de saída (antes do threshold de 0.5).

Durante o treinamento/validação, já calculamos a acurácia. Para F1 e AUC, podemos usar bibliotecas como scikit-learn:

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Supondo que temos listas: y_true (0/1 verdadeiros) e y_pred (0/1 preditos) e y_prob (probabilidade predita)
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, pos_label=1)
auc = roc_auc_score(y_true, y_prob)
print(f"Acur\u00e1cia: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
```

Na avaliação, devemos observar também a **matriz de confusão** para ver quantos erros de cada tipo ocorrem (quantos vídeos de luta foram perdidos vs quantos vídeos normais foram incorretamente marcados como luta). Dado o objetivo de alta precisão, podemos buscar hiperparâmetros que maximizem F1 ou minimizem falsos negativos, de acordo com a necessidade.

Para uma análise mais detalhada, podemos calcular precisão e revocação separadamente:

- *Precisão (precision)* = acertos_luta / (acertos_luta + falsos_positivos_luta) – proporção de predições de luta que realmente eram lutas.
- *Revocação (recall)* = acertos_luta / (acertos_luta + falsos_negativos_luta) – proporção de lutas reais que o modelo detectou.

Finalmente, reportar as métricas no teste final. Por exemplo, espera-se que a acurácia e F1 alcancem valores elevados (talvez >90% dada a natureza do dataset e a abordagem pose+RNN bem direcionada, conforme sugerido em trabalhos anteriores ([GitHub - jpowellgz/FightDetectionPoseLSTM: Fight detection using Open Pose and Bi-LSTM](https://github.com/jpowellgz/FightDetectionPoseLSTM#:~:text=fight%20detection%20research%3A%20Movie%20Fight,of%20classification%20and%20execution%20times))), e AUC próximo de 0.95+ indicando boa separabilidade.

## 7. Otimizações e Aprimoramentos Adicionais

Para melhorar ainda mais o desempenho e robustez do pipeline, ou adaptá-lo a cenários maiores, consideramos algumas otimizações e extensões:

- **Fluxo Óptico (Optical Flow):** Incorporar informações de movimento explícito além das poses. O fluxo óptico calcula vetores de movimento de pixels entre frames consecutivos, capturando quão rápidas e onde ocorrem mudanças. Em cenas de luta, geralmente há movimentos rápidos dos participantes. Poderíamos usar uma abordagem de **dois fluxos (two-stream)**: um modelo focado nas poses (estrutura estática do corpo) e outro focado no fluxo (dinâmica do movimento), combinando as duas fontes. Por exemplo, extrair mapas de fluxo óptico ou até apenas a velocidade de cada ponto-chave entre frames e fornecer esses valores adicionais ao vetor de características. Isso daria à RNN consciência explícita de movimentos bruscos (socos, chutes).
- **Mecanismo de Atenção Temporal:** Integrar atenção na rede recorrente para permitir que o modelo aprenda quais frames da sequência são mais relevantes para a decisão de luta. Uma camada de atenção poderia ponderar os outputs da LSTM antes da camada final, efetivamente focando nos momentos de conflito. Isso não só pode melhorar a acurácia, mas também fornecer interpretabilidade – as pontuações de atenção podem indicar os intervalos de luta (atendendo ao item 5). Alternativamente, poderíamos explorar arquiteturas baseadas em Transformer (autoatenção) no lugar de LSTM, as quais são poderosas para capturar dependências temporais longas e já vêm com mecanismos de atenção integrados.
- **Modelos de Poses Avançados:** Embora o YOLOPose seja eficiente, poderíamos testar outros estimadores de pose de ponta, como **ViTPose** (baseado em Vision Transformers) ([Two-Stage Violence Detection Using ViTPose and Classification ...](https://www.researchgate.net/publication/373551997_Two-Stage_Violence_Detection_Using_ViTPose_and_Classification_Models_at_Smart_Airports#:~:text=Two,spatial%20and%20temporal%20information)), que podem fornecer poses ainda mais acuradas. Poses mais confiáveis significam melhores features para o RNN. No entanto, modelos assim podem ser mais pesados – é um balanço entre acurácia de pose e velocidade (YOLOPose é uma boa escolha de compromisso, já comprovada em tempo real).
- **Representação de Features de Pose:** Em vez de usar diretamente coordenadas absolutas, podemos usar features mais robustas:
  - **Ângulos e Distâncias:** Calcular ângulos entre segmentos (braço-antebraço, coxa-perna, etc.) e distâncias relativas entre pessoas (por exemplo, distância entre centroids dos dois indivíduos). Isso captura a *postura* e a *interação* (proximidade) de forma invariável a escala. O trabalho de referência com OpenPose+LSTM extraiu vetores de ângulos de movimento para representar ações ([GitHub - jpowellgz/FightDetectionPoseLSTM: Fight detection using Open Pose and Bi-LSTM](https://github.com/jpowellgz/FightDetectionPoseLSTM#:~:text=neural%20network%20for%20human%20detection,it%20opens%20different%20possibilities%20for)).
  - **Normalização por pessoa:** Tornar poses relativas à posição/escala da própria pessoa (ex.: centralizar o esqueleto no centro de massa, normalizar pelo tamanho do tronco). Isso reduziria variação caso a câmera aproxime/afaste.
  - **Velocidade/Aceleração:** Incluír diferenças de coordenadas entre frames (primeira derivada temporal) para cada ponto. Isso dá noção de movimento local de cada membro. Poderia ser concatenado ao vetor ou até usado como entrada a uma segunda RNN ou um componente separado.
- **Aprimoramentos na Rede Recorrente:** Poderíamos experimentar **camadas LSTM adicionais** ou neurônios ocultos extras se a capacidade do modelo for insuficiente, embora cuidado para não sobreajustar ao dataset pequeno. Uma variante útil é usar **Bi-LSTM** (já incluímos essa opção) ou até uma **GRU bidirecional** – GRUs têm menos parâmetros que LSTMs e podem ser mais rápidas; podemos testá-las. Outra ideia é um modelo **hierárquico**: primeiro uma RNN por pessoa (tomando a sequência de poses de cada indivíduo) para extrair uma representação de comportamento individual, e depois combinar as representações (por exemplo, concatenar e passar a outra RNN ou FC) para decidir luta vs não-luta. Isso refletiria a interação entre duas pessoas.
- **Aumento de Dados (Data Augmentation):** Com apenas 50 vídeos de luta para treino, aumentar os dados pode ser crucial. Podemos aplicar transformações nos vídeos ou nas sequências de pose:
  - Espelhar vídeos horizontalmente (espelhamento não altera a semântica de luta).
  - Perturbar ligeiramente as poses ou adicionar ruído nos pontos (simulando erro de detecção ou variação natural).
  - Se possível, gerar sequências sintetizadas de luta via simulação ou usar partes de vídeos existentes remontadas.
  - Aproveitar frames extras além da amostragem de 5fps se performance permitir, ou criar versões de sequências com frame rate diferente.
- **Pipeline em Tempo Real:** Embora não explicitamente requisitado, vale notar: uma GPU 1650 provavelmente consegue processar em torno de 5-10 FPS com YOLOPose small em 320x240. Para implantação em tempo real, otimizar cada parte é importante: usar modelo de pose otimizado (até TensorRT), processar frames de forma assíncrona (pipeline), e talvez reduzir resolução. Nossa arquitetura é adequada para processamento offline (classificar vídeos gravados), mas com alguns ajustes (janelas móveis) poderia rodar continuamente em streaming.

Em suma, o pipeline descrito fornece uma base sólida para detecção de lutas em vídeo combinando **informação espacial** (poses humanas por frame) e **informação temporal** (sequência de poses). Utilizando um modelo de pose de última geração (YOLOPose) e uma rede recorrente bem treinada, esperamos alta precisão na classificação ([GitHub - jpowellgz/FightDetectionPoseLSTM: Fight detection using Open Pose and Bi-LSTM](https://github.com/jpowellgz/FightDetectionPoseLSTM#:~:text=fight%20detection%20research%3A%20Movie%20Fight,of%20classification%20and%20execution%20times)). Com as otimizações sugeridas – incorporação de movimento (fluxo óptico), atenção temporal, features de pose mais ricas – poderemos aumentar ainda mais a confiabilidade e identificar exatamente **quando** as lutas ocorrem nos vídeos. Essa abordagem modular, implementada com boas práticas (c\u00f3digo organizado em componentes de dados, modelo e treinamento; uso eficiente de GPU; logging e salvamento apropriado), deve ser compatível com o hardware especificado e escalável para futuras melhorias. 

