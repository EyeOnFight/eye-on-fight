import os
import streamlit as st
from pytubefix import YouTube
from model_bridge import predict_fight  # Importa a função de predição

st.set_page_config(page_title="📹 Monitoramento de Segurança - Detecção de Lutas", layout="centered")

st.markdown(
    """
    <style>
    .big-title {
        font-size: 36px;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
    }
    .subtext {
        font-size: 18px;
        text-align: center;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">🎥 Sistema de Detecção de Lutas</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Análise automática de vídeos de segurança para identificar comportamentos violentos.</div><br>', unsafe_allow_html=True)

# Seleção do modo de entrada: Link do YouTube ou Upload de vídeo local
video_mode = st.radio("Selecione a fonte do vídeo:", ("Link do YouTube", "Upload de vídeo local"))

if video_mode == "Link do YouTube":
    video_url = st.text_input("📎 Cole o link de um vídeo de segurança (YouTube)")
    if video_url.strip():
        try:
            yt = YouTube(video_url)
            st.markdown("### 🎞️ Pré-visualização do vídeo original")
            st.video(video_url)
            st.write(f"🎬 Baixando: **{yt.title}**")
            mp4_path = os.path.join(".", "youtube.mp4")
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if stream:
                stream.download(output_path=".", filename="youtube.mp4")
                st.success("✅ Vídeo baixado com sucesso!")
                probability = predict_fight(mp4_path)
                st.info(f"📊 Probabilidade estimada de comportamento agressivo: **{probability:.2f}%**")
            else:
                st.error("❌ Nenhum stream compatível encontrado.")
        except Exception as e:
            st.error(f"❌ Erro ao processar o link: {str(e)}")

elif video_mode == "Upload de vídeo local":
    uploaded_video = st.file_uploader("Selecione um arquivo de vídeo", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        # Salva o arquivo enviado em um caminho temporário
        temp_video_path = os.path.join(".", "temp_uploaded_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.video(temp_video_path)
        st.success("✅ Vídeo carregado com sucesso!")
        try:
            probability = predict_fight(temp_video_path)
            st.info(f"📊 Probabilidade estimada de comportamento agressivo: **{probability:.2f}%**")
        except Exception as e:
            st.error(f"❌ Erro ao processar o vídeo: {str(e)}")
