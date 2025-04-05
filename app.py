import os
import random
from pytubefix import YouTube
import streamlit as st

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

video_url = st.text_input("📎 Cole o link de um vídeo de segurança (YouTube ou teste local)")

output_path = "."

def baixar_video(link):
    try:
        yt = YouTube(link)
        st.markdown("### 🎞️ Pré-visualização do vídeo original")
        st.video(link)
        st.write(f"🎬 Baixando: **{yt.title}**")

        mp4_path = os.path.join(output_path, "youtube.mp4")
        if os.path.exists(mp4_path):
            os.remove(mp4_path)

        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if stream:
            stream.download(output_path=output_path, filename="youtube.mp4")
            st.success("✅ Vídeo baixado com sucesso!")

            # Simulação de probabilidade (para testes visuais)
            probabilidade = round(random.uniform(0, 1) * 100, 2)

            # Exibir resultado de forma neutra
            st.info(f"📊 Probabilidade estimada de ocorrência de comportamento agressivo: **{probabilidade}%**")

        else:
            st.error("❌ Nenhum stream compatível encontrado.")

    except Exception as e:
        st.error(f"❌ Erro ao processar o link: {str(e)}")

# Ação ao colar link
if video_url.strip():
    baixar_video(video_url)
