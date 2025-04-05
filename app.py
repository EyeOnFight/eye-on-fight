import os
from pytubefix import YouTube
import streamlit as st

st.set_page_config(page_title="YouTube Downloader", layout="centered")
st.title("📥 YouTube Downloader Automático")

video_url = st.text_input("🔗 Cole o link do vídeo do YouTube e pressione Enter")

output_path = "."

def baixar_video(link):
    try:
        yt = YouTube(link)
        st.video(link)
        st.write(f"🎬 Baixando: **{yt.title}**")

        mp4_path = os.path.join(output_path, "youtube.mp4")
        if os.path.exists(mp4_path):
            os.remove(mp4_path)

        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if stream:
            stream.download(output_path=output_path, filename="youtube.mp4")
            st.success("✅ Download concluído com sucesso!")
            st.video("youtube.mp4")
        else:
            st.error("❌ Nenhum stream compatível encontrado.")

    except Exception as e:
        st.error(f"❌ Erro ao processar o link: {str(e)}")

# Só tenta baixar se o campo não estiver vazio
if video_url.strip():
    baixar_video(video_url)
