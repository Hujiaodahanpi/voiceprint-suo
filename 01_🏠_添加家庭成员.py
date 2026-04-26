import streamlit as st
import os
import sqlite3
import numpy as np

from start_streamlit_core import get_predictor, temp_audio_folder, insert_mission, set_bg_img

set_bg_img()

st.markdown(
    """
    <div style="font-size: 45px; font-weight: bold;">
        🏠 添加家庭成员
    </div>
    <div style="font-size: 18px; color: #666; margin-bottom: 20px;">
        录入家人的声音，让门锁“认识”每一位家庭成员
    </div>
    """,
    unsafe_allow_html=True
)

name = st.text_input('家庭成员姓名（如：爸爸、妈妈、小明）')
uploaded_file = st.file_uploader(
    "上传该成员的语音样本（建议在安静环境下录制3-5秒说话声）",
    type=['wav', 'mp3', 'ogg'], accept_multiple_files=False)

if uploaded_file is not None:
    st.write("文件预览")
    st.audio(uploaded_file,
             format=f'audio/{uploaded_file.type.split("/")[-1]}')

if name and uploaded_file:
    if st.button("🔒 录入声纹，添加到门锁"):
        file_path = temp_audio_folder + uploaded_file.name
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.read())

        with st.spinner('处理中，请稍候...'):
            get_predictor().register(user_name=name, audio_data=file_path)

            embedding = get_predictor().extract_embedding(file_path)

            if embedding is not None:
                conn = sqlite3.connect('sqlite3.db')
                c = conn.cursor()

                c.execute('''CREATE TABLE IF NOT EXISTS speaker_embeddings
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            speaker_name TEXT NOT NULL,
                            embedding BLOB NOT NULL,
                            audio_path TEXT NOT NULL,
                            created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

                c.execute("INSERT INTO speaker_embeddings (speaker_name, embedding, audio_path) VALUES (?, ?, ?)",
                          (name, embedding.tobytes(), file_path))

                conn.commit()
                conn.close()
                st.success("✅ 声纹录入成功！现在起，说出开门指令即可识别身份。")
            else:
                st.warning("声纹编码提取失败，请检查音频文件质量。")

        os.remove(file_path)
        insert_mission('添加家庭成员', f'成功录入：{name}')