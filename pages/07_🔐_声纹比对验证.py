import streamlit as st
import os

from start_streamlit_core import get_predictor as predictor
from start_streamlit_core import temp_audio_folder, insert_mission, set_bg_img

set_bg_img()

st.markdown(
    """
    <div style="font-size: 45px; font-weight: bold;">
        🔐 声纹比对验证
    </div>
    <div style="font-size: 18px; color: #666; margin-bottom: 20px;">
        比对两段语音，验证是否为同一说话人
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file_1 = st.file_uploader(
    "📁 基准语音",
    type=['wav', 'mp3', 'ogg'], accept_multiple_files=False, key='1')
if uploaded_file_1 is not None:
    st.write("文件预览")
    st.audio(uploaded_file_1,
             format=f'audio/{uploaded_file_1.type.split("/")[-1]}')

uploaded_file_2 = st.file_uploader(
    "📁 待验证语音",
    type=['wav', 'mp3', 'ogg'], accept_multiple_files=False, key='2')
if uploaded_file_2 is not None:
    st.write("文件预览")
    st.audio(uploaded_file_2,
             format=f'audio/{uploaded_file_2.type.split("/")[-1]}')

if uploaded_file_1 and uploaded_file_2:
    if st.button("🔍 开始比对"):
        file_path_1 = temp_audio_folder + uploaded_file_1.name
        file_path_2 = temp_audio_folder + uploaded_file_2.name
        with open(file_path_1, 'wb') as file:
            file.write(uploaded_file_1.read())
        with open(file_path_2, 'wb') as file:
            file.write(uploaded_file_2.read())

        result = None

        with st.spinner('处理中，请稍候...'):
            result = max(predictor().contrast(audio_data1=file_path_1, audio_data2=file_path_2), 0)

        insert_mission('声纹比对验证', f'相似度得分：{result:.4f}')
        os.remove(file_path_1)
        os.remove(file_path_2)
        st.success('处理完毕')
        st.text(f'声纹相似度得分：{result:.4f}')
        if result > 0.5:
            st.success("✅ 相似度较高，判定为同一人")
        else:
            st.warning("⚠️ 相似度较低，可能为不同说话人")