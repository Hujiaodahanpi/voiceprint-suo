import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from start_streamlit_core import temp_audio_folder, audio_seperation_num_2, insert_mission, set_bg_img

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

set_bg_img()

st.markdown(
    """
    <div style="font-size: 45px; font-weight: bold;">
        🗣️ 多人语音分离
    </div>
    <div style="font-size: 18px; color: #666; margin-bottom: 20px;">
        当门口有多人同时说话时，系统可分离出每个人的独立语音
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "上传一段多人同时说话的混合录音",
    type=['wav', 'mp3', 'ogg'], accept_multiple_files=False)

if uploaded_file is not None:
    st.write("文件预览")
    st.audio(uploaded_file,
             format=f'audio/{uploaded_file.type.split("/")[-1]}')

if uploaded_file:
    if st.button("🔀 分离说话人"):
        file_path = temp_audio_folder + uploaded_file.name
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.read())

        new_file_path = None
        y_orig, sr_orig = None, None

        # 在分离之前，先加载原始音频信号用于波形绘制
        try:
            y_orig, sr_orig = librosa.load(file_path, sr=None, mono=True)
        except Exception as e:
            st.warning(f"无法读取原始音频用于波形绘制: {str(e)}")

        with st.spinner('处理中，请稍候...'):
            new_file_path = audio_seperation_num_2(input_audio_path=file_path)

        # 删除临时原始文件
        if os.path.exists(file_path):
            os.remove(file_path)

        if new_file_path and len(new_file_path) >= 2:
            insert_mission('多人语音分离', 'Success')
            st.success('处理完毕')

            # ---- 原始混合音频波形 ----
            if y_orig is not None and sr_orig is not None:
                st.markdown("### 📢 原始混合音频波形")
                st.audio(uploaded_file)  # 重新显示原始音频播放器（可从上传控件恢复）
                fig_orig, ax_orig = plt.subplots(figsize=(8, 2))
                librosa.display.waveshow(y_orig, sr=sr_orig, ax=ax_orig, color='gray', alpha=0.7)
                ax_orig.set_title("原始混合语音（分离前）")
                ax_orig.set_xlabel("")
                ax_orig.set_ylabel("")
                st.pyplot(fig_orig)
                plt.close(fig_orig)

            # ---- 说话人1 ----
            st.markdown("### 🎤 说话人1")
            st.audio(new_file_path[0])
            try:
                y1, sr1 = librosa.load(new_file_path[0], sr=None)
                fig1, ax1 = plt.subplots(figsize=(8, 2))
                librosa.display.waveshow(y1, sr=sr1, ax=ax1, color='#3b5bdb')
                ax1.set_title("说话人1 波形图")
                ax1.set_xlabel("")
                ax1.set_ylabel("")
                st.pyplot(fig1)
                plt.close(fig1)
            except Exception as e:
                st.warning(f"说话人1波形生成失败: {str(e)}")

            # ---- 说话人2 ----
            st.markdown("### 🎤 说话人2")
            st.audio(new_file_path[1])
            try:
                y2, sr2 = librosa.load(new_file_path[1], sr=None)
                fig2, ax2 = plt.subplots(figsize=(8, 2))
                librosa.display.waveshow(y2, sr=sr2, ax=ax2, color='#e67e22')
                ax2.set_title("说话人2 波形图")
                ax2.set_xlabel("")
                ax2.set_ylabel("")
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception as e:
                st.warning(f"说话人2波形生成失败: {str(e)}")

        else:
            st.error("❌ 语音分离失败，请检查音频文件格式或质量")
            insert_mission('多人语音分离', 'Failed')