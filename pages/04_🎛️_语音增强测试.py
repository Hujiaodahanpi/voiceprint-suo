import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import uuid

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

from start_streamlit_core import temp_audio_folder, noise_suppression, insert_mission, set_bg_img

st.markdown(
    """
    <div style="font-size: 45px; font-weight: bold;">
        🎛️ 语音增强测试
    </div>
    <div style="font-size: 18px; color: #666; margin-bottom: 20px;">
        模拟门锁在嘈杂环境下的降噪效果（电视声、交谈声等背景音抑制）
    </div>
    """,
    unsafe_allow_html=True
)

set_bg_img()

uploaded_file = st.file_uploader(
    "上传一段含背景噪声的语音（如客厅电视声中的说话声）",
    type=['wav', 'mp3', 'ogg'], accept_multiple_files=False)

if uploaded_file is not None:
    st.write("文件预览")
    st.audio(uploaded_file,
             format=f'audio/{uploaded_file.type.split("/")[-1]}')

    file_path = os.path.join(temp_audio_folder, uploaded_file.name)
    os.makedirs(temp_audio_folder, exist_ok=True)

    try:
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.getbuffer())

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            st.success(f"✅ 文件已保存")

            st.subheader("原始语音波形（含背景噪声）")
            try:
                y, sr = librosa.load(file_path)
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                ax.set_title("原始语音（含背景噪声）")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"加载原始音频时出错: {str(e)}")
        else:
            st.error(f"❌ 上传文件保存失败")
    except Exception as e:
        st.error(f"❌ 文件保存失败: {str(e)}")

if uploaded_file and st.button("🔊 开始降噪处理"):
    unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    file_path = os.path.join(temp_audio_folder, unique_filename)

    try:
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.getbuffer())

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            st.error(f"❌ 上传文件保存失败")
            st.stop()
    except Exception as e:
        st.error(f"❌ 文件保存失败: {str(e)}")
        st.stop()

    new_file_path = None

    with st.spinner('处理中，请稍候...'):
        try:
            new_file_path = noise_suppression(input_audio_path=file_path)
            if not new_file_path or not os.path.exists(new_file_path):
                raise ValueError("降噪处理失败，未生成输出文件")
        except Exception as e:
            st.error(f"降噪处理出错: {str(e)}")
            insert_mission('语音增强测试', 'Failed')
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

    if new_file_path and os.path.exists(new_file_path):
        insert_mission('语音增强测试', 'Success')
        st.success('处理完毕')
        st.balloons()
        st.audio(new_file_path)

        st.subheader("降噪后语音波形（纯净人声）")
        try:
            y_denoised, sr_denoised = librosa.load(new_file_path)
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y_denoised, sr=sr_denoised, ax=ax)
            ax.set_title("降噪后语音（纯净人声，更适合声纹识别）")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"加载降噪后音频时出错: {str(e)}")