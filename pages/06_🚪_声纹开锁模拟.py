import streamlit as st
import os

from start_streamlit_core import get_predictor as predictor
from start_streamlit_core import temp_audio_folder, insert_mission, set_bg_img

set_bg_img()

st.markdown(
    """
    <div style="font-size: 45px; font-weight: bold;">
        🚪 声纹开锁模拟
    </div>
    <div style="font-size: 18px; color: #666; margin-bottom: 20px;">
        上传一段语音，系统将识别说话人身份并模拟开锁决策
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "🎤 上传开锁语音",
    type=['wav', 'mp3', 'ogg'], accept_multiple_files=False)
if uploaded_file is not None:
    st.write("文件预览")
    st.audio(uploaded_file,
             format=f'audio/{uploaded_file.type.split("/")[-1]}')

if uploaded_file:
    if st.button("🔓 识别身份并模拟开锁"):
        file_path = temp_audio_folder + uploaded_file.name
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.read())

        name, score = None, None

        with st.spinner('处理中，请稍候...'):
            name, score = predictor().recognition(audio_data=file_path)

        os.remove(file_path)
        if name:
            insert_mission('声纹开锁', f'识别用户：{name}，匹配度：{score:.4f}')
            st.balloons()
            st.success(f"🔓 识别成功！欢迎回家，{name}！")
            st.text(f"声纹匹配度：{score:.4f}")
            st.markdown("---")
            st.markdown("### 🚪 门锁状态：已开锁 ✅")
        else:
            insert_mission('声纹开锁', '身份验证失败')
            st.error("❌ 身份验证失败，门锁未开启。请确认您已录入声纹或尝试靠近门锁再次说话。")