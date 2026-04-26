import streamlit as st
import os
from datetime import datetime

from start_streamlit_core import temp_audio_folder, insert_mission, set_bg_img, word

st.markdown(
    """
    <div style="font-size: 45px; font-weight: bold;">
        💬 访客留言转文字
    </div>
    <div style="font-size: 18px; color: #666; margin-bottom: 20px;">
        模拟门锁将访客语音留言实时转写为文字，方便主人远程查看
    </div>
    """,
    unsafe_allow_html=True
)

set_bg_img()

uploaded_file = st.file_uploader(
    "📨 上传访客留言录音",
    type=['wav', 'mp3', 'ogg'], accept_multiple_files=False)

if uploaded_file is not None:
    st.write("文件预览")
    st.audio(uploaded_file,
             format=f'audio/{uploaded_file.type.split("/")[-1]}')

if uploaded_file:
    if st.button("📝 开始转写"):
        file_path = temp_audio_folder + uploaded_file.name
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.read())

        with st.spinner("语音转文字中，请稍候..."):
            try:
                transcription = word(file_path)
                insert_mission('访客留言转写', 'Success')
            except Exception as e:
                st.error(f"处理失败：{str(e)}")
                insert_mission('访客留言转写', 'Failed')
                os.remove(file_path)
                st.stop()

        os.remove(file_path)
        st.success("转换完成！")
        st.balloons()
        st.audio(uploaded_file)
        st.markdown("### 📋 留言内容")
        st.write(transcription)

        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("<h3 style='font-size:24px; color:orange;'>结果文本</h3>", unsafe_allow_html=True)
                st.text_area("", transcription, height=200)
            with col2:
                st.markdown("<h3 style='font-size:24px; color:orange;'>操作</h3>", unsafe_allow_html=True)
                st.markdown("<h3 style='font-size:24px;'></h3>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 保存留言文本",
                    data=transcription,
                    file_name=f"访客留言_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        st.markdown("---")