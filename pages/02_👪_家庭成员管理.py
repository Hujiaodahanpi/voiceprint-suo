import streamlit as st
import pandas as pd

from start_streamlit_core import get_predictor as predictor
from start_streamlit_core import insert_mission, set_bg_img

set_bg_img()

st.markdown(
    """
    <div style="font-size: 45px; font-weight: bold;">
        👪 家庭成员管理
    </div>
    <div style="font-size: 18px; color: #666; margin-bottom: 20px;">
        查看已录入的家人声纹，可移除不再需要开门的成员
    </div>
    """,
    unsafe_allow_html=True
)

name = None

with st.spinner('加载中，请稍后...'):
    name = st.radio(
        "选择要管理的家庭成员：",
        predictor().get_users(),
        index=None,
    )
    insert_mission('家庭成员管理', '查看成员列表')

if name:
    if st.button(f"🗑️ 移除「{name}」的开门权限"):
        with st.spinner('正在从门锁中移除...'):
            predictor().remove_user(user_name=name)
            insert_mission('家庭成员管理', f'移除成员：{name}')
        st.success(f"已移除 {name} 的声纹，该成员将无法通过声纹开门。")
        st.rerun()