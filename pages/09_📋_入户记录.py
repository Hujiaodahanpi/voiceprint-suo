import streamlit as st
import sqlite3
import pandas as pd

from start_streamlit_core import set_bg_img

set_bg_img()


def fetch_missions():
    conn = sqlite3.connect('sqlite3.db')
    query = '''SELECT * FROM missions'''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


st.markdown(
    """
    <div style="font-size: 45px; font-weight: bold;">
        📋 入户记录
    </div>
    <div style="font-size: 18px; color: #666; margin-bottom: 20px;">
        查看所有声纹开锁记录及系统操作日志
    </div>
    """,
    unsafe_allow_html=True
)

missions_df = fetch_missions()

if not missions_df.empty:
    if st.button('🔄 刷新记录'):
        st.rerun()

    missions_df = missions_df[['type', 'created_at', 'result']]
    missions_df.columns = ['操作类型', '时间', '详细信息']
    st.write('📅 近期入户记录：')
    st.table(missions_df)
else:
    st.write('暂无入户记录。')