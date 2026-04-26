from flask import Flask, request, redirect, render_template
import subprocess
import time
import atexit
import os

app = Flask(__name__)

streamlit_process = None


def start_streamlit_once():
    """仅在应用启动时启动一次 Streamlit，保持后台运行"""
    global streamlit_process
    if streamlit_process is None:
        print("🚀 正在启动 Streamlit 服务...")
        streamlit_process = subprocess.Popen(
            ['streamlit', 'run', '01_🏠_添加家庭成员.py',
             '--server.headless=true', '--server.port=8501']
        )
        # 等待 Streamlit 完全启动（首次启动需要加载模型，约 3-5 秒）
        time.sleep(3)
        print("✅ Streamlit 服务已就绪")


@atexit.register
def cleanup():
    """Flask 退出时关闭 Streamlit 进程"""
    global streamlit_process
    if streamlit_process:
        print("🛑 正在关闭 Streamlit 服务...")
        streamlit_process.terminate()
        streamlit_process.wait(timeout=3)
        print("✅ Streamlit 服务已关闭")


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # 可根据需要修改账号密码
        if username == "admin" and password == "admin123":
            # 直接重定向到已运行的 Streamlit 服务
            return redirect("http://localhost:8501")
        else:
            return render_template('login.html', error="用户名或密码错误", username=username)
    return render_template('login.html')


if __name__ == '__main__':
    # 先启动 Streamlit 子进程
    start_streamlit_once()
    # 再启动 Flask 服务
    print("🌐 Flask 服务启动中，访问 http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)  # use_reloader=False 避免二次启动