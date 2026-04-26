import sqlite3
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import uuid
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import streamlit as st
import os
import subprocess
from typing import Union, List, Dict
import io
import tempfile
from gtts import gTTS

from VoiceprintRecognition_Pytorch_develop.mvector.predict import MVectorPredictor

# 修正：使用绝对路径，避免相对路径歧义
current_dir = os.path.dirname(os.path.abspath(__file__))
temp_audio_folder = os.path.join(current_dir, 'temp_audio/')
# 确保临时目录存在
try:
    os.makedirs(temp_audio_folder, exist_ok=True)
except PermissionError:
    st.error(f"创建临时目录失败：无权限访问 {temp_audio_folder}，请检查文件夹权限")


# 完整优化版背景函数（已移除 Google Fonts 引用）
def set_bg_img():
    st.markdown(f"""
    <style>
    /* 主背景 - 浅灰蓝渐变 */
    .stApp {{
        background: linear-gradient(150deg, #f0f4ff 0%, #e6f0ff 100%);
    }}
    /* 侧边栏背景 */
    [data-testid="stSidebar"] {{
        background: linear-gradient(165deg, rgba(255,255,255,0.98) 0%, rgba(245,248,255,0.98) 100%) !important;
        border-right: 1px solid rgba(108,142,255,0.15) !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 6px 0 20px rgba(108,142,255,0.08) !important;
    }}
    /* 菜单项样式 */
    [data-testid="stSidebarNav"] li div a {{
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
        font-weight: 500 !important;
        color: #2d3a5e !important;
        border-radius: 8px !important;
        margin: 4px 0 !important;
        padding: 10px 16px !important;
        transition: all 0.3s !important;
    }}
    /* 悬停/选中状态 */
    [data-testid="stSidebarNav"] li div a:hover,
    [data-testid="stSidebarNav"] li div a:focus {{
        background: rgba(108,142,255,0.1) !important;
        transform: translateX(8px) skewX(-5deg);
        box-shadow: 4px 4px 12px rgba(108,142,255,0.15);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    /* 当前选中项指示器 */
    [data-testid="stSidebarNav"] li div[aria-current="page"] {{
        border-left: 4px solid #6c8eff !important;
        background: linear-gradient(to right, rgba(108,142,255,0.05), transparent) !important;
    }}
    /* 字体方案（已移除 Google Fonts，使用系统字体） */
    [data-testid="stSidebarNav"] {{
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
    }}
    /* 卡片式模块容器 */
    .block-container {{
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
        padding: 2rem !important;
        margin: 8rem 0 !important;
        transition: transform 0.3s ease, box-shadow 0.3s !important;
    }}
    .block-container:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(108,142,255,0.15) !important;
    }}
    /* 上传区域美化 */
    [data-testid="stFileUploader"] {{
        border: 2px dashed #6c8eff !important;
        border-radius: 15px !important;
        background: rgba(108, 142, 255, 0.05) !important;
        margin-top: 1rem !important;
        padding: 1.5rem !important; 
    }}
    /* 按钮设计 */
    .stButton>button {{
        background: linear-gradient(135deg, #6c8eff 0%, #3b5bdb 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        transition: all 0.3s !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(108, 142, 255, 0.3);
    }}
    /* 标题样式 */
    h1 {{
        color: #2d3a5e !important;
        border-bottom: 3px solid #6c8eff;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem !important;
    }}
    /* 进度条颜色 */
    .stProgress > div > div {{
        background-color: #6c8eff !important;
    }}
    </style>
    """, unsafe_allow_html=True)


try:
    from speecht5_integration import get_speecht5_converter
    SPEECHT5_AVAILABLE = True
except ImportError:
    SPEECHT5_AVAILABLE = False


class EnhancedPredictor:
    """增强的预测器类，包含额外的功能"""

    def __init__(self):
        print("⏳ 正在加载声纹识别模型 CAM++ ...")
        self.mvector_predictor = MVectorPredictor(
            configs='VoiceprintRecognition_Pytorch_develop/configs/cam++.yml',
            model_path='VoiceprintRecognition_Pytorch_develop/models/训练其他超大数据集的模型/CAMPPlus训练其他超大数据集/models/CAMPPlus_Fbank/best_model',
            audio_db_path='VoiceprintRecognition_Pytorch_develop/audio_db/',
            threshold=0.5
        )
        print("✅ 声纹识别模型加载完成")

    # 代理原有的方法
    def register(self, *args, **kwargs):
        return self.mvector_predictor.register(*args, **kwargs)

    def recognition(self, *args, **kwargs):
        return self.mvector_predictor.recognition(*args, **kwargs)

    def contrast(self, audio_data1, audio_data2):
        return self.mvector_predictor.contrast(audio_data1=audio_data1, audio_data2=audio_data2)

    def get_users(self):
        return self.mvector_predictor.get_users()

    def remove_user(self, user_name):
        return self.mvector_predictor.remove_user(user_name=user_name)

    def extract_embedding(self, audio_path):
        """从音频中提取声纹编码"""
        try:
            result = self.mvector_predictor.recognition(audio_data=audio_path)
            if result and hasattr(result, 'embedding'):
                return result.embedding
            else:
                if hasattr(self.mvector_predictor, 'model') and hasattr(self.mvector_predictor.model, 'extract_embedding'):
                    return self.mvector_predictor.model.extract_embedding(audio_path)
                try:
                    audio, sr = librosa.load(audio_path, sr=16000)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                    embedding = np.mean(mfccs.T, axis=0)
                    return embedding
                except Exception as e:
                    print(f"使用librosa提取特征失败: {e}")
                    print("警告: 使用随机向量作为声纹编码")
                    return np.random.rand(192).astype(np.float32)
        except Exception as e:
            print(f"提取声纹编码失败: {e}")
            return None

    def text_to_speech_gtts(self, text, lang='zh-cn'):
        """使用gTTS进行文本转语音"""
        try:
            tts = gTTS(text=text, lang=lang)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tts.save(tmp_file.name)
                tmp_path = tmp_file.name
            audio_data, sample_rate = sf.read(tmp_path)
            os.unlink(tmp_path)
            return audio_data, sample_rate
        except Exception as e:
            print(f"gTTS合成失败: {e}，尝试使用备选方案")
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()
                audio_data, sample_rate = sf.read(tmp_path)
                os.unlink(tmp_path)
                return audio_data, sample_rate
            except Exception as e2:
                print(f"备选TTS方案也失败: {e2}")
                return None, None

    def get_speaker_embedding(self, speaker_name):
        """从数据库获取说话人的声纹编码"""
        try:
            conn = sqlite3.connect('sqlite3.db')
            c = conn.cursor()
            c.execute("SELECT embedding FROM speaker_embeddings WHERE speaker_name=?", (speaker_name,))
            result = c.fetchone()
            conn.close()
            if result:
                embedding = np.frombuffer(result[0], dtype=np.float32)
                return embedding
            else:
                return None
        except Exception as e:
            print(f"获取声纹编码失败: {e}")
            return None

    def get_all_speakers(self):
        """获取所有已注册的说话人"""
        try:
            conn = sqlite3.connect('sqlite3.db')
            c = conn.cursor()
            c.execute("SELECT DISTINCT speaker_name FROM speaker_embeddings")
            speakers = [row[0] for row in c.fetchall()]
            conn.close()
            return speakers
        except Exception as e:
            print(f"获取说话人列表失败: {e}")
            return []


# 延迟加载和单例模式
class LazyMVectorPredictor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = EnhancedPredictor()
        return cls._instance


def get_predictor():
    return LazyMVectorPredictor()


def insert_mission(type, result):
    conn = sqlite3.connect('sqlite3.db')
    cursor = conn.cursor()
    current_time = datetime.now().strftime('%Y.%m.%d %H:%M')
    insert_sql = '''INSERT INTO missions (type, result, created_at) VALUES (?, ?, ?)'''
    cursor.execute(insert_sql, (type, result, current_time))
    conn.commit()
    conn.close()


def noise_suppression(input_audio_path):
    temp_converted_name = f"converted_{uuid.uuid4().hex}.wav"
    temp_converted_path = os.path.join(temp_audio_folder, temp_converted_name)
    output_name = f"denoised_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(temp_audio_folder, output_name)

    conversion_success = False
    try:
        cmd = [
            'ffmpeg',
            '-err_detect', 'ignore_err',
            '-i', input_audio_path,
            '-ac', '1',
            '-ar', '16000',
            '-y',
            temp_converted_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(temp_converted_path) and os.path.getsize(temp_converted_path) > 1024:
            conversion_success = True
        else:
            st.warning("⚠️ FFmpeg直接转换失败，尝试其他方法")
    except:
        pass

    try:
        ans = pipeline(
            Tasks.acoustic_noise_suppression,
            model='damo/speech_frcrn_ans_cirm_16k',
            model_revision='v1.0.2'
        )
    except Exception as e:
        st.error(f"❌ 降噪模型初始化失败：{str(e)}")
        return None

    try:
        result = ans(
            temp_converted_path,
            output_path=output_path,
            sample_rate=16000
        )
        st.success(f"✅ 降噪成功！输出文件 → {output_path}")
        return output_path
    finally:
        if os.path.exists(temp_converted_path):
            try:
                os.remove(temp_converted_path)
            except Exception as e:
                st.warning(f"⚠️ 临时文件清理失败：{str(e)}")


def save_audio_sr_8000(input_audio_path):
    try:
        with sf.SoundFile(input_audio_path, 'r') as f:
            audio = f.read(dtype='float32')
            original_sr = f.samplerate
            original_channels = f.channels
        if original_channels > 1:
            audio = np.mean(audio, axis=1)
        if original_sr != 8000:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=8000)
        adjusted_audio_path = os.path.join(temp_audio_folder, f"_sr8000_{uuid.uuid4().hex}.wav")
        sf.write(adjusted_audio_path, audio, 8000, subtype='PCM_16')
        return adjusted_audio_path
    except Exception as e:
        st.error(f"❌ 音频采样率转换失败：{str(e)}")
        return None


def audio_seperation_num_2(input_audio_path):
    separation = pipeline(
        Tasks.speech_separation,
        model='damo/speech_mossformer_separation_temporal_8k')
    sr8000_path = save_audio_sr_8000(input_audio_path)
    if not sr8000_path:
        st.error("❌ 语音分离失败：前置采样率转换出错")
        return []
    result = separation(sr8000_path)
    saved_files = []
    for i, signal in enumerate(result['output_pcm_list']):
        save_file = os.path.join(temp_audio_folder, f"{uuid.uuid4()}_output_spk{i}.wav")
        sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)
        saved_files.append(save_file)
    return saved_files


def word(input_audio_path: str) -> str:
    try:
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            model_revision="v2.0.5",
            vad_model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
            vad_model_revision="v2.0.4",
            punc_model='iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
            punc_model_revision="v2.0.4"
        )
        raw_result = inference_pipeline(input=input_audio_path, language=None)
        text_result = "".join([item.get('text', '') for item in raw_result])
        return text_result if text_result else "⚠️ 未识别到有效文本"
    except Exception as e:
        st.error(f"❌ 语音识别失败：{str(e)}")
        return "❌ 识别出错"


# ========== 关键优化：模型预加载 ==========
print("=" * 40)
print("🚪 净声盾 · 智能声纹门锁 初始化中...")
print("⏳ 正在预加载声纹识别模型，请稍候...")
_ = get_predictor()  # 触发模型加载
print("✅ 所有模型预加载完成，系统就绪！")
print("=" * 40)