import streamlit as st
from streamlit_chat import message
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import base64

st.set_page_config(
    page_title="Baymax",
    page_icon=":robot:"
)

# 添加背景图的CSS
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode()  # 使用 base64 编码图像数据
    bg_style = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# 调用函数添加背景图
add_bg_from_local(r"D:\BUPT\Models\Llamma1-7b\medicine.png")

# 添加标题
st.title("Baymax")

@st.cache_resource
def get_model():
    model_path = r"D:\BUPT\Models\Llamma1-7b\modelcheckpoint"

    # 使用本地路径加载模型配置
    config = AutoConfig.from_pretrained(model_path)

    # 使用本地路径加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 使用本地路径加载模型
    model = AutoModel.from_pretrained(model_path, config=config).half().cuda()
    model = model.eval()
    return tokenizer, model

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

def predict(input, max_length, top_p, temperature, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > MAX_BOXES:
            history = history[-MAX_TURNS:]
        for i, (query, response) in enumerate(history):
            message(query, avatar_style="big-smile", key=str(i) + "_user")
            message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p, temperature=temperature):
                query, response = history[-1]
                st.write(response)

    return history

container = st.container()

prompt_text = st.text_area(label="用户命令输入", height=100, placeholder="请在这儿输入您的命令")

max_length = st.sidebar.slider('max_length', 0, 4096, 2048, step=1)
top_p = st.sidebar.slider('top_p', 0.0, 1.0, 0.6, step=0.01)
temperature = st.sidebar.slider('temperature', 0.0, 1.0, 0.95, step=0.01)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
