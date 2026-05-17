import streamlit as st
from PIL import Image

st.set_page_config(page_title="Clean Dirt AI", layout="wide")

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .box {
        padding: 20px;
        border-radius: 20px;
        background-color: #f5f7fb;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧼 Clean vs Dirt AI</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Drop your image here")

with col2:
    st.subheader("Result")

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Input Image")
    st.info("YOLO result will appear here 🚀")
