import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(page_title="Clean AI", layout="wide")

model = YOLO("runs/classify/train/weights/best.pt")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0b1220;
}

.title {
    text-align:center;
    font-size:48px;
    font-weight:800;
    color:white;
    margin-bottom:20px;
}

.card {
    background:#111a2e;
    padding:25px;
    border-radius:20px;
    box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

.result {
    font-size:70px;
    font-weight:900;
    text-align:center;
    padding:20px;
    border-radius:15px;
}

.clean {background:#22c55e; color:white;}
.dirt {background:#ef4444; color:white;}

.sub {
    text-align:center;
    color:#94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='title'>🧹 Clean / Dirt AI</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

# ---------------- UPLOAD ----------------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    file = st.file_uploader("이미지 업로드", type=["jpg","png","jpeg"])
    
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICT ----------------
if file:
    with st.spinner("AI 분석 중..."):
        time.sleep(1)
        results = model(image)

    names = results[0].names
    probs = results[0].probs.data.tolist()

    pred = names[probs.index(max(probs))]
    conf = max(probs)

    # ---------------- RESULT ----------------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if pred == "clean":
            st.markdown(f"<div class='result clean'>CLEAN</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result dirt'>DIRT</div>", unsafe_allow_html=True)

        st.write("### Confidence")
        st.progress(float(conf))
        st.write(f"{conf:.2f}")

        st.markdown("<p class='sub'>YOLOv8 AI Model</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)