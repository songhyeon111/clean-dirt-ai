import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =========================
# UI 설정 (Figma 느낌)
# =========================
st.set_page_config(page_title="Clean vs Dirt AI", layout="wide")

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧼 Clean vs Dirt AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLO Object Detection System</div>', unsafe_allow_html=True)

# =========================
# YOLO 모델 로딩
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# UI
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

# =========================
# 실행
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    # =========================
    # YOLO 추론
    # =========================
    results = model(image)
    result = results[0]

    annotated_image = result.plot()

    with col2:
        st.subheader("🔍 Detection Result")
        st.image(annotated_image, use_container_width=True)

        st.subheader("📊 AI Prediction")

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]

                st.write(f"👉 **{name}** ({conf:.2f})")
        else:
            st.info("No objects detected 🚀")

else:
    with col2:
        st.info("Upload image to see AI prediction 🚀")

