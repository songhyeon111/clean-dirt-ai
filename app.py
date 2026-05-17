import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =========================
# UI 설정
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
    .result-box {
        padding: 20px;
        border-radius: 15px;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧼 Clean vs Dirt AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLO-based Cleaning Detection System</div>', unsafe_allow_html=True)

# =========================
# YOLO 모델
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
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

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

    # =========================
    # 판정 로직 (핵심)
    # =========================
    detected_classes = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            detected_classes.append(name)

    # Dirty 여부 판단
    is_dirty = "dirt" in [c.lower() for c in detected_classes]

    # =========================
    # 결과 출력
    # =========================
    with col2:
        st.subheader("🔍 Detection Result")
        st.image(annotated_image, use_container_width=True)

        if is_dirty:
            st.error("🧹 청소가 필요합니다")
        else:
            if len(detected_classes) == 0:
                st.success("✨ 패널이 깨끗합니다")
            else:
                st.success("✨ 패널이 깨끗합니다")

        # 상세 정보
        st.subheader("📊 Detected Objects")

        if len(detected_classes) > 0:
            for c in detected_classes:
                st.write(f"👉 {c}")
        else:
            st.write("No objects detected")

else:
    with col2:
        st.info("이미지를 업로드하면 결과가 표시됩니다 🚀")

