import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_PATH = 'models/best.pt' 

EMOTION_MAP = {
    'anger': 'Giận dữ',
    'content': 'Mãn nguyện',
    'disgust': 'Ghê tởm',
    'fear': 'Sợ hãi',
    'happy': 'Hạnh phúc',
    'neutral': 'Bình thường',
    'sad': 'Buồn bã',
    'surprise': 'Ngạc nhiên'
}

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Không tìm thấy file model tại: {MODEL_PATH}")
    st.stop()

st.title("😊 Hệ thống Nhận diện Cảm xúc YOLOv8")
st.write("Đồ án - Sinh viên: Trần Xuân Đức")

tab1, tab2 = st.tabs(["🖼️ Nhận diện qua Ảnh", "📷 Nhận diện qua Webcam"])

# --- TAB 1: UPLOAD ẢNH ---
with tab1:
    st.header("Tải ảnh lên để nhận diện")
    uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Hiển thị ảnh gốc
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã tải lên', width="stretch")
        
        # Nút bấm xử lý
        if st.button('🔍 Phân tích Cảm xúc ngay'):
            with st.spinner('Đang phân tích...'):
                # Dự đoán
                results = model.predict(image, conf=0.20, iou=0.5, imgsz=1280, agnostic_nms=True, augment=True)
                
                # Vẽ kết quả lên ảnh
                # results[0].plot() trả về mảng numpy (BGR), cần chuyển sang RGB để hiển thị đúng màu
                res_plotted = results[0].plot()[:, :, ::-1]
                
                # Hiển thị kết quả
                st.success("Xong!")
                st.image(res_plotted, caption='Kết quả nhận diện', width="stretch")
                
                # In chi tiết ra text
                st.subheader("Chi tiết:")
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    vn_label = EMOTION_MAP.get(label, label)
                    conf = float(box.conf[0])
                    st.write(f"- Phát hiện: **{vn_label}** (Độ tin cậy: {conf:.1%})")

# --- TAB 2: WEBCAM ---
with tab2:
    st.header("Chụp ảnh từ Webcam")
    st.warning("Lưu ý: Trên trình duyệt web, bạn cần nhấn nút 'Take Photo' để chụp ảnh tĩnh và gửi đi phân tích.")
    
    # Widget Webcam của Streamlit
    img_file_buffer = st.camera_input("Bắt đầu!")

    if img_file_buffer is not None:
        # Xử lý khi có ảnh chụp
        image = Image.open(img_file_buffer)
        
        # Dự đoán
        results = model.predict(image, conf=0.20, iou=0.5, imgsz=1280, agnostic_nms=True, augment=True)
        res_plotted = results[0].plot()[:, :, ::-1]
        
        st.image(res_plotted, caption='Kết quả từ Webcam', width="stretch")