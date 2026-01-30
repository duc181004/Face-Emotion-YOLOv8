import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import ImageDraw, ImageFont, Image

def draw_text_vietnamese(img_cv, text, pos, color, font_size=20):
    # Chuyển ảnh OpenCV (Numpy) sang PIL
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 2. Load font chữ
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Vẽ chữ
    draw.text(pos, text, font=font, fill=color)
    
    # Chuyển ngược lại thành OpenCV (BGR) để các hàm khác xử lý tiếp
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

YOLO_PATH = 'models/best.pt'
RESNET_PATH = 'models/resnet50_emotion_finetuned.pth'

CLASS_NAMES = ['anger', 'content', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_MAP = {
    'anger': 'Giận dữ', 'content': 'Mãn nguyện', 'disgust': 'Ghê tởm',
    'fear': 'Sợ hãi', 'happy': 'Hạnh phúc', 'neutral': 'Bình thường',
    'sad': 'Buồn bã', 'surprise': 'Ngạc nhiên'
}

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_PATH)

@st.cache_resource
def load_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, len(CLASS_NAMES))
    )
    
    try:
        state_dict = torch.load(RESNET_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Lỗi load ResNet: {e}")
        return None, None

yolo_model = load_yolo()
resnet_model, device = load_resnet()

# HÀM XỬ LÝ ẢNH CHO RESNET 
# Biến đổi ảnh mặt cắt ra về chuẩn 224x224
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🧠 Hệ thống AI 2 Giai đoạn (YOLOv8 + ResNet50)")
st.write("Đồ án tốt nghiệp - Sinh viên: Trần Xuân Đức")

tab1, tab2 = st.tabs(["🖼️ Upload Ảnh", "📷 Webcam"])

def process_and_draw(image_pil):
    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    results = yolo_model(image_pil, conf=0.15, iou=0.5, imgsz=1280, augment=True, agnostic_nms=True, max_det=20)
    
    faces_found = 0
    
    for box in results[0].boxes:
        faces_found += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Xử lý tọa độ
        h, w, _ = img_cv.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_img = image_pil.crop((x1, y1, x2, y2))
        
        if resnet_model:
            # Tiền xử lý
            input_tensor = preprocess(face_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = resnet_model(input_tensor)
                # Lấy xác suất (Softmax)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                label_en = CLASS_NAMES[pred_idx.item()]
                label_vn = EMOTION_MAP.get(label_en, label_en)
                score = conf.item()

        # Vẽ kết quả
        color = (0, 255, 0) if label_en in ['happy', 'content'] else (0, 0, 255)
        
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        text = f"{label_vn} ({score:.0%})"

        pil_color = (color[2], color[1], color[0]) # Đảo ngược BGR -> RGB
        
        img_cv = draw_text_vietnamese(img_cv, text, (x1, y1 - 30), pil_color, font_size=30)

    # Convert lại sang RGB để hiển thị lên Web
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), faces_found

# Xử lý giao diện 
with tab1:
    uploaded_file = st.file_uploader("Chọn ảnh...", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Ảnh gốc', width="stretch")
        
        if st.button('🔍 Phân tích 2-Stage'):
            with st.spinner('YOLO đang tìm mặt, ResNet đang soi cảm xúc...'):
                final_img, count = process_and_draw(image)
                st.success(f"Đã tìm thấy {count} khuôn mặt!")
                st.image(final_img, caption='Kết quả 2-Stage', width="stretch")

with tab2:
    img_buffer = st.camera_input("Chụp ảnh")
    if img_buffer:
        image = Image.open(img_buffer).convert('RGB')
        final_img, count = process_and_draw(image)
        st.image(final_img, caption='Kết quả Webcam', width="stretch")