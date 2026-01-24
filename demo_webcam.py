import cv2
from ultralytics import YOLO

# --- CẤU HÌNH ---
MODEL_PATH = 'models/best.pt'  # Đường dẫn đến file model
CONFIDENCE_THRESHOLD = 0.5     # Độ tin cậy tối thiểu (0.5 = 50%)

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

# Màu sắc cho từng cảm xúc (BGR format)
COLORS = {
    'anger': (0, 0, 255),      # Đỏ
    'happy': (0, 255, 0),      # Lục
    'sad': (255, 0, 0),        # Lam
    'neutral': (128, 128, 128) # Xám
    # Các màu khác mặc định sẽ là trắng
}

def main():
    # 1. Load Model
    print("⏳ Đang tải model...")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Tải model thành công!")
    except Exception as e:
        print(f"❌ Lỗi không tìm thấy model: {e}")
        return

    # 2. Mở Webcam (Số 0 là cam mặc định, nếu không lên thử đổi thành 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở Webcam.")
        return

    print("🎥 Đang chạy camera... Nhấn phím 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Nhận diện
        # stream=True giúp xử lý nhanh hơn cho video
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # 4. Vẽ kết quả lên màn hình
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ và nhãn
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] # Tên gốc (anger, happy...)
                
                # Chuyển sang tiếng Việt
                label_vi = EMOTION_MAP.get(cls_name, cls_name)
                conf = float(box.conf[0])
                
                # Chọn màu sắc
                color = COLORS.get(cls_name, (0, 255, 255)) # Mặc định là Vàng

                # Vẽ hình chữ nhật
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Viết chữ lên trên
                text = f"{label_vi} ({conf:.1f})"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 5. Hiển thị
        cv2.imshow('YOLOv8 Emotion Detection', frame)

        # Bấm 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()