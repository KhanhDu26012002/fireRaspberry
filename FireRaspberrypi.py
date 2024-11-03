import cv2
from ultralytics import YOLO

# Tải mô hình YOLO đã huấn luyện
model = YOLO('best4.pt')

# Sử dụng camera của Raspberry Pi
cap = cv2.VideoCapture(0)  # 0 là chỉ số của camera mặc định

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize khung hình để giảm tải cho Raspberry Pi
    frame_resized = cv2.resize(frame, (320, 240))

    # Thực hiện phát hiện đối tượng trên từng khung hình
    results = model(frame_resized)

    if results and results[0].boxes:
        for detection in results[0].boxes:
            # Lấy tọa độ khung bao quanh đối tượng
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            confidence = detection.conf[0]  # Mức độ tin cậy
            class_index = int(detection.cls[0])  # Chỉ số lớp
            label = model.names[class_index]  # Lấy tên nhãn từ model

            # Vẽ khung chữ nhật bao quanh đối tượng
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # In ra thông báo nếu phát hiện lửa hoặc khói
            if label == 'Fire':
                print("Fire detected!")
            elif label == 'smoke':
                print("Smoke detected!")

    # Hiển thị khung hình đã được xử lý
    cv2.imshow("Stream View", frame_resized)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()
