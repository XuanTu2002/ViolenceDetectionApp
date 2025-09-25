import cv2
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from ultralytics import YOLO
from collections import deque

# ---- CÀI ĐẶT ----
# Đường dẫn đến model VideoMAE bạn đã fine-tune và lưu lại
MODEL_PATH = "./videomae-finetuned-rlvs-saved"
# Sử dụng GPU nếu có, nếu không thì dùng CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Số lượng frame cần thu thập cho model VideoMAE
NUM_FRAMES_REQUIRED = 16
# Ngưỡng tin cậy cho việc phát hiện người
CONFIDENCE_THRESHOLD = 0.5
# Số frame chờ giữa các lần chạy VideoMAE cho cùng một người (giảm tải CPU/GPU)
INFERENCE_INTERVAL = 12

def initialize_webcam(camera_indices=(0,), backends=None):
    '''Try to open the webcam across common Windows backends.'''
    if backends is None:
        backends = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        backends.append(cv2.CAP_ANY)

    for index in camera_indices:
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                if hasattr(cv2, "CAP_DSHOW") and backend == cv2.CAP_DSHOW:
                    backend_name = "CAP_DSHOW"
                elif hasattr(cv2, "CAP_MSMF") and backend == cv2.CAP_MSMF:
                    backend_name = "CAP_MSMF"
                elif backend == cv2.CAP_ANY:
                    backend_name = "CAP_ANY"
                else:
                    backend_name = str(backend)
                print(f"Opened webcam {index} using backend {backend_name}.")
                return cap
            cap.release()

    return None


print("Loading models...")

# 1. Tải model Phân loại Hành động (VideoMAE)
try:
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
    video_classifier = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    video_classifier.eval()  # Chuyển sang chế độ đánh giá
    print("VideoMAE model loaded successfully.")
except Exception as e:
    print(f"Lỗi khi tải model VideoMAE: {e}")
    print("Vui lòng đảm bảo đường dẫn MODEL_PATH là chính xác và thư mục chứa đầy đủ các file model.")
    exit()

# 2. Tải model Dò tìm Đối tượng (YOLOv8)
object_detector = YOLO("yolov8n.pt")
print("YOLO model loaded successfully.")

# ---- XỬ LÝ VIDEO REAL-TIME ----

cap = initialize_webcam()
if cap is None:
    print("Lỗi: Không thể mở webcam với các backend đã thử (CAP_DSHOW, CAP_MSMF, CAP_ANY).")
    print("Vui lòng kiểm tra quyền truy cập camera hoặc ứng dụng khác đang sử dụng camera.")
    exit()

tracked_frames = {}
track_states = {}

print("Bắt đầu xử lý video từ webcam... Nhấn 'q' để thoát.")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Không thể nhận frame từ webcam. Kết thúc.")
            break

        results = object_detector.track(frame, persist=True, classes=[0], verbose=False)
        active_track_ids = set()

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                active_track_ids.add(track_id)

                x1, y1, x2, y2 = box
                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                    continue

                if track_id not in tracked_frames:
                    tracked_frames[track_id] = deque(maxlen=NUM_FRAMES_REQUIRED)

                state = track_states.setdefault(
                    track_id,
                    {
                        "label": "Analyzing...",
                        "color": (255, 255, 0),
                        "frames_until_inference": 0,
                    },
                )

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                tracked_frames[track_id].append(person_crop_rgb)

                has_enough_frames = len(tracked_frames[track_id]) == NUM_FRAMES_REQUIRED
                if has_enough_frames and state["frames_until_inference"] == 0:
                    try:
                        inputs = processor(list(tracked_frames[track_id]), return_tensors="pt").to(DEVICE)
                        with torch.no_grad():
                            outputs = video_classifier(**inputs)
                            logits = outputs.logits
                            predicted_class_idx = logits.argmax(-1).item()
                            predicted_label = video_classifier.config.id2label[predicted_class_idx]

                        if predicted_label.lower() == "violence":
                            state["label"] = "BAO LUC"
                            state["color"] = (0, 0, 255)
                        else:
                            state["label"] = "BINH THUONG"
                            state["color"] = (0, 255, 0)
                    except Exception:
                        state["label"] = "Error"
                        state["color"] = (128, 128, 128)
                    finally:
                        state["frames_until_inference"] = INFERENCE_INTERVAL
                elif has_enough_frames and state["frames_until_inference"] > 0:
                    state["frames_until_inference"] = max(state["frames_until_inference"] - 1, 0)
                else:
                    state["frames_until_inference"] = 0
                    if state["label"] == "Error":
                        state["label"] = "Analyzing..."
                        state["color"] = (255, 255, 0)

                label_text = state["label"]
                box_color = state["color"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame,
                    f"ID: {track_id} - {label_text}",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2,
                )

        for track_id in list(tracked_frames.keys()):
            if track_id not in active_track_ids:
                tracked_frames.pop(track_id, None)
                track_states.pop(track_id, None)

        cv2.imshow("Real-time Violence Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Nhấn 'q' - Đang thoát chương trình.")
            break
except KeyboardInterrupt:
    print("Nhận KeyboardInterrupt - Đang thoát chương trình.")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng chương trình.")
