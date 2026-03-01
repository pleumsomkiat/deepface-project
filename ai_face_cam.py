# ai_face_cam.py
import os
import time
import cv2
from deepface import DeepFace

from PIL import ImageFont, ImageDraw, Image
import numpy as np

# ================== CONFIG ==================
DATASET_PATH = "dataset"
MODEL_NAME = "Facenet512"
DETECTOR = "opencv"
DISTANCE_METRIC = "cosine"

# ยิ่งต่ำยิ่งเข้มงวด (Facenet512+cosine มักอยู่ราวๆ 0.30-0.40)
THRESHOLD = 0.35

TIMEOUT_SEC = 15
CAM_INDEX = 0

# Windows แนะนำ CAP_DSHOW (ถ้าเครื่องคุณใช้ MSMF ดีกว่า ให้เปลี่ยนเป็น cv2.CAP_MSMF)
CAP_BACKEND = cv2.CAP_DSHOW

# ตรวจหน้าไม่ถี่เกิน (DeepFace.find หนัก)
CHECK_INTERVAL_SEC = 0.7
# ===========================================


# ------------------ Thai text helper (PIL) ------------------
def _get_font(size=32):
    candidates = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_text_thai(bgr_img, text, x, y, size=32, color=(255, 255, 0)):
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    draw.text((x, y), text, font=font, fill=color)  # color เป็น RGB
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ------------------ Face detect (bbox) ------------------
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_bbox(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # เลือกหน้าที่ใหญ่สุด
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return (x, y, w, h)


def verify_face_from_crop(face_bgr):
    """
    รับรูปหน้าที่ crop แล้ว
    คืนค่า: (name, distance) หรือ ("UNKNOWN", None)
    """
    if face_bgr is None or face_bgr.size == 0:
        return "UNKNOWN", None

    try:
        results = DeepFace.find(
            img_path=face_bgr,
            db_path=DATASET_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            silent=True,
            refresh_database=False,  # ไม่ rebuild db ทุกครั้ง (เร็วขึ้น)
        )

        if not results or results[0] is None or results[0].empty:
            return "UNKNOWN", None

        df = results[0].sort_values("distance", ascending=True)
        best = df.iloc[0]

        dist = float(best["distance"])
        identity_path = str(best["identity"])
        name = os.path.basename(os.path.dirname(identity_path)).strip()

        if name and dist <= THRESHOLD:
            return name, dist

        return "UNKNOWN", dist

    except Exception as e:
        # ปล่อยไว้ให้ดูสาเหตุได้ตอน debug
        # print("verify_face_from_crop error:", e)
        return "UNKNOWN", None


def recognize_face_cam(timeout_sec=TIMEOUT_SEC, show_window=True, cam_index=CAM_INDEX):
    """
    เปิดกล้องแล้วพยายามยืนยันตัวตนภายในเวลาที่กำหนด
    คืนค่า: (best_name, best_dist)
    """
    cap = cv2.VideoCapture(cam_index, CAP_BACKEND)
    if not cap.isOpened():
        return "UNKNOWN", None

    start = time.time()
    last_check_time = 0

    best_name = "UNKNOWN"
    best_dist = None

    last_seen_name = "UNKNOWN"
    last_seen_dist = None
    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detect_face_bbox(frame)

        now = time.time()
        if bbox is not None and (now - last_check_time) >= CHECK_INTERVAL_SEC:
            x, y, w, h = bbox
            face_crop = frame[max(0, y): y + h, max(0, x): x + w]
            name, dist = verify_face_from_crop(face_crop)

            last_seen_name = name
            last_seen_dist = dist
            last_check_time = now

            # เก็บ "ผลดีที่สุด" ตาม distance (ยิ่งต่ำยิ่งดี) เฉพาะที่เป็นคนจริง
            if name != "UNKNOWN" and dist is not None:
                if best_dist is None or dist < best_dist:
                    best_name = name
                    best_dist = dist

        if show_window:
            disp = frame.copy()

            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)

            best_txt = "N/A" if best_dist is None else f"{best_dist:.3f}"
            last_txt = "N/A" if last_seen_dist is None else f"{last_seen_dist:.3f}"

            line1 = f"BEST: {best_name}  dist:{best_txt}"
            line2 = f"LAST: {last_seen_name}  dist:{last_txt}"
            line3 = "กด Q เพื่อออก (หรือรอให้ครบเวลา)"

            disp = draw_text_thai(disp, line1, 20, 20, size=32, color=(255, 255, 0))
            disp = draw_text_thai(disp, line2, 20, 60, size=28, color=(255, 255, 0))
            disp = draw_text_thai(disp, line3, 20, 100, size=26, color=(255, 255, 0))

            cv2.imshow("Face Recognition", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break

        if (time.time() - start) >= timeout_sec:
            break

    cap.release()
    cv2.destroyAllWindows()

    return best_name, best_dist


if __name__ == "__main__":
    name, dist = recognize_face_cam(timeout_sec=15, show_window=True)
    print("RESULT_NAME:", name)
    print("RESULT_DISTANCE:", dist)