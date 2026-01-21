import os
from deepface import DeepFace

DATASET_PATH = "dataset"
MODEL_NAME = "Facenet512"
DETECTOR = "opencv"
THRESHOLD = 0.35  # เข้ม ไม่ผ่านมั่วง่าย

def verify_face(frame, enforce_detection=True):
    """
    รับ frame (ภาพจากกล้อง)
    คืนค่า:
      (name, distance) ถ้าผ่าน
      ("UNKNOWN", None/ระยะ) ถ้าไม่ผ่าน
    """
    try:
        results = DeepFace.find(
            img_path=frame,
            db_path=DATASET_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            distance_metric="cosine",
            enforce_detection=enforce_detection,
            silent=True
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

    except Exception:
        return "UNKNOWN", None
