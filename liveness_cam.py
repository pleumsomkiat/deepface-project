# liveness_cam.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import time
import random
import cv2
import mediapipe as mp
import numpy as np

from PIL import ImageFont, ImageDraw, Image

BLINK_REQUIRED = 2
MOVE_REQUIRED = 2
TIMEOUT_SEC = 35
CAM_INDEX = 0

BLINK_THRESHOLD = 0.25
YAW_THRESHOLD = 0.04
PITCH_THRESHOLD = 0.04

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152
FOREHEAD = 10

# ---------- Thai text (PIL) ----------
def _get_font(size=28):
    candidates = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def draw_text_thai(bgr_img, text, x, y, size=28, color=(255, 255, 0)):
    # color เป็น RGB
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------- math ----------
def eye_aspect_ratio(lm, eye_idx):
    p = [lm[i] for i in eye_idx]
    v1 = np.linalg.norm(np.array([p[1].x, p[1].y]) - np.array([p[5].x, p[5].y]))
    v2 = np.linalg.norm(np.array([p[2].x, p[2].y]) - np.array([p[4].x, p[4].y]))
    h  = np.linalg.norm(np.array([p[0].x, p[0].y]) - np.array([p[3].x, p[3].y]))
    return (v1 + v2) / (2.0 * h + 1e-6)

def head_pose(lm):
    nose = lm[NOSE_TIP]
    lch = lm[LEFT_CHEEK]
    rch = lm[RIGHT_CHEEK]
    chin = lm[CHIN]
    forehead = lm[FOREHEAD]

    cx = (lch.x + rch.x) / 2.0
    cy = (forehead.y + chin.y) / 2.0

    # เดิม: yaw = nose.x - cx  # + ขวา, - ซ้าย (ในภาพ "ไม่ flip")
    # แต่เราจะ flip ภาพให้เป็นกระจก => ต้องกลับเครื่องหมาย yaw เพื่อให้คำสั่งซ้าย/ขวาตรงกับที่เห็น
    yaw = -(nose.x - cx)      # ✅ แก้ตรงนี้
    pitch = nose.y - cy       # + ก้ม, - เงย
    return yaw, pitch

def detect_move(yaw, pitch, move):
    if move == "LEFT":
        return yaw > YAW_THRESHOLD   # ✅ สลับ
    if move == "RIGHT":
        return yaw < -YAW_THRESHOLD  # ✅ สลับ
    if move == "UP":
        return pitch < -PITCH_THRESHOLD
    if move == "DOWN":
        return pitch > PITCH_THRESHOLD
    return False


def move_to_text(m):
    return {"LEFT": "หันซ้าย", "RIGHT": "หันขวา", "UP": "เงยหน้า", "DOWN": "ก้มหน้า"}.get(m, m)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("LIVENESS_FAIL")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    moves = ["LEFT", "RIGHT", "UP", "DOWN"]
    random.shuffle(moves)
    chosen = moves[:2]  # สุ่ม 2 ท่า

    stage = "BLINK"
    blink_count = 0
    eye_closed = False

    current_move_index = 0
    move_count = 0
    move_ready = False  # ทำท่าถูกแล้ว รอ "กระพริบตา" เพื่อยืนยันนับ

    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ แก้: flip ภาพแบบกระจก เพื่อให้ผู้ใช้ทำตามง่าย
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        info1 = ""
        info2 = ""
        info3 = "กด Q เพื่อออก"

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # blink detect
            ear = (eye_aspect_ratio(lm, LEFT_EYE) + eye_aspect_ratio(lm, RIGHT_EYE)) / 2.0
            if ear < BLINK_THRESHOLD and not eye_closed:
                eye_closed = True
            elif ear >= BLINK_THRESHOLD and eye_closed:
                blink_count += 1
                eye_closed = False

                # ล็อกด้วยกระพริบตา: ถ้าท่าถูกแล้วค่อยนับ
                if stage == "MOVE" and move_ready:
                    move_count += 1
                    move_ready = False

            yaw, pitch = head_pose(lm)

            if stage == "BLINK":
                info1 = f"STEP 2: กรุณากระพริบตาให้ครบ {BLINK_REQUIRED} ครั้ง"
                info2 = f"ตอนนี้: {blink_count}/{BLINK_REQUIRED}"
                if blink_count >= BLINK_REQUIRED:
                    stage = "MOVE"
                    current_move_index = 0
                    move_count = 0
                    move_ready = False

            elif stage == "MOVE":
                target = chosen[current_move_index]
                info1 = f"ทำท่า: {move_to_text(target)} {MOVE_REQUIRED} ครั้ง"
                info2 = f"ความคืบหน้า: {move_count}/{MOVE_REQUIRED} (ทำท่าแล้วกระพริบตาเพื่อนับ)"

                if detect_move(yaw, pitch, target):
                    move_ready = True

                if move_count >= MOVE_REQUIRED:
                    current_move_index += 1
                    move_count = 0
                    move_ready = False

                    if current_move_index >= len(chosen):
                        print("LIVENESS_OK")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
        else:
            info1 = "ไม่พบใบหน้า กรุณาอยู่ในกล้อง"
            info2 = ""

        frame = draw_text_thai(frame, info1, 20, 20, size=30, color=(255, 255, 0))
        frame = draw_text_thai(frame, info2, 20, 60, size=26, color=(255, 255, 0))
        frame = draw_text_thai(frame, info3, 20, 95, size=24, color=(255, 255, 0))

        cv2.imshow("STEP 2 - Liveness", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break

        if (time.time() - start) >= TIMEOUT_SEC:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("LIVENESS_FAIL")

# ใส่ท้ายไฟล์ liveness_cam.py
def run_liveness():
    # เรียก main() เดิม แต่เปลี่ยนให้คืน True/False
    # วิธีง่าย: ให้ main() print แล้วเราไม่ parse => ปรับ main() ให้ return True/False เลย
    return main()

if __name__ == "__main__":
    ok = main()
    print("LIVENESS_OK" if ok else "LIVENESS_FAIL")

