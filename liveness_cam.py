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

# ================== CONFIG ==================
REPS_REQUIRED = 2
TIMEOUT_SEC = 35
CAM_INDEX = 0

# Windows แนะนำ CAP_DSHOW (ถ้าเครื่องคุณใช้ MSMF ดีกว่า ให้เปลี่ยนเป็น cv2.CAP_MSMF)
CAP_BACKEND = cv2.CAP_DSHOW

BLINK_THRESHOLD = 0.25
YAW_THRESHOLD = 0.04
PITCH_THRESHOLD = 0.04

FLIP_LEFT_RIGHT = True
# ===========================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152
FOREHEAD = 10

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
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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

    yaw = nose.x - cx
    pitch = nose.y - cy
    return yaw, pitch

def move_to_text(m):
    return {
        "BLINK": "กระพริบตา",
        "LEFT": "หันซ้าย",
        "RIGHT": "หันขวา",
        "UP": "เงยหน้า",
        "DOWN": "ก้มหน้า",
    }.get(m, m)

def is_neutral(yaw, pitch):
    return abs(yaw) < (YAW_THRESHOLD * 0.6) and abs(pitch) < (PITCH_THRESHOLD * 0.6)

def detect_move(yaw, pitch, move):
    if FLIP_LEFT_RIGHT:
        yaw = -yaw

    if move == "LEFT":
        return yaw < -YAW_THRESHOLD
    if move == "RIGHT":
        return yaw > YAW_THRESHOLD
    if move == "UP":
        return pitch < -PITCH_THRESHOLD
    if move == "DOWN":
        return pitch > PITCH_THRESHOLD
    return False

def run_action(face_mesh, cap, action, reps_required=2, timeout_sec=35):
    start = time.time()
    reps = 0
    eye_closed = False
    in_pose = False

    while True:
        if (time.time() - start) >= timeout_sec:
            return False

        ret, frame = cap.read()
        if not ret:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        info1 = f"STEP 2: ทำท่า: {move_to_text(action)} ให้ครบ {reps_required} ครั้ง"
        info2 = f"ความคืบหน้า: {reps}/{reps_required}"
        info3 = "กด Q เพื่อออก"

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            if action == "BLINK":
                ear = (eye_aspect_ratio(lm, LEFT_EYE) + eye_aspect_ratio(lm, RIGHT_EYE)) / 2.0
                if ear < BLINK_THRESHOLD and not eye_closed:
                    eye_closed = True
                elif ear >= BLINK_THRESHOLD and eye_closed:
                    reps += 1
                    eye_closed = False
            else:
                yaw, pitch = head_pose(lm)
                target_pose = detect_move(yaw, pitch, action)

                if target_pose and not in_pose:
                    in_pose = True

                if in_pose and is_neutral(yaw, pitch):
                    reps += 1
                    in_pose = False

            if reps >= reps_required:
                frame = draw_text_thai(frame, "ผ่านท่านี้แล้ว ✅", 20, 130, size=30, color=(0, 255, 0))
                cv2.imshow("STEP 2 - Liveness", frame)
                cv2.waitKey(700)
                return True
        else:
            info1 = "ไม่พบใบหน้า กรุณาอยู่ในกล้อง"
            info2 = ""

        frame = draw_text_thai(frame, info1, 20, 20, size=30, color=(255, 255, 0))
        frame = draw_text_thai(frame, info2, 20, 60, size=26, color=(255, 255, 0))
        frame = draw_text_thai(frame, info3, 20, 95, size=24, color=(255, 255, 0))

        cv2.imshow("STEP 2 - Liveness", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            return False

def main():
    cap = cv2.VideoCapture(CAM_INDEX, CAP_BACKEND)
    if not cap.isOpened():
        print("LIVENESS_FAIL")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    actions = ["BLINK", "LEFT", "RIGHT", "UP", "DOWN"]
    random.shuffle(actions)
    chosen = actions[:2]

    ok_all = True
    for act in chosen:
        per_action_timeout = max(10, TIMEOUT_SEC // len(chosen))
        ok = run_action(face_mesh, cap, act, reps_required=REPS_REQUIRED, timeout_sec=per_action_timeout)
        if not ok:
            ok_all = False
            break

    cap.release()
    cv2.destroyAllWindows()

    print("LIVENESS_OK" if ok_all else "LIVENESS_FAIL")

if __name__ == "__main__":
    main()