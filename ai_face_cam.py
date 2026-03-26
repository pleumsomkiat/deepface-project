import os
import time
from collections import Counter, deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from verify import verify_face


TIMEOUT_SEC = 10
CAM_INDEX = 0
CAP_BACKEND = cv2.CAP_DSHOW
CHECK_INTERVAL_SEC = 0.5
MATCH_WINDOW_SIZE = 4
MIN_CONFIRMATIONS = 2

CASCADE_SCALE_FACTOR = 1.1
CASCADE_MIN_NEIGHBORS = 4
CASCADE_MIN_SIZE = (60, 60)
CASCADE_FALLBACKS = [
    ("haarcascade_frontalface_default.xml", 1.1, 4, (60, 60)),
    ("haarcascade_frontalface_alt2.xml", 1.05, 3, (45, 45)),
    ("haarcascade_profileface.xml", 1.1, 4, (60, 60)),
]


def _get_font(size=32):
    candidates = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text_thai(bgr_img, text, x, y, size=32, color=(255, 255, 0)):
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


_face_cascades = [
    (
        cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name),
        scale_factor,
        min_neighbors,
        min_size,
    )
    for cascade_name, scale_factor, min_neighbors, min_size in CASCADE_FALLBACKS
]


def detect_face_bbox(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    best_face = None
    best_area = -1

    for cascade, scale_factor, min_neighbors, min_size in _face_cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        for x, y, w, h in faces:
            area = int(w) * int(h)
            if area > best_area:
                best_face = (int(x), int(y), int(w), int(h))
                best_area = area

    return best_face


def verify_face_from_crop(face_bgr):
    if face_bgr is None or face_bgr.size == 0:
        return "UNKNOWN", None
    return verify_face(face_bgr, enforce_detection=False)


def _choose_confirmed_match(recent_matches):
    known_matches = [(name, dist) for name, dist in recent_matches if name != "UNKNOWN" and dist is not None]
    if not known_matches:
        return "UNKNOWN", None, 0

    counts = Counter(name for name, _dist in known_matches)
    ranked_names = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_name, top_votes = ranked_names[0]
    runner_up_votes = ranked_names[1][1] if len(ranked_names) > 1 else 0

    if top_votes < MIN_CONFIRMATIONS or top_votes <= runner_up_votes:
        return "UNKNOWN", None, top_votes

    distances = [dist for name, dist in known_matches if name == top_name]
    return top_name, min(distances), top_votes


def recognize_face_cam(timeout_sec=TIMEOUT_SEC, show_window=True, cam_index=CAM_INDEX):
    cap = cv2.VideoCapture(cam_index, CAP_BACKEND)
    if not cap.isOpened():
        return "UNKNOWN", None

    start = time.time()
    last_check_time = 0.0

    confirmed_name = "UNKNOWN"
    confirmed_dist = None
    confirmed_votes = 0

    last_seen_name = "UNKNOWN"
    last_seen_dist = None
    recent_matches = deque(maxlen=MATCH_WINDOW_SIZE)
    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if (now - last_check_time) >= CHECK_INTERVAL_SEC:
            bbox = detect_face_bbox(frame)
            name, dist = verify_face(frame, enforce_detection=False)
            recent_matches.append((name, dist))
            last_seen_name = name
            last_seen_dist = dist
            last_check_time = now

            candidate_name, candidate_dist, candidate_votes = _choose_confirmed_match(recent_matches)
            if candidate_name != "UNKNOWN":
                should_replace = (
                    candidate_votes > confirmed_votes
                    or (
                        candidate_votes == confirmed_votes
                        and candidate_dist is not None
                        and (confirmed_dist is None or candidate_dist < confirmed_dist)
                    )
                )
                if should_replace:
                    confirmed_name = candidate_name
                    confirmed_dist = candidate_dist
                    confirmed_votes = candidate_votes
        elif bbox is None:
            bbox = detect_face_bbox(frame)

        if show_window:
            disp = frame.copy()

            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)

            confirmed_txt = "N/A" if confirmed_dist is None else f"{confirmed_dist:.3f}"
            last_txt = "N/A" if last_seen_dist is None else f"{last_seen_dist:.3f}"

            line1 = f"CONFIRMED: {confirmed_name}  dist:{confirmed_txt}"
            line2 = f"LAST: {last_seen_name}  dist:{last_txt}"
            line3 = f"WINDOW VOTES: {confirmed_votes}/{MATCH_WINDOW_SIZE}"
            line4 = "Press Q or ESC to exit"

            disp = draw_text_thai(disp, line1, 20, 20, size=32, color=(255, 255, 0))
            disp = draw_text_thai(disp, line2, 20, 60, size=28, color=(255, 255, 0))
            disp = draw_text_thai(disp, line3, 20, 100, size=26, color=(255, 255, 0))
            disp = draw_text_thai(disp, line4, 20, 135, size=24, color=(255, 255, 0))

            cv2.imshow("Face Recognition", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break

        if (time.time() - start) >= timeout_sec:
            break

    cap.release()
    cv2.destroyAllWindows()

    return confirmed_name, confirmed_dist


if __name__ == "__main__":
    name, dist = recognize_face_cam(timeout_sec=15, show_window=True)
    print("RESULT_NAME:", name)
    print("RESULT_DISTANCE:", dist)
