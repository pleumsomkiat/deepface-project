# main_checkin.py
import os
import csv
import subprocess
import socket
import json
from datetime import datetime
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

from ai_face_cam import recognize_face_cam

# =============== CONFIG ===============
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_LIVENESS_PY = os.path.join(PROJECT_DIR, "env_liveness", "Scripts", "python.exe")
LIVENESS_SCRIPT = os.path.join(PROJECT_DIR, "liveness_cam.py")

LOG_FILE = os.path.join(PROJECT_DIR, "checkin_log.csv")

# ---- API (เปิด/ปิดได้) ----
USE_API = True
API_URL = "http://localhost:3000/checkin"   # เปลี่ยนเป็น IP เครื่องเพื่อนถ้าคนละเครื่อง
API_TIMEOUT_SEC = 5
API_RETRY = 1  # retry เพิ่มอีก 1 ครั้งพอ (รวมเป็น 2 attempts)
API_KEY = ""   # ถ้าเพื่อนทำ auth ให้ใส่ key ตรงนี้ เช่น "abc123"
# =====================================

DEVICE_ID = socket.gethostname()

def _fmt_dist(x):
    if x is None:
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

def run_liveness():
    if not os.path.exists(ENV_LIVENESS_PY):
        print("ไม่พบ env_liveness\\Scripts\\python.exe (ตั้งค่า env_liveness ก่อน)")
        return False

    if not os.path.exists(LIVENESS_SCRIPT):
        print("ไม่พบไฟล์ liveness_cam.py")
        return False

    # กันค้าง: timeout = 45 วิ (หรือ TIMEOUT_SEC ใน liveness + buffer)
    try:
        p = subprocess.run(
            [ENV_LIVENESS_PY, LIVENESS_SCRIPT],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=45
        )
    except subprocess.TimeoutExpired:
        print("Liveness timeout")
        return False

    combined = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()

    # เช็คแบบ token ชัด ๆ
    ok = ("LIVENESS_OK" in combined) and (p.returncode == 0 or p.returncode is None)
    return ok

def append_log(name, dist1, dist3, status):
    header = ["datetime", "device_id", "name", "distance_step1", "distance_step3", "status"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        DEVICE_ID,
        name,
        _fmt_dist(dist1),
        _fmt_dist(dist3),
        status
    ]

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

def send_to_api(payload: dict):
    if not USE_API:
        return

    # เติมข้อมูลกลางให้ทุก request
    payload = dict(payload)
    payload.setdefault("device_id", DEVICE_ID)
    payload.setdefault("datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-KEY"] = API_KEY

    last_err = None
    for attempt in range(API_RETRY + 1):
        try:
            req = urlrequest.Request(API_URL, data=data, headers=headers, method="POST")
            with urlrequest.urlopen(req, timeout=API_TIMEOUT_SEC) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                print(f"API => {resp.status}: {body}")
                return
        except (HTTPError, URLError) as e:
            last_err = e
    print("API error:", last_err)

def check_in_flow():
    print("\nSTEP 1: ยืนยันตัวตนรอบแรก (15 วินาที)")
    name1, dist1 = recognize_face_cam(timeout_sec=15, show_window=True)

    if name1 == "UNKNOWN":
        print("FAIL: ไม่ผ่านการยืนยันตัวตน (UNKNOWN)")
        append_log("UNKNOWN", dist1, None, "FAIL_STEP1")

        send_to_api({
            "name": "UNKNOWN",
            "status": "FAIL_STEP1",
            "distance_step1": dist1,
            "distance_step3": None,
            "step1_name": "UNKNOWN",
            "step3_name": None,
        })
        return

    print(f"PASS: พบตัวตน = {name1} (dist={dist1})")

    print("\nSTEP 2: Liveness (กระพริบตา + สุ่มท่า 2 อย่าง)")
    ok_live = run_liveness()
    if not ok_live:
        print("FAIL: ไม่ผ่าน Liveness")
        append_log(name1, dist1, None, "FAIL_LIVENESS")

        send_to_api({
            "name": name1,
            "status": "FAIL_LIVENESS",
            "distance_step1": dist1,
            "distance_step3": None,
            "step1_name": name1,
            "step3_name": None,
        })
        return

    print("PASS: Liveness ผ่าน")

    print("\nSTEP 3: ยืนยันตัวตนอีกรอบ (15 วินาที)")
    name3, dist3 = recognize_face_cam(timeout_sec=15, show_window=True)

    if name3 != name1:
        print(f"FAIL: รอบสุดท้ายไม่ตรง (ได้ {name3})")
        append_log(name1, dist1, dist3, "FAIL_STEP3_MISMATCH")

        send_to_api({
            "name": name1,  # เจ้าของ attempt
            "status": "FAIL_STEP3_MISMATCH",
            "distance_step1": dist1,
            "distance_step3": dist3,
            "step1_name": name1,
            "step3_name": name3,
        })
        return

    print(f"PASS: ยืนยันตัวตนครบ ระบบบันทึกเช็คชื่อให้แล้ว ({name3})")
    append_log(name3, dist1, dist3, "SUCCESS")

    send_to_api({
        "name": name3,
        "status": "SUCCESS",
        "distance_step1": dist1,
        "distance_step3": dist3,
        "step1_name": name1,
        "step3_name": name3,
    })

def menu():
    while True:
        print("========================================")
        print(" AI FACE CHECK-IN SYSTEM ")
        print("========================================")
        print("1) Check-in")
        print("0) Exit")
        print("========================================")
        choice = input("เลือกเมนู: ").strip()

        if choice == "1":
            check_in_flow()
            input("\nกด Enter เพื่อกลับเมนู...")
        elif choice == "0":
            break
        else:
            print("กรุณาเลือก 1 หรือ 0")

if __name__ == "__main__":
    menu()