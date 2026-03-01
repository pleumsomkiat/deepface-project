# main_checkin.py
import os
import csv
import subprocess
from datetime import datetime

from ai_face_cam import recognize_face_cam

# =============== CONFIG ===============
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_LIVENESS_PY = os.path.join(PROJECT_DIR, "env_liveness", "Scripts", "python.exe")
LIVENESS_SCRIPT = os.path.join(PROJECT_DIR, "liveness_cam.py")

LOG_FILE = os.path.join(PROJECT_DIR, "checkin_log.csv")

# ---- API (เปิด/ปิดได้) ----
USE_API = False
API_URL = "http://localhost:3000/checkin"   # ปรับตาม server ของคุณ
API_TIMEOUT_SEC = 5
# =====================================

def run_liveness():
    if not os.path.exists(ENV_LIVENESS_PY):
        print("ไม่พบ env_liveness\\Scripts\\python.exe (ตั้งค่า env_liveness ก่อน)")
        return False

    if not os.path.exists(LIVENESS_SCRIPT):
        print("ไม่พบไฟล์ liveness_cam.py")
        return False

    p = subprocess.run(
        [ENV_LIVENESS_PY, LIVENESS_SCRIPT],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR
    )

    out = (p.stdout or "").strip()
    return "LIVENESS_OK" in out

def append_log(name, dist1, dist3, status):
    header = ["datetime", "name", "distance_step1", "distance_step3", "status"]
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, dist1, dist3, status]

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

def send_to_api(name, status, dist1=None, dist3=None):
    if not USE_API:
        return

    try:
        import requests  # import ตรงนี้ เผื่อเครื่องที่ไม่ใช้ API จะไม่ต้องลง requests
        payload = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "status": status,
            "distance_step1": dist1,
            "distance_step3": dist3,
        }
        r = requests.post(API_URL, json=payload, timeout=API_TIMEOUT_SEC)
        print(f"API => {r.status_code}: {r.text}")
    except Exception as e:
        print("API error:", e)

def check_in_flow():
    print("\nSTEP 1: ยืนยันตัวตนรอบแรก (15 วินาที)")
    name1, dist1 = recognize_face_cam(timeout_sec=15, show_window=True)

    if name1 == "UNKNOWN":
        print("FAIL: ไม่ผ่านการยืนยันตัวตน (UNKNOWN)")
        append_log("UNKNOWN", dist1, None, "FAIL_STEP1")
        send_to_api("UNKNOWN", "FAIL_STEP1", dist1, None)
        return

    print(f"PASS: พบตัวตน = {name1} (dist={dist1})")

    print("\nSTEP 2: Liveness (กระพริบตา + สุ่มท่า 2 อย่าง)")
    ok_live = run_liveness()
    if not ok_live:
        print("FAIL: ไม่ผ่าน Liveness")
        append_log(name1, dist1, None, "FAIL_LIVENESS")
        send_to_api(name1, "FAIL_LIVENESS", dist1, None)
        return

    print("PASS: Liveness ผ่าน")

    print("\nSTEP 3: ยืนยันตัวตนอีกรอบ (15 วินาที)")
    name3, dist3 = recognize_face_cam(timeout_sec=15, show_window=True)

    if name3 != name1:
        print(f"FAIL: รอบสุดท้ายไม่ตรง (ได้ {name3})")
        append_log(name1, dist1, dist3, "FAIL_STEP3_MISMATCH")
        send_to_api(name1, "FAIL_STEP3_MISMATCH", dist1, dist3)
        return

    print(f"PASS: ยืนยันตัวตนครบ ระบบบันทึกเช็คชื่อให้แล้ว ({name3})")
    append_log(name3, dist1, dist3, "SUCCESS")
    send_to_api(name3, "SUCCESS", dist1, dist3)

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