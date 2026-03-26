import csv
import json
import os
import socket
import subprocess
from datetime import datetime
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from ai_face_cam import recognize_face_cam
from dataset_source import get_dataset_path, sync_dataset_from_drive

# =============== CONFIG ===============
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_DIR, "attendance_backend")
USER_MAPPING_FILE = os.path.join(PROJECT_DIR, "user_mapping.json")

LIVENESS_PY_RELATIVE_CANDIDATES = [
    os.path.join("env_liveness311", "Scripts", "python.exe"),
    os.path.join("env_liveness", "Scripts", "python.exe"),
    os.path.join("venv_liveness", "Scripts", "python.exe"),
]
LIVENESS_SCRIPT = os.path.join(PROJECT_DIR, "liveness_cam.py")

LOG_FILE = os.path.join(PROJECT_DIR, "checkin_log.csv")
LOG_HEADER = ["datetime", "device_id", "name", "distance_step1", "distance_step3", "status"]
LEGACY_LOG_HEADER = ["datetime", "name", "distance_step1", "distance_step3", "status"]

def _env_flag(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


USE_API = _env_flag("CHECKIN_USE_API", default=False)
API_URL = os.environ.get("CHECKIN_API_URL", "http://localhost:3000/api/checkin").strip()
API_TIMEOUT_SEC = 5
API_RETRY = 1
API_KEY = ""
SYNC_DATASET_ON_STARTUP = True
SYNC_FORCE_REFRESH = True
STEP1_TIMEOUT_SEC = 10
STEP3_TIMEOUT_SEC = 10
# =====================================

DEVICE_ID = socket.gethostname()


def _resolve_project_path(relative_paths):
    for rel_path in relative_paths:
        candidate = os.path.join(PROJECT_DIR, rel_path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(PROJECT_DIR, relative_paths[0])


ENV_LIVENESS_PY = _resolve_project_path(LIVENESS_PY_RELATIVE_CANDIDATES)


def _fmt_dist(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _load_user_mapping():
    if not os.path.exists(USER_MAPPING_FILE):
        return {}

    try:
        with open(USER_MAPPING_FILE, "r", encoding="utf-8") as handle:
            raw_mapping = json.load(handle)
    except Exception as exc:
        print(f"User mapping skipped: {exc}")
        return {}

    if not isinstance(raw_mapping, dict):
        print("User mapping skipped: root JSON must be an object")
        return {}

    normalized_mapping = {}
    for face_name, value in raw_mapping.items():
        key = str(face_name).strip()
        if not key:
            continue

        if isinstance(value, str):
            user_id = value.strip()
            if user_id:
                normalized_mapping[key] = {"user_id": user_id, "full_name": key}
            continue

        if not isinstance(value, dict):
            continue

        user_id = str(value.get("user_id", "")).strip()
        full_name = str(value.get("full_name") or value.get("name") or key).strip()
        if user_id:
            normalized_mapping[key] = {
                "user_id": user_id,
                "full_name": full_name or key,
            }

    return normalized_mapping


def resolve_user_identity(name):
    clean_name = str(name or "").strip()
    if not clean_name:
        return "", ""

    mapping = _load_user_mapping()
    if clean_name in mapping:
        entry = mapping[clean_name]
        return entry["user_id"], entry.get("full_name", clean_name)

    return clean_name, clean_name


def _migrate_legacy_log_if_needed():
    if not os.path.exists(LOG_FILE):
        return

    with open(LOG_FILE, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if not rows or rows[0] != LEGACY_LOG_HEADER:
        return

    migrated_rows = [LOG_HEADER]
    for row in rows[1:]:
        if not row:
            continue
        if len(row) >= len(LOG_HEADER):
            migrated_rows.append(row[: len(LOG_HEADER)])
        elif len(row) == len(LEGACY_LOG_HEADER):
            migrated_rows.append([row[0], "", row[1], row[2], row[3], row[4]])
        else:
            padded = row + [""] * (len(LOG_HEADER) - len(row))
            migrated_rows.append(padded[: len(LOG_HEADER)])

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(migrated_rows)


def run_liveness():
    if not os.path.exists(ENV_LIVENESS_PY):
        print("Liveness Python not found.")
        print("Tried:")
        for rel_path in LIVENESS_PY_RELATIVE_CANDIDATES:
            print(f" - {rel_path}")
        return False

    if not os.path.exists(LIVENESS_SCRIPT):
        print("Missing liveness_cam.py")
        return False

    try:
        process = subprocess.run(
            [ENV_LIVENESS_PY, LIVENESS_SCRIPT],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        print("Liveness timeout")
        return False
    except OSError as exc:
        print(f"Liveness launch error: {exc}")
        return False

    combined_output = ((process.stdout or "") + "\n" + (process.stderr or "")).strip()
    return "LIVENESS_OK" in combined_output and process.returncode == 0


def append_log(name, dist1, dist3, status):
    _migrate_legacy_log_if_needed()

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        DEVICE_ID,
        name,
        _fmt_dist(dist1),
        _fmt_dist(dist3),
        status,
    ]

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(LOG_HEADER)
        writer.writerow(row)


def send_to_api(payload):
    if not USE_API:
        return

    request_payload = dict(payload)
    request_status = str(request_payload.get("status") or "").strip()
    name_hint = (
        request_payload.get("name")
        or request_payload.get("step3_name")
        or request_payload.get("step1_name")
        or ""
    )

    if request_status and request_status != "SUCCESS":
        print(f"API skipped: backend accepts successful check-ins only ({request_status})")
        return

    if not request_payload.get("user_id"):
        user_id, full_name = resolve_user_identity(name_hint)
        if user_id and user_id != "UNKNOWN":
            request_payload["user_id"] = user_id
            request_payload.setdefault("full_name", full_name)

    if not request_payload.get("user_id"):
        print("API skipped: missing user_id for backend")
        return

    request_payload.setdefault("device_id", DEVICE_ID)
    request_payload.setdefault("datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    data = json.dumps(request_payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-KEY"] = API_KEY

    last_error = None
    for _attempt in range(API_RETRY + 1):
        try:
            req = urlrequest.Request(API_URL, data=data, headers=headers, method="POST")
            with urlrequest.urlopen(req, timeout=API_TIMEOUT_SEC) as response:
                body = response.read().decode("utf-8", errors="ignore")
                print(f"API => {response.status}: {body}")
                return
        except (HTTPError, URLError) as exc:
            last_error = exc

    print("API error:", last_error)


def prepare_dataset():
    if not SYNC_DATASET_ON_STARTUP:
        dataset_path = get_dataset_path()
        print(f"Using dataset: {dataset_path}")
        return dataset_path

    try:
        dataset_path = sync_dataset_from_drive(force=SYNC_FORCE_REFRESH, quiet=False)
        print(f"Dataset synced from Google Drive: {dataset_path}")
    except Exception as exc:
        dataset_path = get_dataset_path()
        print(f"Dataset sync skipped: {exc}")
        print(f"Using dataset: {dataset_path}")

    return dataset_path


def check_in_flow():
    print(f"\nSTEP 1: Face recognition ({STEP1_TIMEOUT_SEC} sec)")
    name1, dist1 = recognize_face_cam(timeout_sec=STEP1_TIMEOUT_SEC, show_window=True)

    if name1 == "UNKNOWN":
        print("FAIL: unknown face in step 1")
        append_log("UNKNOWN", dist1, None, "FAIL_STEP1")
        send_to_api(
            {
                "name": "UNKNOWN",
                "status": "FAIL_STEP1",
                "distance_step1": dist1,
                "distance_step3": None,
                "step1_name": "UNKNOWN",
                "step3_name": None,
            }
        )
        return

    print(f"PASS: step 1 matched {name1} (dist={dist1})")

    print("\nSTEP 2: Liveness")
    if not run_liveness():
        print("FAIL: liveness check")
        append_log(name1, dist1, None, "FAIL_LIVENESS")
        send_to_api(
            {
                "name": name1,
                "status": "FAIL_LIVENESS",
                "distance_step1": dist1,
                "distance_step3": None,
                "step1_name": name1,
                "step3_name": None,
            }
        )
        return

    print("PASS: liveness")

    print(f"\nSTEP 3: Face recognition again ({STEP3_TIMEOUT_SEC} sec)")
    name3, dist3 = recognize_face_cam(timeout_sec=STEP3_TIMEOUT_SEC, show_window=True)

    if name3 != name1:
        print(f"FAIL: step 3 mismatch ({name3})")
        append_log(name1, dist1, dist3, "FAIL_STEP3_MISMATCH")
        send_to_api(
            {
                "name": name1,
                "status": "FAIL_STEP3_MISMATCH",
                "distance_step1": dist1,
                "distance_step3": dist3,
                "step1_name": name1,
                "step3_name": name3,
            }
        )
        return

    print(f"PASS: check-in complete for {name3}")
    append_log(name3, dist1, dist3, "SUCCESS")
    send_to_api(
        {
            "name": name3,
            "status": "SUCCESS",
            "distance_step1": dist1,
            "distance_step3": dist3,
            "step1_name": name1,
            "step3_name": name3,
        }
    )


def menu():
    prepare_dataset()

    while True:
        print("========================================")
        print(" AI FACE CHECK-IN SYSTEM ")
        print("========================================")
        print("1) Check-in")
        print("0) Exit")
        print("========================================")
        choice = input("Select menu: ").strip()

        if choice == "1":
            check_in_flow()
            input("\nPress Enter to return to menu...")
        elif choice == "0":
            break
        else:
            print("Please choose 1 or 0.")


if __name__ == "__main__":
    menu()
