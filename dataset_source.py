import json
import os
import shutil
from datetime import datetime

try:
    import gdown
except ImportError:  # pragma: no cover - handled at runtime
    gdown = None


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
DRIVE_SYNC_ROOT = os.path.join(PROJECT_DIR, "dataset_drive_cache")
DRIVE_STAGING_ROOT = os.path.join(PROJECT_DIR, "dataset_drive_cache_staging")
METADATA_FILE = os.path.join(PROJECT_DIR, ".dataset_source.json")

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1nERWHZPKMj1skrc4iWhwmqOWMbRpBq1r?usp=sharing"
DRIVE_FOLDER_ID = "1nERWHZPKMj1skrc4iWhwmqOWMbRpBq1r"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _has_image_files(path):
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            return True
    return False


def _is_dataset_root(path):
    if not os.path.isdir(path):
        return False
    child_dirs = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    return any(_has_image_files(child_dir) for child_dir in child_dirs)


def _find_dataset_root(root_path):
    best_path = None
    best_score = -1

    for current_root, dir_names, _file_names in os.walk(root_path):
        child_dirs = [os.path.join(current_root, name) for name in dir_names]
        score = sum(1 for child_dir in child_dirs if _has_image_files(child_dir))
        if score > best_score:
            best_path = current_root
            best_score = score

    if best_score > 0:
        return best_path
    return None


def _load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_metadata(dataset_path):
    payload = {
        "dataset_path": dataset_path,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "drive_folder_url": DRIVE_FOLDER_URL,
        "drive_folder_id": DRIVE_FOLDER_ID,
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def get_dataset_path():
    metadata = _load_metadata()
    metadata_path = metadata.get("dataset_path")
    if metadata_path and _is_dataset_root(metadata_path):
        return metadata_path

    cached_path = _find_dataset_root(DRIVE_SYNC_ROOT)
    if cached_path:
        return cached_path

    return LOCAL_DATASET_DIR


def sync_dataset_from_drive(force=False, quiet=False):
    if gdown is None:
        raise RuntimeError("gdown is not installed in the active environment.")

    target_root = DRIVE_STAGING_ROOT if force else DRIVE_SYNC_ROOT
    if os.path.exists(target_root):
        shutil.rmtree(target_root)

    os.makedirs(target_root, exist_ok=True)
    downloaded_files = gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=target_root,
        quiet=quiet,
        resume=not force,
    )
    if not downloaded_files:
        raise RuntimeError("No files were downloaded from Google Drive.")

    dataset_root = _find_dataset_root(target_root)
    if not dataset_root:
        raise RuntimeError("Downloaded Drive folder does not look like a valid dataset.")

    if force:
        if os.path.exists(DRIVE_SYNC_ROOT):
            shutil.rmtree(DRIVE_SYNC_ROOT)
        os.replace(DRIVE_STAGING_ROOT, DRIVE_SYNC_ROOT)
        dataset_root = _find_dataset_root(DRIVE_SYNC_ROOT)

    _save_metadata(dataset_root)
    return dataset_root
