import math
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DEEPFACE_LOG_LEVEL", "40")
warnings.filterwarnings("ignore", message=r".*sparse_softmax_cross_entropy is deprecated.*")

from deepface import DeepFace

from dataset_source import get_dataset_path


MODEL_NAME = "Facenet512"
DETECTOR = "opencv"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.35
MIN_DIFFERENT_IDENTITY_GAP = 0.08
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_LAST_DATASET_SIGNATURE = None
_EMBEDDING_CACHE_SIGNATURE = None
_EMBEDDING_CACHE = []


def _dataset_signature(dataset_path):
    signature = []
    if not os.path.isdir(dataset_path):
        return tuple(signature)

    for current_root, _dir_names, file_names in os.walk(dataset_path):
        for file_name in sorted(file_names):
            if os.path.splitext(file_name)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            full_path = os.path.join(current_root, file_name)
            try:
                stat = os.stat(full_path)
            except OSError:
                continue

            signature.append(
                (
                    os.path.relpath(full_path, dataset_path),
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            )

    return tuple(signature)


def _should_refresh_database(dataset_path):
    global _LAST_DATASET_SIGNATURE

    current_signature = _dataset_signature(dataset_path)
    should_refresh = current_signature != _LAST_DATASET_SIGNATURE
    _LAST_DATASET_SIGNATURE = current_signature
    return should_refresh


def _represent_face(frame, detector_backend):
    representations = DeepFace.represent(
        img_path=frame,
        model_name=MODEL_NAME,
        detector_backend=detector_backend,
        enforce_detection=False,
    )
    if not representations:
        return None
    return representations[0]["embedding"]


def _cosine_distance(embedding1, embedding2):
    dot = sum(value1 * value2 for value1, value2 in zip(embedding1, embedding2))
    norm1 = math.sqrt(sum(value * value for value in embedding1))
    norm2 = math.sqrt(sum(value * value for value in embedding2))
    if norm1 == 0.0 or norm2 == 0.0:
        return None
    return 1.0 - (dot / (norm1 * norm2))


def _ensure_embedding_cache(dataset_path):
    global _EMBEDDING_CACHE_SIGNATURE, _EMBEDDING_CACHE

    current_signature = _dataset_signature(dataset_path)
    if current_signature == _EMBEDDING_CACHE_SIGNATURE and _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE

    cache = []
    for current_root, _dir_names, file_names in os.walk(dataset_path):
        for file_name in sorted(file_names):
            if os.path.splitext(file_name)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            full_path = os.path.join(current_root, file_name)
            try:
                embedding = _represent_face(full_path, detector_backend=DETECTOR)
            except Exception:
                continue

            if embedding is None:
                continue

            identity_name = os.path.basename(os.path.dirname(full_path)).strip()
            cache.append(
                {
                    "name": identity_name,
                    "path": full_path,
                    "embedding": embedding,
                }
            )

    _EMBEDDING_CACHE_SIGNATURE = current_signature
    _EMBEDDING_CACHE = cache
    return _EMBEDDING_CACHE


def _runner_up_distance_from_matches(matches, best_name):
    for candidate_name, candidate_dist in matches:
        if candidate_name != best_name:
            return candidate_dist
    return None


def _finalize_match(best_name, best_dist, runner_up_dist):
    if not best_name or best_dist is None or best_dist > THRESHOLD:
        return "UNKNOWN", best_dist

    if runner_up_dist is not None and (runner_up_dist - best_dist) < MIN_DIFFERENT_IDENTITY_GAP:
        return "UNKNOWN", best_dist

    return best_name, best_dist


def _fallback_embedding_match(frame, dataset_path):
    try:
        query_embedding = _represent_face(frame, detector_backend=DETECTOR)
    except Exception:
        return "UNKNOWN", None

    if query_embedding is None:
        return "UNKNOWN", None

    cache = _ensure_embedding_cache(dataset_path)
    if not cache:
        return "UNKNOWN", None

    scored_matches = []
    for item in cache:
        dist = _cosine_distance(query_embedding, item["embedding"])
        if dist is None:
            continue
        scored_matches.append((item["name"], float(dist)))

    if not scored_matches:
        return "UNKNOWN", None

    scored_matches.sort(key=lambda item: item[1])
    best_name, best_dist = scored_matches[0]
    runner_up_dist = _runner_up_distance_from_matches(scored_matches, best_name)
    return _finalize_match(best_name, best_dist, runner_up_dist)


def find_best_match(frame, enforce_detection=True, detector_backend=DETECTOR):
    dataset_path = get_dataset_path()
    if not os.path.isdir(dataset_path):
        return "UNKNOWN", None

    try:
        results = DeepFace.find(
            img_path=frame,
            db_path=dataset_path,
            model_name=MODEL_NAME,
            detector_backend=detector_backend,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=enforce_detection,
            silent=True,
            refresh_database=_should_refresh_database(dataset_path),
        )
    except Exception:
        return _fallback_embedding_match(frame, dataset_path)

    if not results or results[0] is None or results[0].empty:
        return _fallback_embedding_match(frame, dataset_path)

    df = results[0].sort_values("distance", ascending=True)
    best = df.iloc[0]

    best_dist = float(best["distance"])
    identity_path = str(best["identity"])
    best_name = os.path.basename(os.path.dirname(identity_path)).strip()

    ranked_matches = []
    for row in df.itertuples():
        candidate_name = os.path.basename(os.path.dirname(str(row.identity))).strip()
        ranked_matches.append((candidate_name, float(row.distance)))

    runner_up_dist = _runner_up_distance_from_matches(ranked_matches, best_name)
    return _finalize_match(best_name, best_dist, runner_up_dist)


def verify_face(frame, enforce_detection=True):
    return find_best_match(frame, enforce_detection=enforce_detection, detector_backend=DETECTOR)
