"""InsightFace + ArcFace 기반 얼굴 등록/로그인 기능을 담당하는 모듈입니다."""

from pathlib import Path
from typing import Optional, Tuple
import secrets
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime

# DB 연동 모듈
from db import (
    create_or_update_user,
    set_user_face_embedding,
    user_has_face,
    get_user_face_embedding,
    get_all_users_with_embedding,
    user_exists,
    get_user_lock_info,
    update_login_attempt,
    clear_expired_lock,
    log_login_attempt,
    get_connection,  # DB 연결 함수
    USER_TABLE       # 사용자 테이블명
)

# 설정값
FACE_IMAGE_DIR = Path("registered_faces")
DEFAULT_SIMILARITY_THRESHOLD = 0.45
_FACE_APP: Optional[FaceAnalysis] = None

def get_face_app() -> FaceAnalysis:
    """InsightFace FaceAnalysis 객체를 초기화하고 반환합니다."""
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP
    _FACE_APP = FaceAnalysis(name="buffalo_l")
    _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _FACE_APP

def ensure_dirs() -> None:
    FACE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def read_camera_image(camera_file) -> Optional[np.ndarray]:
    if camera_file is None:
        return None
    file_bytes = np.frombuffer(camera_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def detect_largest_face(image_bgr: np.ndarray):
    app = get_face_app()
    if image_bgr is None or image_bgr.size == 0:
        return None, "이미지를 읽을 수 없습니다."

    image_rgb = bgr_to_rgb(image_bgr)
    faces = app.get(image_rgb)
    if len(faces) == 0:
        return None, "얼굴을 찾지 못했습니다."

    largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return largest_face, "얼굴 검출 성공"

def draw_face_contour(image_bgr: np.ndarray, face) -> np.ndarray:
    """검출된 얼굴에 윤곽선(landmark)을 그려 BGR 이미지로 반환합니다(촬영 결과 확인용, InsightFace 기반)."""
    annotated = image_bgr.copy()
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is not None:
        for (x, y) in landmarks.astype(int):
            cv2.circle(annotated, (x, y), 1, (0, 230, 60), -1)
    else:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 60), 2)
        if getattr(face, "kps", None) is not None:
            for (x, y) in face.kps.astype(int):
                cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
    return annotated

_HAAR_CASCADE: Optional[cv2.CascadeClassifier] = None

def get_fast_face_detector() -> cv2.CascadeClassifier:
    """실시간 미리보기용 경량 얼굴 검출기(Haar Cascade). OpenCV에 기본 내장돼 즉시 로딩됩니다."""
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _HAAR_CASCADE

def draw_fast_face_box(image_bgr: np.ndarray) -> np.ndarray:
    """실시간 미리보기에서 매 프레임 호출하기 위한 가벼운 얼굴 박스 표시(Haar Cascade)."""
    annotated = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = get_fast_face_detector()
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 230, 60), 2)
    return annotated

def extract_embedding(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    face, message = detect_largest_face(image_bgr)
    if face is None:
        return None, message

    embedding = face.embedding.astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None, "얼굴 특징 벡터를 생성하지 못했습니다."
    return embedding / norm, "얼굴 특징 추출 성공"

def save_registered_face_image(user_id: str, image_bgr: np.ndarray) -> Path:
    ensure_dirs()
    safe_user_id = "".join(ch for ch in user_id if ch.isalnum() or ch in ("_", "-"))
    image_path = FACE_IMAGE_DIR / f"{safe_user_id}.jpg"
    cv2.imwrite(str(image_path), image_bgr)
    return image_path

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def register_face(user_id: str, password: str, user_name: str, image_bgr: np.ndarray, role: str = "viewer") -> Tuple[bool, str]:
    if not all([user_id.strip(), password.strip(), user_name.strip()]):
        return False, "입력값을 확인하세요."

    embedding, message = extract_embedding(image_bgr)
    if embedding is None: return False, message

    image_path = save_registered_face_image(user_id, image_bgr)
    create_or_update_user(user_id.strip(), password, user_name.strip(), embedding, "", role)
    image_path.unlink(missing_ok=True)
    return True, f"{user_id} 등록 완료."

def register_user(user_name: str, role: str, image_bgr: np.ndarray) -> Tuple[bool, str]:
    user_name = user_name.strip()
    safe_user_id = "".join(ch for ch in user_name if ch.isalnum() or ch in ("_", "-"))
    user_id = safe_user_id if not user_exists(safe_user_id) else f"{safe_user_id}_{secrets.token_hex(2)}"

    embedding, message = extract_embedding(image_bgr)
    if embedding is None: return False, message

    create_or_update_user(user_id, secrets.token_hex(16), user_name, embedding, "", role)
    return True, f"{user_name}님 등록 완료 (ID: {user_id})"

def register_face_for_existing_user(user_id: str, user_name: str, image_bgr: np.ndarray) -> Tuple[bool, str]:
    """이미 ID/PW가 있는 계정에 얼굴 임베딩만 등록(2차 인증용)합니다."""
    embedding, message = extract_embedding(image_bgr)
    if embedding is None:
        return False, message
    set_user_face_embedding(user_id, embedding)
    return True, "얼굴 등록 완료."

def verify_face_for_user(user_id: str, image_bgr: np.ndarray, user_name: Optional[str] = None, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
    locked, lock_message = check_lock_status(user_id)
    if locked:
        log_login_attempt(user_id, user_name, success=False)
        return False, 0.0, lock_message

    registered_embedding = get_user_face_embedding(user_id)
    login_embedding, message = extract_embedding(image_bgr)

    if registered_embedding is None:
        return False, 0.0, "등록된 얼굴 정보가 없습니다. 얼굴 등록을 먼저 진행하세요."

    if login_embedding is None:
        update_login_attempt(user_id, success=False)
        log_login_attempt(user_id, user_name, success=False)
        return False, 0.0, message

    score = cosine_similarity(login_embedding, registered_embedding)
    if score >= threshold:
        update_login_attempt(user_id, success=True)
        log_login_attempt(user_id, user_name, success=True, similarity=score)
        return True, score, "인증 성공!"
    else:
        update_login_attempt(user_id, success=False)
        log_login_attempt(user_id, user_name, success=False, similarity=score)
        return False, score, "얼굴 불일치"

def authenticate_user(image_bgr: np.ndarray, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
    login_embedding, message = extract_embedding(image_bgr)
    if login_embedding is None: return False, None, None, 0.0, message

    users = [u for u in get_all_users_with_embedding() if u["embedding"] is not None]
    best_user, best_score = None, -1.0

    for user in users:
        score = cosine_similarity(login_embedding, user["embedding"])
        if score > best_score:
            best_score, best_user = score, user

    if best_user is None or best_score < threshold:
        return False, None, None, best_score, "일치하는 얼굴 없음."

    locked, lock_message = check_lock_status(best_user['user_id'])
    if locked:
        log_login_attempt(best_user['user_id'], best_user['user_name'], success=False)
        return False, None, None, best_score, lock_message

    update_login_attempt(best_user['user_id'], success=True)
    log_login_attempt(best_user['user_id'], best_user['user_name'], success=True, similarity=best_score)
    return True, best_user["user_name"], best_user["role"], best_score, "인증 성공."

def is_user_locked(user_id: str) -> bool:
    """계정 잠금 여부 확인"""
    locked, _ = check_lock_status(user_id)
    return locked

def check_lock_status(user_id: str) -> Tuple[bool, str]:
    """계정 잠금 여부와 남은 잠금 시간을 함께 반환합니다. 잠금 시간이 지났으면 카운터를 초기화합니다."""
    try:
        conn = get_connection()
        with conn.cursor(dictionary=True) as cur:
            cur.execute(f"SELECT lock_until FROM {USER_TABLE} WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
        conn.close()

        if row and row["lock_until"]:
            if row["lock_until"] > datetime.now():
                remaining_minutes = max(1, int((row["lock_until"] - datetime.now()).total_seconds() // 60) + 1)
                return True, f"⚠️ 로그인 시도 초과로 계정이 잠겼습니다. {remaining_minutes}분 후 다시 시도하세요."
            clear_expired_lock(user_id)
        return False, ""
    except Exception:
        return False, ""