"""
페이스 로그인 모듈
OpenCV + DeepFace 기반 얼굴 인식 인증 시스템
Streamlit 통합용
"""
import os
import cv2
import numpy as np
import json
import base64
import hashlib
from pathlib import Path
from datetime import datetime
import pickle

<<<<<<< HEAD
# Get project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB_PATH = os.path.join(BASE_DIR, 'face_data')
USER_DB_PATH = os.path.join(BASE_DIR, 'face_data', 'users.json')
=======
# [수정] 특정 PC 경로(C:/project_file/...)에 고정되어 있던 부분을
# 이 파일이 위치한 폴더 기준 상대 경로로 변경 (다른 PC/서버에서도 동작하도록)
FACE_DB_PATH = str(Path(__file__).resolve().parent / "face_data")
USER_DB_PATH = str(Path(FACE_DB_PATH) / "users.json")
>>>>>>> 74d49c4 (feat: 로컬 프로젝트 초기 커밋)
os.makedirs(FACE_DB_PATH, exist_ok=True)

# ─── 사용자 DB 관리 ───────────────────────────────────────────────────────────
def load_user_db():
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_user_db(db):
    with open(USER_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# ─── 얼굴 감지 (OpenCV Haar Cascade) ─────────────────────────────────────────
def detect_face_opencv(image_array):
    """OpenCV Haar Cascade로 얼굴 감지"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    return faces, gray

def extract_face_region(image_array, face_coords=None):
    """얼굴 영역 추출 및 정규화"""
    if face_coords is not None and len(face_coords) > 0:
        x, y, w, h = face_coords[0]
        # 여백 추가
        pad = int(min(w, h) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_array.shape[1], x + w + pad)
        y2 = min(image_array.shape[0], y + h + pad)
        face_region = image_array[y1:y2, x1:x2]
    else:
        face_region = image_array

    # 128x128으로 리사이즈
    face_resized = cv2.resize(face_region, (128, 128))
    return face_resized

def compute_face_embedding(face_image):
    """얼굴 임베딩 벡터 계산 (픽셀 기반 간단 특징)"""
    # 그레이스케일 변환
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image

    # 히스토그램 균등화
    gray = cv2.equalizeHist(gray)

    # LBP (Local Binary Pattern) 특징 추출
    lbp = compute_lbp(gray)

    # HOG 특징 (간단 버전)
    hog_features = compute_simple_hog(gray)

    # 픽셀 다운샘플링 특징
    small = cv2.resize(gray, (32, 32)).flatten().astype(np.float32) / 255.0

    # 결합
    embedding = np.concatenate([lbp, hog_features, small])
    return embedding

def compute_lbp(gray_image):
    """Local Binary Pattern 특징"""
    h, w = gray_image.shape
    lbp = np.zeros_like(gray_image)
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = gray_image[i, j]
            code = 0
            neighbors = [
                gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                gray_image[i+1, j-1], gray_image[i, j-1]
            ]
            for k, n in enumerate(neighbors):
                if n >= center:
                    code |= (1 << k)
            lbp[i, j] = code

    # LBP 히스토그램
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist

def compute_simple_hog(gray_image):
    """간단한 HOG 특징"""
    # 그래디언트 계산
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi % 180

    # 8x8 셀로 분할, 9방향 히스토그램
    cell_size = 16
    n_cells_x = gray_image.shape[1] // cell_size
    n_cells_y = gray_image.shape[0] // cell_size
    hog_features = []

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            cell_mag = magnitude[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            cell_ang = angle[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            hist, _ = np.histogram(cell_ang, bins=9, range=(0, 180), weights=cell_mag)
            hog_features.extend(hist)

    hog_arr = np.array(hog_features, dtype=np.float32)
    norm = np.linalg.norm(hog_arr) + 1e-7
    return hog_arr / norm

def cosine_similarity(a, b):
    """코사인 유사도"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ─── 사용자 등록 ──────────────────────────────────────────────────────────────
def register_user(username, role, image_array):
    """
    사용자 얼굴 등록
    Args:
        username: 사용자 이름
        role: 역할 (admin/analyst/viewer)
        image_array: numpy 배열 이미지
    Returns:
        (success, message)
    """
    db = load_user_db()

    if username in db:
        return False, f"'{username}' 사용자가 이미 등록되어 있습니다."

    # 얼굴 감지
    faces, gray = detect_face_opencv(image_array)
    if len(faces) == 0:
        return False, "얼굴을 감지할 수 없습니다. 정면을 향해 다시 시도해 주세요."
    if len(faces) > 1:
        return False, "여러 얼굴이 감지되었습니다. 한 명만 나오도록 해주세요."

    # 얼굴 임베딩 추출
    face_img = extract_face_region(image_array, faces)
    embedding = compute_face_embedding(face_img)

    # 얼굴 이미지 저장
    face_path = os.path.join(FACE_DB_PATH, f"{username}_face.jpg")
    cv2.imwrite(face_path, face_img)

    # 임베딩 저장
    emb_path = os.path.join(FACE_DB_PATH, f"{username}_embedding.pkl")
    with open(emb_path, 'wb') as f:
        pickle.dump(embedding, f)

    # 사용자 DB 업데이트
    db[username] = {
        'username': username,
        'role': role,
        'face_path': face_path,
        'embedding_path': emb_path,
        'registered_at': datetime.now().isoformat(),
        'last_login': None,
        'login_count': 0
    }
    save_user_db(db)

    return True, f"'{username}' 사용자 등록 완료!"

def authenticate_user(image_array, threshold=0.65):
    """
    얼굴 인증
    Args:
        image_array: 입력 이미지
        threshold: 유사도 임계값 (높을수록 엄격)
    Returns:
        (success, username, role, similarity, message)
    """
    db = load_user_db()
    if not db:
        return False, None, None, 0.0, "등록된 사용자가 없습니다. 먼저 얼굴을 등록해 주세요."

    # 얼굴 감지
    faces, gray = detect_face_opencv(image_array)
    if len(faces) == 0:
        return False, None, None, 0.0, "얼굴을 감지할 수 없습니다."

    # 입력 임베딩
    face_img = extract_face_region(image_array, faces)
    input_embedding = compute_face_embedding(face_img)

    # 모든 등록 사용자와 비교
    best_match = None
    best_similarity = 0.0

    for username, user_info in db.items():
        emb_path = user_info['embedding_path']
        if not os.path.exists(emb_path):
            continue
        with open(emb_path, 'rb') as f:
            stored_embedding = pickle.load(f)

        similarity = cosine_similarity(input_embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = username

    if best_match and best_similarity >= threshold:
        # 로그인 기록 업데이트
        db[best_match]['last_login'] = datetime.now().isoformat()
        db[best_match]['login_count'] = db[best_match].get('login_count', 0) + 1
        save_user_db(db)

        role = db[best_match]['role']
        return True, best_match, role, best_similarity, f"인증 성공! 안녕하세요, {best_match}님"
    else:
        return False, None, None, best_similarity, f"인증 실패 (유사도: {best_similarity:.2%}). 다시 시도해 주세요."

def get_all_users():
    """등록된 모든 사용자 목록 반환"""
    db = load_user_db()
    return db

<<<<<<< HEAD
=======
# ─── [추가] ID/PW 2차 인증(특정 계정 1:1 검증)용 함수 ────────────────────────────
def user_has_face(username):
    """해당 계정에 '실제로 촬영된' 얼굴이 등록되어 있는지 확인합니다.
    데모 계정(is_demo=True)은 가짜(랜덤) 임베딩이라 실제 얼굴 등록 전 상태로 간주합니다."""
    db = load_user_db()
    user = db.get(username)
    if not user:
        return False
    if user.get('is_demo'):
        return False
    emb_path = user.get('embedding_path', '')
    return bool(emb_path) and os.path.exists(emb_path)

def register_face_for_user(username, role, image_array):
    """ID/PW 1차 인증을 통과한 계정에 얼굴을 (최초 등록 또는 재등록) 합니다.
    기존 register_user()와 달리 이미 존재하는 계정이어도 덮어쓸 수 있습니다(데모 계정 → 실제 얼굴 전환용)."""
    faces, gray = detect_face_opencv(image_array)
    if len(faces) == 0:
        return False, "얼굴을 감지할 수 없습니다. 정면을 향해 다시 시도해 주세요."
    if len(faces) > 1:
        return False, "여러 얼굴이 감지되었습니다. 한 명만 나오도록 해주세요."

    face_img = extract_face_region(image_array, faces)
    embedding = compute_face_embedding(face_img)

    face_path = os.path.join(FACE_DB_PATH, f"{username}_face.jpg")
    cv2.imwrite(face_path, face_img)

    emb_path = os.path.join(FACE_DB_PATH, f"{username}_embedding.pkl")
    with open(emb_path, 'wb') as f:
        pickle.dump(embedding, f)

    db = load_user_db()
    prev_login_count = db.get(username, {}).get('login_count', 0)
    db[username] = {
        'username': username,
        'role': role,
        'face_path': face_path,
        'embedding_path': emb_path,
        'registered_at': datetime.now().isoformat(),
        'last_login': None,
        'login_count': prev_login_count,
        'is_demo': False
    }
    save_user_db(db)
    return True, f"'{username}' 얼굴 등록 완료!"

def verify_face_for_user(username, image_array, threshold=0.65):
    """특정 계정(username)에 등록된 얼굴과 1:1로만 비교합니다(2차 인증용).
    Returns:
        (success, similarity, message)
    """
    db = load_user_db()
    user = db.get(username)
    emb_path = user.get('embedding_path', '') if user else ''

    if not user or not emb_path or not os.path.exists(emb_path):
        return False, 0.0, "등록된 얼굴 정보가 없습니다. 얼굴 등록을 먼저 진행하세요."

    faces, gray = detect_face_opencv(image_array)
    if len(faces) == 0:
        return False, 0.0, "얼굴을 감지할 수 없습니다."
    if len(faces) > 1:
        return False, 0.0, "여러 얼굴이 감지되었습니다. 한 명만 나오도록 해주세요."

    face_img = extract_face_region(image_array, faces)
    input_embedding = compute_face_embedding(face_img)

    with open(emb_path, 'rb') as f:
        stored_embedding = pickle.load(f)

    similarity = cosine_similarity(input_embedding, stored_embedding)

    if similarity >= threshold:
        db[username]['last_login'] = datetime.now().isoformat()
        db[username]['login_count'] = db[username].get('login_count', 0) + 1
        save_user_db(db)
        return True, similarity, "얼굴 인증 성공!"
    else:
        return False, similarity, f"얼굴 불일치 (유사도: {similarity:.2%})"

>>>>>>> 74d49c4 (feat: 로컬 프로젝트 초기 커밋)
def delete_user(username):
    """사용자 삭제"""
    db = load_user_db()
    if username not in db:
        return False, f"'{username}' 사용자를 찾을 수 없습니다."

    user = db[username]
    # 파일 삭제
    for path_key in ['face_path', 'embedding_path']:
        if os.path.exists(user.get(path_key, '')):
            os.remove(user[path_key])

    del db[username]
    save_user_db(db)
    return True, f"'{username}' 사용자 삭제 완료"

# ─── 데모 사용자 생성 ─────────────────────────────────────────────────────────
def create_demo_users():
    """데모용 가상 사용자 생성 (실제 얼굴 없이 테스트용)"""
    db = load_user_db()
    demo_users = [
        {'username': 'admin', 'role': 'admin'},
        {'username': 'analyst1', 'role': 'analyst'},
        {'username': 'viewer1', 'role': 'viewer'},
    ]

    for user in demo_users:
        if user['username'] not in db:
            # 가상 임베딩 생성 (데모용)
            np.random.seed(hash(user['username']) % 2**32)
            fake_embedding = np.random.rand(256 + 576 + 1024).astype(np.float32)
            fake_embedding /= np.linalg.norm(fake_embedding)

            emb_path = os.path.join(FACE_DB_PATH, f"{user['username']}_embedding.pkl")
            with open(emb_path, 'wb') as f:
                pickle.dump(fake_embedding, f)

            db[user['username']] = {
                'username': user['username'],
                'role': user['role'],
                'face_path': '',
                'embedding_path': emb_path,
                'registered_at': datetime.now().isoformat(),
                'last_login': None,
                'login_count': 0,
                'is_demo': True
            }

    save_user_db(db)
    return True

if __name__ == '__main__':
    create_demo_users()
    print("데모 사용자 생성 완료")
    print(json.dumps(load_user_db(), ensure_ascii=False, indent=2))
