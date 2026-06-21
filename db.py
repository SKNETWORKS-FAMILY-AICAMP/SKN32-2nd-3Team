import os
import hashlib
import secrets
import mysql.connector
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

USER_TABLE = "face_users"
LOGIN_LOG_TABLE = "face_login_logs"

DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
DB_USER = os.getenv("MYSQL_USER")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD")
DB_NAME = os.getenv("MYSQL_DATABASE", "mydb")

if not DB_USER or not DB_PASSWORD:
    raise RuntimeError(
        "MYSQL_USER / MYSQL_PASSWORD가 설정되지 않았습니다. "
        "프로젝트 루트에 .env 파일을 만들고 MYSQL_USER, MYSQL_PASSWORD를 지정해주세요."
    )

# 얼굴 임베딩 암호화 키 (환경변수로 주지 않으면 로컬에 키 파일을 만들어 재사용)
_KEY_FILE = Path(__file__).parent / "face_embedding.key"


def _load_or_create_key() -> bytes:
    env_key = os.getenv("FACE_EMBEDDING_KEY")
    if env_key:
        return env_key.encode()
    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes().strip()
    key = Fernet.generate_key()
    _KEY_FILE.write_bytes(key)
    return key


_FERNET = Fernet(_load_or_create_key())


def get_connection(database: Optional[str] = DB_NAME):
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        database=database, charset="utf8mb4", use_unicode=True,
        use_pure=True,  # 일부 Windows 환경에서 C 확장이 연결 단계에서 실패하는 문제 회피
    )


def init_db() -> None:
    """DB와 테이블을 초기화하고 필요한 컬럼이 없으면 추가합니다."""
    conn = get_connection(database=None)
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.close()

    conn = get_connection(database=DB_NAME)
    with conn.cursor() as cur:
        # 테이블 생성
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {USER_TABLE} (
                user_id VARCHAR(100) PRIMARY KEY,
                user_name VARCHAR(100) NOT NULL,
                role VARCHAR(20) NOT NULL DEFAULT 'viewer',
                password_salt VARCHAR(64) NOT NULL,
                password_hash VARCHAR(128) NOT NULL,
                face_embedding LONGBLOB,
                face_image_path VARCHAR(500),
                fail_count INT DEFAULT 0,
                lock_until DATETIME DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # 컬럼 존재 여부 체크 및 추가 (role, fail_count, lock_until 등)
        cols = ['role', 'fail_count', 'lock_until']
        for col in cols:
            cur.execute(f"SHOW COLUMNS FROM {USER_TABLE} LIKE '{col}'")
            if not cur.fetchone():
                col_def = "VARCHAR(20) NOT NULL DEFAULT 'viewer'" if col == 'role' else "INT DEFAULT 0" if col == 'fail_count' else "DATETIME DEFAULT NULL"
                cur.execute(f"ALTER TABLE {USER_TABLE} ADD COLUMN {col} {col_def}")

        # 기존 테이블이 face_embedding을 NOT NULL로 만든 경우 얼굴 미등록 계정 생성을 위해 NULL 허용으로 변경
        cur.execute(f"ALTER TABLE {USER_TABLE} MODIFY COLUMN face_embedding LONGBLOB NULL")

        # 로그인 시도 로그 테이블
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {LOGIN_LOG_TABLE} (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                user_name VARCHAR(100),
                success TINYINT(1) NOT NULL,
                similarity FLOAT,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

    conn.commit()
    conn.close()


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    if salt is None: salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return salt, digest.hex()


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    raw = embedding.astype(np.float32).tobytes()
    return _FERNET.encrypt(raw)


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    raw = _FERNET.decrypt(bytes(blob))
    embedding = np.frombuffer(raw, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding


def user_exists(user_id: str) -> bool:
    init_db()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {USER_TABLE} WHERE user_id = %s", (user_id,))
        count = cur.fetchone()[0]
    conn.close()
    return count > 0


def create_or_update_user(user_id, password, user_name, embedding=None, image_path="", role="viewer"):
    init_db()
    salt, password_hash = hash_password(password)
    embedding_blob = embedding_to_bytes(embedding) if embedding is not None else None
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {USER_TABLE} (user_id, user_name, role, password_salt, password_hash, face_embedding, face_image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                user_name=VALUES(user_name), role=VALUES(role), password_salt=VALUES(password_salt),
                password_hash=VALUES(password_hash),
                face_embedding=COALESCE(VALUES(face_embedding), face_embedding),
                face_image_path=VALUES(face_image_path)
        """, (user_id, user_name, role, salt, password_hash, embedding_blob, image_path))
    conn.commit()
    conn.close()


def set_user_face_embedding(user_id: str, embedding: np.ndarray) -> None:
    """기존 계정에 얼굴 임베딩만 등록/갱신합니다."""
    init_db()
    embedding_blob = embedding_to_bytes(embedding)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {USER_TABLE} SET face_embedding = %s WHERE user_id = %s", (embedding_blob, user_id))
    conn.commit()
    conn.close()


def user_has_face(user_id: str) -> bool:
    init_db()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT face_embedding FROM {USER_TABLE} WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
    conn.close()
    return bool(row and row[0])


def seed_demo_accounts() -> None:
    """README에 안내된 데모 계정을 DB에 1회 생성합니다(얼굴은 별도 등록 필요)."""
    demo_accounts = {
        "admin": ("admin123", "관리자", "admin"),
        "analyst1": ("analyst123", "분석가", "analyst"),
        "viewer1": ("viewer123", "뷰어", "viewer"),
    }
    for user_id, (password, user_name, role) in demo_accounts.items():
        if not user_exists(user_id):
            create_or_update_user(user_id, password, user_name, embedding=None, image_path="", role=role)


def verify_user_password(user_id: str, password: str) -> Tuple[bool, Optional[str], Optional[str], str]:
    init_db()
    conn = get_connection()
    with conn.cursor(dictionary=True) as cur:
        cur.execute(f"SELECT user_name, role, password_salt, password_hash FROM {USER_TABLE} WHERE user_id = %s",
                    (user_id,))
        row = cur.fetchone()
    conn.close()
    if not row: return False, None, None, "등록되지 않은 아이디입니다."
    _, input_hash = hash_password(password, salt=row["password_salt"])
    if not secrets.compare_digest(input_hash, row["password_hash"]):
        return False, None, None, "암호가 일치하지 않습니다."
    return True, row["user_name"], row["role"], "아이디와 암호 확인 완료."


def get_user_face_embedding(user_id: str) -> Optional[np.ndarray]:
    init_db()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT face_embedding FROM {USER_TABLE} WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
    conn.close()
    return bytes_to_embedding(row[0]) if row and row[0] else None


def get_all_users_with_embedding():
    init_db()
    conn = get_connection()
    with conn.cursor(dictionary=True) as cur:
        cur.execute(
            f"SELECT user_id, user_name, role, face_embedding FROM {USER_TABLE} WHERE face_embedding IS NOT NULL")
        rows = cur.fetchall()
    conn.close()
    return [{"user_id": r["user_id"], "user_name": r["user_name"], "role": r["role"],
             "embedding": bytes_to_embedding(r["face_embedding"])} for r in rows]


def get_all_users():
    """관리자 화면 등에서 쓸 전체 사용자 목록(얼굴 등록 여부 포함)."""
    init_db()
    conn = get_connection()
    with conn.cursor(dictionary=True) as cur:
        cur.execute(f"SELECT user_id, user_name, role, (face_embedding IS NOT NULL) AS has_face FROM {USER_TABLE}")
        rows = cur.fetchall()
    conn.close()
    return rows


def delete_user_face(user_id: str) -> None:
    """해당 계정의 얼굴 임베딩을 삭제합니다(개인정보 삭제 요청 대응)."""
    init_db()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {USER_TABLE} SET face_embedding = NULL WHERE user_id = %s", (user_id,))
    conn.commit()
    conn.close()


def get_user_lock_info(user_id: str) -> Optional[dict]:
    conn = get_connection()
    with conn.cursor(dictionary=True) as cur:
        cur.execute(f"SELECT fail_count, lock_until FROM {USER_TABLE} WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
    conn.close()
    return row


def clear_expired_lock(user_id: str) -> None:
    """잠금 시간이 이미 지난 계정의 fail_count/lock_until을 초기화합니다.
    상세 시도 이력은 face_login_logs에 그대로 남으므로 보안 추적에는 영향 없습니다."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {USER_TABLE} SET fail_count = 0, lock_until = NULL "
            f"WHERE user_id = %s AND lock_until IS NOT NULL AND lock_until <= NOW()",
            (user_id,),
        )
    conn.commit()
    conn.close()


def update_login_attempt(user_id: str, success: bool):
    conn = get_connection()
    with conn.cursor() as cur:
        if success:
            cur.execute(f"UPDATE {USER_TABLE} SET fail_count = 0, lock_until = NULL WHERE user_id = %s", (user_id,))
        else:
            cur.execute(f"SELECT fail_count FROM {USER_TABLE} WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row:
                new_fail = (row[0] or 0) + 1
                lock_time = datetime.now() + timedelta(minutes=5) if new_fail >= 5 else None
                cur.execute(f"UPDATE {USER_TABLE} SET fail_count = %s, lock_until = %s WHERE user_id = %s",
                            (new_fail, lock_time, user_id))
    conn.commit()
    conn.close()


def log_login_attempt(user_id: str, user_name: Optional[str], success: bool,
                      similarity: Optional[float] = None) -> None:
    init_db()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {LOGIN_LOG_TABLE} (user_id, user_name, success, similarity) VALUES (%s, %s, %s, %s)",
            (user_id, user_name, 1 if success else 0, similarity),
        )
    conn.commit()
    conn.close()


def get_login_logs(limit: int = 200):
    """관리자 조회용 얼굴 로그인 시도 로그(최신순)."""
    init_db()
    conn = get_connection()
    with conn.cursor(dictionary=True) as cur:
        cur.execute(
            f"SELECT user_id, user_name, success, similarity, attempted_at FROM {LOGIN_LOG_TABLE} ORDER BY attempted_at DESC LIMIT %s",
            (limit,))
        rows = cur.fetchall()
    conn.close()
    return rows