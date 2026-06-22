"""앱의 CSS/HTML 디자인 요소 모음.

기능 로직(app.py)과 분리해서 디자인만 따로 관리합니다.
여기 있는 함수/상수는 화면에 무엇을 보여줄지 결정하는 로직을 갖지 않고,
순수하게 마크업/스타일 문자열만 다룹니다.
"""
import base64
import os

# ─── 전역(화이트 테마) CSS ────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
    .stApp {
        background-color: #FFFFFF;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    [data-testid="stSidebar"] * {
        color: #1F2937;
    }
</style>
"""

# ─── 로그인 화면 CSS ───────────────────────────────────────────────────────────
LOGIN_PAGE_CSS = """
<style>
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding: 0 !important; max-width: 100% !important; }

    /* === 배경 셸: 화면을 절반씩 채우는 좌(블루)/우(화이트) 패널 === */
    .login-bg-left {
        position: fixed; top: 0; left: 0;
        width: 50vw; height: 100vh;
        background: linear-gradient(160deg, #1f6feb 0%, #58a6ff 100%);
    }
    .login-bg-right {
        position: fixed; top: 0; left: 50vw;
        width: 50vw; height: 100vh;
        background: #ffffff;
    }

    /* === 레이아웃: 좌/우 컬럼 정렬, 폭, 글자 크기 === */
    .st-key-login_shell, .st-key-login_shell input, .st-key-login_shell button {
        font-size: 1.2rem;
    }
    .st-key-login_shell div[data-testid="stHorizontalBlock"] {
        gap: 0;
        min-height: 100vh;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 60px 40px;
        padding-top: 38vh;
        min-height: 100vh;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 60px 50px;
        padding-top: 38vh;
        min-height: 100vh;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stMarkdown,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) div[data-testid="stHorizontalBlock"] {
        width: 100%;
        max-width: 460px;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stMarkdown,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stForm {
        width: 100%;
        max-width: 380px;
    }
    /* 화이트 패널(오른쪽)의 기본 텍스트는 검정 — 블루 패널(왼쪽)은 자체 흰색 규칙을 따로 둡니다. */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) p,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) label,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) span,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) small {
        color: #000000;
    }

    /* === 로고 === */
    .login-logo {
        position: fixed;
        top: 30px;
        left: 32px;
        height: 64px;
        z-index: 2;
    }

    /* === 왼쪽 타이틀 ("어떤 OTT 관리자이신가요?") === */
    .login-left-title {
        color: #ffffff;
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 36px;
    }

    /* === 체크박스 (OTT 서비스 선택) === */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox {
        background: rgba(255,255,255,0.12);
        border: 2px solid rgba(255,255,255,0.35);
        border-radius: 18px;
        padding: 22px 16px;
        text-align: center;
        transition: all 0.15s ease;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox:has(input:checked) {
        background: rgba(255,255,255,0.95);
        border-color: #ffffff;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox label {
        justify-content: center;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox label p {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox:has(input:checked) label p {
        color: #1f6feb;
    }

    /* === 오른쪽 타이틀 ("OTT Analytics 관제 시스템") === */
    .login-right-title {
        color: #1a1a1a;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 40px;
        text-align: center;
    }

    /* === 입력창 (아이디/비밀번호) === */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput label p {
        color: #000000;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput input {
        border: none;
        border-bottom: 2px solid #d0d7de;
        border-radius: 0;
        padding-left: 4px;
        background: transparent;
        color: #000000 !important;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput input::placeholder {
        color: #6b7280;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput input:focus {
        border-bottom: 2px solid #1f6feb;
        box-shadow: none;
    }

    /* === 로그인 버튼 === */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stButton button,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stFormSubmitButton button {
        background: #1f6feb !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 0;
        font-weight: 700;
    }
</style>
<div class="login-bg-left"></div>
<div class="login-bg-right"></div>
"""


def logo_data_uri(base_dir: str):
    """assets/logo.png를 base64 data URI로 변환합니다 (없으면 None)."""
    logo_path = os.path.join(base_dir, "assets", "logo.png")
    try:
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return None


def logo_img_html(logo_uri: str) -> str:
    return f'<img class="login-logo" src="{logo_uri}" />'


def login_left_title_html(text: str) -> str:
    return f'<div class="login-left-title">{text}</div>'


def login_right_title_html(text: str) -> str:
    return f'<div class="login-right-title">{text}</div>'
