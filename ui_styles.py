"""앱의 CSS/HTML 디자인 요소 모음.

기능 로직(app.py)과 분리해서 디자인만 따로 관리합니다.
여기 있는 함수/상수는 화면에 무엇을 보여줄지 결정하는 로직을 갖지 않고,
순수하게 마크업/스타일 문자열만 다룹니다.
"""
import base64
import os

# ─── 전역(다크 테마) CSS ────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
    .stApp { background-color: #0F1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
</style>
"""

# ─── 로그인 화면 CSS ───────────────────────────────────────────────────────────
LOGIN_PAGE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

    header[data-testid="stHeader"] { display: none; }
    .block-container { padding: 0; max-width: 100%; }

    /* ── 배경: 좌(#58B4E8) / 우(흰색) ── */
    .login-bg-left {
        position: fixed; top: 0; left: 0;
        width: 50vw; height: 100vh;
        background: #58B4E8;
        z-index: 0;
    }
    .login-bg-right {
        position: fixed; top: 0; left: 50vw;
        width: 50vw; height: 100vh;
        background: #ffffff;
        z-index: 0;
    }

    /* ── 장식 원 ── */
    .deco-circle-1 {
        position: fixed; bottom: -80px; left: calc(50vw - 360px);
        width: 280px; height: 280px; border-radius: 50%;
        background: rgba(255,255,255,0.10);
        z-index: 0; pointer-events: none;
    }
    .deco-circle-2 {
        position: fixed; bottom: -40px; left: calc(50vw - 200px);
        width: 180px; height: 180px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
        z-index: 0; pointer-events: none;
    }
    .deco-circle-3 {
        position: fixed; top: -60px; left: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
        z-index: 0; pointer-events: none;
    }
    .deco-circle-4 {
        position: fixed; top: 40px; left: calc(50vw - 150px);
        width: 120px; height: 120px; border-radius: 50%;
        border: 1.5px solid rgba(255,255,255,0.22);
        z-index: 0; pointer-events: none;
    }
    .deco-wave {
        position: fixed; bottom: 0; left: 0;
        width: 50vw; height: 80px;
        z-index: 0; pointer-events: none;
    }

    /* ── 전체 폰트 ── */
    .st-key-login_shell,
    .st-key-login_shell input,
    .st-key-login_shell button,
    .st-key-login_shell label,
    .st-key-login_shell p,
    .st-key-login_shell span {
        font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    }

    /* ── 좌측 텍스트 흰색 ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) p,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) label,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) span {
        color: #ffffff;
    }

    /* ── 우측 텍스트 검정 ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) p,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) label,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) span,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) small {
        color: #1a1a1a;
    }

    /* ── 컬럼 레이아웃 ── */
    .st-key-login_shell div[data-testid="stHorizontalBlock"] {
        gap: 0;
        min-height: 100vh;
    }

    /* 좌측: 상단 여백 주고 중앙 정렬 */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) {
        display: flex;
        flex-direction: column;
        align-items: flex-start;       /* 좌측 정렬 */
        justify-content: center;
        padding: 40px 48px;
        min-height: 100vh;
        position: relative;
        z-index: 1;
    }

    /* 우측: 중앙 정렬 */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 64px;
        min-height: 100vh;
        position: relative;
        z-index: 1;
    }

    /* ── 좌측 전체 너비 요소 ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stMarkdown,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox {
        width: 100%;
    }

    /* ── 우측 폼 너비 ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stMarkdown,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stForm,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput {
        width: 100%;
        max-width: 520px;
    }

    /* ── 로고 (좌측 상단 고정) ── */
    .login-logo {
        position: fixed;
        top: 20px; left: 20px;
        height: 60px;
        z-index: 10;
        object-fit: contain;
    }

    /* ── 좌측 타이틀 ── */
    .login-left-title {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 10px;
        line-height: 1.25;
        font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    }
    .login-left-subtitle {
        color: rgba(255,255,255,0.88);
        font-size: 1.05rem;
        margin-bottom: 36px;
        font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    }

    /* ── 체크박스: 전체 너비, 세로 일렬, 투명→화이트 토글형 카드 ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox {
        background: transparent;
        border: 1.5px solid rgba(255,255,255,0.55);
        border-radius: 14px;
        padding: 0;
        margin-bottom: 14px;
        transition: all 0.18s ease;
        overflow: hidden;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox:hover {
        background: rgba(255,255,255,0.12);
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox:has(input:checked) {
        background: #ffffff;
        border-color: #ffffff;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }

    /* 라벨 전체를 길고 큰 클릭 영역으로 */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox label {
        justify-content: flex-start;
        align-items: center;
        gap: 14px;
        width: 100%;
        padding: 18px 24px;
        cursor: pointer;
    }

    /* 네이티브 체크박스 사각형(틱 표시)은 숨기고, 박스 색 변화로만 선택 표시 */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox label > span:first-child {
        display: none;
    }

    /* 라벨 텍스트 (이모지 + OTT명) */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox label p,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox label span {
        color: #ffffff;
        font-weight: 500;
        font-size: 1.15rem;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox:has(input:checked) label p,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(1) .stCheckbox:has(input:checked) label span {
        color: #1a7abf;
        font-weight: 700;
    }

    /* ── 우측 타이틀 ── */
    .login-right-title {
        color: #1a1a1a;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 4px;
        text-align: left;
        width: 100%;
        max-width: 520px;
        font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    }
    .login-right-subtitle {
        color: #666666;
        font-size: 0.95rem;
        margin-bottom: 36px;
        text-align: left;
        width: 100%;
        max-width: 520px;
        font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    }

    /* ── 입력창 label ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #1a1a1a;
        margin-bottom: 6px;
    }

    /* ── 입력창 box: 둥근 박스, 연한 베이지 배경 ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput input {
        background: #f5f0e8;
        border: 1.5px solid #e8e0d0;
        border-radius: 12px;
        padding: 14px 16px;
        color: #1a1a1a;
        font-size: 1rem;
        box-shadow: none;
        height: 52px;
        line-height: 1.2;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput input:focus {
        border-color: #58B4E8;
        background: #ffffff;
        box-shadow: none;
        outline: none;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stTextInput input::placeholder {
        color: #aaaaaa;
    }

    /* ── 경고 캡션 박스 ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) [data-testid="stCaptionContainer"] {
        background: #fdf3dc;
        border: 1.5px solid #c8960c;
        border-radius: 12px;
        padding: 14px 16px;
        width: 100%;
        max-width: 520px;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) [data-testid="stCaptionContainer"] p,
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) [data-testid="stCaptionContainer"] span {
        color: #7a5c00;
        font-size: 0.95rem;
    }

    /* ── 로그인 버튼 ── */
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stFormSubmitButton button {
        background: #58B4E8;
        color: #ffffff;
        border: none;
        border-radius: 14px;
        padding: 16px 0;
        font-weight: 700;
        font-size: 1.1rem;
        width: 100%;
        letter-spacing: 0.5px;
        transition: background 0.2s ease;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stFormSubmitButton button:hover {
        background: #2e86c1;
    }
    .st-key-login_shell div[data-testid="column"]:nth-of-type(2) .stFormSubmitButton button:disabled {
        background: #58B4E8;
        opacity: 1;
    }
</style>

<!-- 배경 분할 -->
<div class="login-bg-left"></div>
<div class="login-bg-right"></div>

<!-- 장식 원 -->
<div class="deco-circle-1"></div>
<div class="deco-circle-2"></div>
<div class="deco-circle-3"></div>
<div class="deco-circle-4"></div>

<!-- 하단 물결 SVG -->
<svg class="deco-wave" viewBox="0 0 800 80" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M0,40 C100,10 200,70 400,40 C560,10 680,60 800,35 L800,80 L0,80 Z" fill="rgba(255,255,255,0.07)"/>
  <path d="M0,55 C160,30 320,75 520,50 C640,35 740,60 800,50 L800,80 L0,80 Z" fill="rgba(255,255,255,0.06)"/>
</svg>
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
    return f'<img class="login-logo" src="{logo_uri}" alt="S-Board 로고" />'


def login_left_title_html(text: str) -> str:
    return f'<div class="login-left-title">{text}</div>'


def login_left_subtitle_html(text: str) -> str:
    return f'<div class="login-left-subtitle">{text}</div>'


def login_right_title_html(text: str) -> str:
    return f'<div class="login-right-title">{text}</div>'


def login_right_subtitle_html(text: str) -> str:
    return f'<div class="login-right-subtitle">{text}</div>'