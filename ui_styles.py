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
def get_login_page_css(bg_uri: str = None) -> str:
    """배경 이미지 URI를 받아서 동적으로 CSS를 생성합니다."""
    bg_style = f"background: url('{bg_uri}');" if bg_uri else "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;700&display=swap');

    header[data-testid="stHeader"] {{ display: none; }}
    .block-container {{ padding: 0; max-width: 100%; }}

    /* ── 전체 배경: 이미지 사용 ── */
    .login-bg {{
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        {bg_style}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        z-index: 0;
    }}

    /* ── 장식 요소 ── */
    .deco-blob-1 {{
        position: fixed; top: -100px; right: -100px;
        width: 400px; height: 400px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        filter: blur(80px);
        z-index: 0;
    }}
    .deco-blob-2 {{
        position: fixed; bottom: -100px; left: -100px;
        width: 350px; height: 350px;
        background: rgba(255, 127, 80, 0.2);
        border-radius: 50%;
        filter: blur(60px);
        z-index: 0;
    }}

    /* ── 로그인 컨테이너: 단일 카드 + 스냅 스크롤 ── */
    # .st-key-login_shell > div {{
    #     display: flex;
    #     flex-direction: column;
    #     align-items: center;
    #     justify-content: center;
    #     min-height: 100vh;
    #     padding: 40px;
    #     scroll-snap-type: y mandatory;
    #     overflow-y: auto;
    # }}

    /* ── 메인 카드 ── */
    .st-key-login_shell .stContainer {{
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        padding: 40px 50px;
        max-width: 480px;
        width: 100%;
        position: relative;
        z-index: 1;
        scroll-snap-align: start;
        scroll-margin-top: 40px;
    }}

    /* ── 내부 컨테이너 그룹화 ── */
    .st-key-login_shell .stContainer > div {{
        display: flex;
        flex-direction: column;
        gap: 24px;
    }}

    /* ── 전체 폰트 ── */
    .st-key-login_shell,
    .st-key-login_shell input,
    .st-key-login_shell button,
    .st-key-login_shell label,
    .st-key-login_shell p,
    .st-key-login_shell span {{
        font-family: 'Poppins', 'Noto Sans KR', sans-serif;
    }}

    /* ── 로고: 작게 표시 ── */
    .login-logo {{
        height: 150px;
        width: auto;
        object-fit: contain;
        margin: 0 auto 30px auto;
        display: block;
    }}

    /* ── 타이틀 섹션 ── */
    .login-title {{
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 4px;
        line-height: 1.4;
        text-align: center;
    }}

    /* ── 섹션 카드 스타일 ── */
    .section-card {{
        background: #f8f9fa;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e9ecef;
    }}

    /* ── 체크박스 섹션 ── */
    .st-key-login_shell .stHorizontalBlock {{
        margin-top: 8px;
    }}

    /* ── 체크박스 컨테이너: 가로 나열 ── */
    .st-key-login_shell .stCheckbox {{
        background: #f9fafb;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 0;
        transition: all 0.2s ease;
        flex: 1;
        margin: 0 8px;
    }}
    .st-key-login_shell .stCheckbox:hover {{
        border-color: #667eea;
        background: #f3f4f6;
    }}
    .st-key-login_shell .stCheckbox:has(input:checked) {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }}

    .st-key-login_shell .stCheckbox label {{
        justify-content: center;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 12px;
        cursor: pointer;
    }}

    .st-key-login_shell .stCheckbox label > span:first-child {{
        display: none;
    }}

    .st-key-login_shell .stCheckbox label p,
    .st-key-login_shell .stCheckbox label span {{
        color: #374151;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    .st-key-login_shell .stCheckbox:has(input:checked) label p,
    .st-key-login_shell .stCheckbox:has(input:checked) label span {{
        color: #ffffff;
    }}

    /* ── 체크박스 가로 컨테이너 ── */
    .checkbox-horizontal-container {{
        display: flex;
        gap: 8px;
        margin-bottom: 30px;
        width: 100%;
    }}

    /* ── 입력창 스타일 ── */
    .st-key-login_shell .stTextInput label {{
        font-size: 0.85rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 6px;
        text-align: left;
        width: 100%;
    }}

    .st-key-login_shell .stTextInput input {{
        background: #ffffff !important;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 18px;
        color: #1f2937 !important;
        font-size: 0.95rem;
        box-shadow: none;
        # height: 50px;
        transition: all 0.2s ease;
    }}
    .st-key-login_shell .stTextInput input:focus {{
        border-color: #667eea;
        background: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }}
    .st-key-login_shell .stTextInput input::placeholder {{
        color: #9ca3af;
    }}

    /* ── 경고 캡션 ── */
    .st-key-login_shell [data-testid="stCaptionContainer"] {{
        background: #fef3c7;
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 12px 16px;
        width: 100%;
        margin-top: 12px;
    }}
    .st-key-login_shell [data-testid="stCaptionContainer"] p,
    .st-key-login_shell [data-testid="stCaptionContainer"] span {{
        color: #92400e;
        font-size: 0.85rem;
        font-weight: 500;
    }}

    /* ── 로그인 버튼: 주황색 ── */
    .st-key-login_shell .stFormSubmitButton button {{
        background: linear-gradient(135deg, #FF7F50 0%, #FF6347 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 16px 0;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(255, 127, 80, 0.4);
        margin-top: 20px;
    }}
    .st-key-login_shell .stFormSubmitButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 127, 80, 0.5);
        background: linear-gradient(135deg, #FF6347 0%, #FF4500 100%);
    }}
    .st-key-login_shell .stFormSubmitButton button:disabled {{
        background: #d1d5db;
        box-shadow: none;
        cursor: not-allowed;
        transform: none;
    }}

    /* ── 폼 너비 ── */
    .st-key-login_shell .stForm,
    .st-key-login_shell .stTextInput {{
        width: 100%;
    }}
</style>

<!-- 배경 -->
<div class="login-bg"></div>
<div class="deco-blob-1"></div>
<div class="deco-blob-2"></div>
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


def bg_image_data_uri(base_dir: str):
    """assets/bg_wave.png를 base64 data URI로 변환합니다 (없으면 None)."""
    bg_path = os.path.join(base_dir, "assets", "bg_wave.png")
    try:
        with open(bg_path, "rb") as f:
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
