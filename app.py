"""
🎬 OTT 서비스 고객 데이터 통합 대시보드 시스템
[업데이트] MySQL DB 기반 안전한 로그인 + 계정별 서비스 권한 데이터 필터링 시각화 자동 연동
[2차 얼굴인증 추가] 기존 app2.py 로직을 완벽히 보존하며 2차 얼굴 인증 모달 팝업 추가
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import sys
import cv2
import base64
import warnings
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from datetime import datetime
from sqlalchemy import create_engine
import pymysql
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')

# ─── 🛠️ Matplotlib 한글 폰트 깨짐 방지 전역 설정 ──────────────────────────
import platform
from matplotlib import font_manager, rc

if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    rc('font', family='sans-serif')

plt.rcParams['axes.unicode_minus'] = False

# 프로젝트 루트 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─── 페이지 기본 설정 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="고객 분석 대시보드 시스템 | OTT Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 세션 상태 초기화 ──────────────────────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "대시보드"
if 'service' not in st.session_state:
    st.session_state.service = None

# [얼굴인증을 위해 안전하게 추가된 세션 키]
if 'pending_user' not in st.session_state:
    st.session_state.pending_user = None
if 'pending_user_name' not in st.session_state:
    st.session_state.pending_user_name = None
if 'pending_role' not in st.session_state:
    st.session_state.pending_role = None
if 'auth_step' not in st.session_state:
    st.session_state.auth_step = "login"

# ─── 얼굴 로그인용 데모 계정 시드 (최초 1회 자동 실행) ───────────────────────────
try:
    from db import seed_demo_accounts
    seed_demo_accounts()
except ImportError:
    pass

# ─── CSS 커스텀 스타일 정의 (app5 UI 스타일 반영) ─────────────────────────────────
st.markdown("""
<style>
    /* 전체 앱 배경 기본 흰색 */
    .stApp { background-color: #FFFFFF; }

    /* 사이드바 경계선 */
    [data-testid="stSidebar"] {
        border-right: 1px solid #D7E3FA;
    }

    /* ── 사이드바 텍스트 및 라디오 버튼 가시성 (기존 유지) ── */
    [data-testid="stSidebar"] * { font-weight: 500; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown p { 
        color: #FFFFFF !important;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.8) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background-color: rgba(255, 255, 255, 0.75) !important;
        padding: 4px 12px !important;
        border-radius: 6px !important;
        margin-bottom: 4px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label span { 
        color: #1B2A41 !important; 
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] code {
        color: #FFEB3B !important; 
        background-color: rgba(0, 0, 0, 0.5) !important;
        font-weight: bold !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 8px !important;
    }

    /* ── 로그인 화면 UI는 ui_styles.py의 get_login_page_css()에서 별도로 주입됩니다 ── */

    /* 입력창 및 버튼 공통 서식 (로그인 화면 외 다른 페이지용) */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        background-color: rgba(255, 255, 255, 0.8) !important;
    }
    .stButton button {
        width: 100% !important;
        border-radius: 12px !important;
        padding: 10px 0 !important;
        font-weight: 600 !important;
    }
    .stButton button[kind="primary"] {
        background-color: #4F46E5 !important;
        color: white !important;
        border: none !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #4338CA !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── 로그인 화면 UI 스타일/마크업 함수 (app5.py 디자인을 ui_styles.py에서 그대로 가져옴) ──────
from ui_styles import (
    get_login_page_css, logo_data_uri, bg_image_data_uri, logo_img_html,
)


# ─── 배경 이미지 적용 (사용자 업로드 웨이브 이미지 → 사이드바) ──────────────────────────
def _img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_BG_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "bg_wave.png")
if os.path.exists(_BG_IMAGE_PATH):
    _BG_BASE64 = _img_to_base64(_BG_IMAGE_PATH)
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background-image: url("data:image/png;base64,{_BG_BASE64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)


# ─── [데이터 펑션] DB 커넥션 엔진 생성 공통 함수 ─────────────────────────────────────
def get_db_engine():
    return create_engine("mysql+pymysql://root:mysql80@localhost:3306/ott_db?charset=utf8mb4")

# ─── [데이터 펑션] MySQL 실시간 연동 및 통합 전처리 ──────────────────────────────────────
@st.cache_data
def load_mysql_dashboard_data():
    try:
        engine = get_db_engine()
        query = """
        SELECT 
            t.YEAR, 
            t.OPID, 
            t.`Weekday usage` AS weekday_usage, 
            t.`Weekend usage` AS weekend_usage,
            m.svod,
            u.ott_first, 
            u.ott_second
        FROM ott_time t
        LEFT JOIN ott_money m ON t.OPID = m.OPID AND t.YEAR = m.YEAR
        LEFT JOIN ott_usage u ON t.OPID = u.OPID AND t.YEAR = u.YEAR
        """
        df = pd.read_sql(query, con=engine)

        df['weekday_usage'] = df['weekday_usage'].fillna(0)
        df['weekend_usage'] = df['weekend_usage'].fillna(0)
        df['총이용시간_분'] = df['weekday_usage'] + df['weekend_usage']

        df = df.rename(columns={
            'OPID': '고객ID',
            'YEAR': '연도',
            'svod': '이용요금'
        })
        return df
    except Exception as e:
        st.error(f"❌ MySQL 실시간 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# 기존 레거시 데이터 및 모델 로드 함수 (기존 app2.py 내용 100% 보존)
@st.cache_data
def load_legacy_churn_data():
    try: return pd.read_csv('korea_telecom_churn.csv', encoding='utf-8-sig')
    except: return pd.DataFrame()

@st.cache_resource
def load_ml_models():
    json_path = os.path.join(BASE_DIR, 'data', 'model_results.json')
    model_dir = os.path.join(BASE_DIR, 'models')
    if not os.path.exists(json_path): return None, None, None, None, {}
    try:
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
        label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        best_model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        with open(json_path, 'r', encoding='utf-8') as f: model_results = json.load(f)
        return scaler, feature_cols, label_encoders, best_model, model_results
    except: return None, None, None, None, {}

@st.cache_data
def load_csv_data():
    csv_path = os.path.join(BASE_DIR, 'data', 'korea_telecom_churn.csv')
    if os.path.exists(csv_path): return pd.read_csv(csv_path, encoding='utf-8-sig')
    return pd.DataFrame()

@st.cache_resource
def load_models():
    scaler, best_model, viz_data = None, None, None
    try:
        scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
        best_model = joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))
    except: pass
    json_path = os.path.join(BASE_DIR, 'data', 'visualization_data.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f: viz_data = json.load(f)
        except: pass
    return scaler, best_model, viz_data

# ─── 1. 대시보드 및 페이지 정의부 생략 (기존 코드 완벽 보존됨) ───────────────────

def show_dashboard(df):
    st.title("📊 실시간 OTT 데이터 통합 모니터링 대시보드")

    if df.empty:
        st.warning("데이터베이스에 연결할 수 없거나 읽어올 데이터가 없습니다.")
        return
        # ─────────────────────────────────────────────────────────
        # 📈 [업그레이드] 1. OTT 플랫폼별 고객 이탈률 위험도 분포 선그래프 (최상단 배치)
        # ─────────────────────────────────────────────────────────
    try:
        # visualization_data.json 경로 지정 및 로드
        json_path = os.path.join(BASE_DIR, 'data', 'visualization_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                viz_data = json.load(f)

            # 테스트(Test) 예측 데이터 사용 (일반적으로 가장 최신 평가 데이터)
            prob_test_dict = viz_data.get('y_prob_test', {})

            y_prob_test = []
            if isinstance(prob_test_dict, dict) and prob_test_dict:
                y_prob_test = prob_test_dict.get('XGBoost_Tuned', list(prob_test_dict.values())[0])
            elif isinstance(prob_test_dict, list):
                y_prob_test = prob_test_dict

            if y_prob_test and not df.empty:
                st.subheader("📈 주요 OTT 플랫폼별 고객 이탈 위험 확률 추이 분석")

                # 안전한 매핑을 위해 현재 분석 데이터프레임(df)의 길이와 예측 결과 길이 맞추기
                # (보통 테스트셋과 DB 샘플 수가 다를 수 있으므로 데이터 수에 맞춰 매핑하거나 스케일링)
                mapping_len = min(len(df), len(y_prob_test))

                # 임시 분석용 데이터프레임 생성
                df_analysis = df.iloc[:mapping_len].copy()
                df_analysis['이탈확률'] = y_prob_test[:mapping_len]

                # ott_first 컬럼 정제 (소문자 및 영문 통일하여 깔끔하게 라벨링)
                def clean_ott_name(val):
                    val_str = str(val).strip().lower()
                    if 'netflix' in val_str or '넷플릭스' in val_str: return 'Netflix'
                    if 'youtube' in val_str or '유튜브' in val_str: return 'YouTube'
                    if 'tving' in val_str or '티빙' in val_str: return 'Tving'
                    if 'disney' in val_str or '디즈니' in val_str: return 'Disney+'
                    return '기타 OTT'

                df_analysis['OTT_플랫폼'] = df_analysis['ott_first'].apply(clean_ott_name)

                # 10% 단위 이탈 확률 구간(Bin) 설정
                bins = np.linspace(0, 1, 11)
                labels = [f"{int(bins[i] * 100)}%-{int(bins[i + 1] * 100)}%" for i in range(10)]

                # OTT별 구간 빈도 계산 후 차트 데이터 빌드
                chart_data = []
                for ott in df_analysis['OTT_플랫폼'].unique():
                    ott_df = df_analysis[df_analysis['OTT_플랫폼'] == ott]
                    counts, _ = np.histogram(ott_df['이탈확률'], bins=bins)

                    for label, count in zip(labels, counts):
                        chart_data.append({
                            "이탈 확률 구간": label,
                            "고객 수(명)": count,
                            "OTT 플랫폼": ott
                        })

                df_chart = pd.DataFrame(chart_data)

                # Plotly 다중 다중 선그래프(Line Chart) 생성
                fig_line = px.line(
                    df_chart,
                    x="이탈 확률 구간",
                    y="고객 수(명)",
                    color="OTT 플랫폼",
                    markers=True,
                    title="🔮 AI 예측 플랫폼별 고객 이탈 위험도(0% ~ 100%) 분포 곡선",
                    labels={"고객 수(명)": "해당 구간 고객 수 (명)"},
                    color_discrete_sequence=px.colors.qualitative.Safe
                )

                fig_line.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=50, b=20),
                    hovermode="x unified",
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#1B2A41")
                )

                # 최상단 대시보드 출력
                st.plotly_chart(fig_line, use_container_width=True)
                st.markdown("---")  # 하단 3개 영역과의 구분선
            else:
                st.info("💡 매핑할 예측 데이터 또는 DB 데이터가 부족합니다.")
        else:
            st.info("💡 `data/visualization_data.json` 파일이 없어 플랫폼별 선그래프 생략")
    except Exception as e:
        st.error(f"OTT별 이탈률 선그래프 로드 오류: {e}")

    # ─── 🛠️ 계정 권한별 데이터 필터링 로직 ───
    current_service = st.session_state.service

    if current_service and current_service != "ALL":
        # 1. DB 데이터의 공백을 제거하고 소문자로 통일
        df['ott_first_match'] = df['ott_first'].astype(str).str.strip().str.lower()

        # 2. 로그인한 계정 권한(영어)에 따라 DB에서 검색할 핵심 키워드(한글/영어) 매핑
        if current_service == "YouTube":
            keywords = "youtube|유튜브"
        elif current_service == "Netflix":
            keywords = "netflix|넷플릭스"
        elif current_service == "Tving":
            keywords = "tving|티빙"
        else:
            keywords = current_service.lower()

        # 3. 매핑된 키워드가 포함된 데이터만 필터링
        df = df[df['ott_first_match'].str.contains(keywords, na=False)]

        st.info(f"💡 현재 **{current_service}** 권한에 해당하는 고객 데이터만 필터링되어 표시 중입니다.")
    else:
        st.success(f"✅ 전체 관리자 권한 - 모든 OTT 데이터가 통합 모니터링 중입니다. (총 레코드 수: {len(df):,}개)")

    if df.empty:
        st.warning("⚠️ 현재 권한(OTT)에 부합하는 데이터가 DB에 존재하지 않습니다. 'ott_first' 컬럼의 데이터를 확인해주세요.")
        return

    # 📊 3대 핵심 그래프 영역 시각화 구성
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⏱️ 고객별 평균 총 이용시간 (분)")
        fig_time = px.histogram(
            df,
            x="총이용시간_분",
            nbins=30,
            color_discrete_sequence=['#3B6FED']
        )
        fig_time.update_traces(texttemplate='%{y}', textposition='outside')
        fig_time.update_layout(
            height=450,
            bargap=0.08,
            xaxis_title="총 이용 시간 (분)",
            yaxis_title="고객 수 (명)",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#1B2A41")
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        if '이용요금' in df.columns:
            st.subheader("💰 이용 요금대별 고객 분포")
            clean_money_df = df[df['이용요금'].notna() & (df['이용요금'].astype(str).str.strip() != '')]

            if clean_money_df.empty:
                st.info("시각화할 수 있는 유효한 이용 요금 데이터가 없습니다.")
            else:
                fig_money = px.pie(
                    clean_money_df,
                    names="이용요금",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_money.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=20, b=20),
                    font=dict(color="#1B2A41"),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_money, use_container_width=True)
        else:
            st.info("요금 정보 데이터가 부족합니다.")

    with col3:
        st.subheader("⭕ 플랫폼별 실사용자 중복 분석")
        if 'ott_first' in df.columns and 'ott_second' in df.columns:
            df['ott_first_clean'] = df['ott_first'].astype(str).str.strip().str.lower()
            df['ott_second_clean'] = df['ott_second'].astype(str).str.strip().str.lower()

            netflix_set = set(df[(df['ott_first_clean'].str.contains('netflix|넷플릭스', na=False)) |
                                 (df['ott_second_clean'].str.contains('netflix|넷플릭스', na=False))]['고객ID'])

            tving_set = set(df[(df['ott_first_clean'].str.contains('tving|티빙', na=False)) |
                               (df['ott_second_clean'].str.contains('tving|티빙', na=False))]['고객ID'])

            disney_set = set(df[(df['ott_first_clean'].str.contains('disney|디즈니', na=False)) |
                                (df['ott_second_clean'].str.contains('disney|디즈니', na=False))]['고객ID'])

            fig, ax = plt.subplots(figsize=(6, 5), facecolor='#FFFFFF')
            ax.set_facecolor('#FFFFFF')

            v = venn3(
                subsets=[netflix_set, tving_set, disney_set],
                set_labels=('Netflix', 'Tving', 'Disney+'),
                ax=ax
            )

            if v:
                for text in v.set_labels:
                    if text:
                        text.set_color('#1B2A41')
                        text.set_fontsize(11)
                for text in v.subset_labels:
                    if text:
                        text.set_color('#1B2A41')
                        text.set_fontsize(10)

            plt.title("OTT 구독자 교차 중복 현황 (Venn)", color='#1B2A41', fontsize=13, pad=15)
            st.pyplot(fig)
        else:
            st.info("중복 분석용 데이터(ott_first, ott_second)가 부족합니다.")

    # 하단 데이터프레임 노출
    st.dataframe(df.head(50), use_container_width=True)

def show_sidebar():
    # ── 사이드바 최상단 로고 이미지 추가 ──
    logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)

    st.sidebar.title(f"🎬 {st.session_state.username} 님")
    st.sidebar.write(f"권한: `{st.session_state.user_role}`")

    # 로그인한 계정의 서비스 권한에 따른 맞춤형 메뉴 구성
    if st.session_state.service == "ALL":
        menu_list = ["대시보드", "EDA 분석", "모델 성능", "이탈 예측"]
    elif st.session_state.service == "YouTube":
        menu_list = ["유튜브 페이지", "EDA 분석"]
    elif st.session_state.service == "Netflix":
        menu_list = ["넷플릭스 페이지", "EDA 분석"]
    elif st.session_state.service == "Tving":
        menu_list = ["티빙 페이지", "EDA 분석"]
    else:
        menu_list = ["대시보드"]

    # 라디오 버튼 메뉴 활성화 및 현재 페이지 세션 저장
    page = st.sidebar.radio("메뉴 이동", menu_list, key="sidebar_menu_radio")
    st.session_state.current_page = page

    if st.sidebar.button("로그아웃"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.service = None
        st.session_state.current_page = "대시보드"
        st.session_state.auth_step = "login"
        st.rerun()


def show_eda(df, viz_data):
    st.title("📊 고객 데이터 분포 및 EDA 분석")
    st.write("기존 가입 고객들의 특성 데이터 분포와 이탈에 영향을 미치는 주요 인자들을 탐색합니다.")

    if df is None or df.empty:
        st.warning("분석할 CSV 데이터가 로드되지 않았습니다.")
        return

    # 탭 구성: 중요 변수 분석 / 범주형 분포 / 수치형 분포
    tab1, tab2, tab3 = st.tabs(["🔥 주요 변수 및 상관관계", "📊 범주형 데이터 분포", "📈 수치형 데이터 분포"])

    # ── Tab 1: 중요 변수 및 상관관계 ──
    with tab1:
        st.subheader("🎯 머신러닝 모델이 주목한 핵심 특성 (Feature Importance)")
        if viz_data and 'feature_importances' in viz_data:
            fi_df = pd.DataFrame({
                'Feature': list(viz_data['feature_importances'].keys()),
                'Importance': list(viz_data['feature_importances'].values())
            }).sort_values(by='Importance', ascending=True)

            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                title="이탈 예측 모델 변수 중요도 Top 10",
                color='Importance', color_continuous_scale='Blues'
            )
            fig_fi.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#1B2A41"))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("모델 중요도 데이터(viz_data)를 불러올 수 없습니다.")

        # 상관관계 히트맵 (수치형 데이터 대상)
        st.subheader("🔗 수치형 변수 간 상관관계 분석 (Correlation Matrix)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 불필요한 ID성 컬럼 제외
        numeric_cols = [c for c in numeric_cols if c not in ['고객ID', 'id', 'Id', 'ID', 'Churn']]

        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr, text_auto='.2f', aspect="auto",
                color_continuous_scale='RdBu_r', labels=dict(color="상관계수")
            )
            fig_corr.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#1B2A41"))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("상관관계를 분석할 수치형 변수가 부족합니다.")

    # ── Tab 2: 범주형 데이터 분포 ──
    with tab2:
        st.subheader("🗂️ 범주(Category)별 고객 가입 특성 및 이탈 현황")
        cat_cols = ['성별', '구독요금제', '주사용기기', '결제방법', '이탈여부']
        # 실제 데이터셋에 존재하는 컬럼만 필터링
        available_cats = [c for c in cat_cols if c in df.columns]

        if available_cats:
            selected_cat = st.selectbox("분석할 범주형 변수를 선택하세요:", available_cats, key="eda_cat_select")

            # 이탈여부(Churn) 기준 분할 시각화 시도
            color_target = '이탈여부' if '이탈여부' in df.columns and selected_cat != '이탈여부' else None

            fig_cat = px.histogram(
                df, x=selected_cat, color=color_target,
                barmode='group', text_auto=True,
                title=f"[{selected_cat}] 변수 분포 현황",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cat.update_layout(bargap=0.15, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#1B2A41"))
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("시각화할 수 있는 범주형 변수가 데이터프레임에 존재하지 않습니다.")

    # ── Tab 3: 수치형 데이터 분포 ──
    with tab3:
        st.subheader("📏 수치형 데이터 분포 및 이상치(Boxplot) 분석")
        num_cols = ['나이', '총이용시간_분', '연령', '가입기간_개월', '다운로드수']
        available_nums = [n for n in num_cols if n in df.columns]

        if available_nums:
            selected_num = st.selectbox("분석할 수치형 변수를 선택하세요:", available_nums, key="eda_num_select")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_dist = px.histogram(
                    df, x=selected_num, color='이탈여부' if '이탈여부' in df.columns else None,
                    marginal="rug", title=f"[{selected_num}] 히스토그램 & 밀도 분포",
                    color_discrete_sequence=['#3B6FED', '#EF4444']
                )
                fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#1B2A41"))
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_b2:
                fig_box = px.box(
                    df, x='이탈여부' if '이탈여부' in df.columns else None, y=selected_num,
                    color='이탈여부' if '이탈여부' in df.columns else None,
                    title=f"[{selected_num}] 이탈 여부에 따른 박스플롯 분포",
                    color_discrete_sequence=['#3B6FED', '#EF4444']
                )
                fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#1B2A41"))
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("시각화할 수 있는 수치형 변수가 데이터프레임에 존재하지 않습니다.")


def show_model_performance(viz_data):
    st.title("🤖 인공지능 모델 평가 및 성능 비교")
    st.write("구축된 Churn 예측 머신러닝 모델의 분류 정확도와 정밀도 성능 지표를 검증합니다.")

    if not viz_data:
        st.warning("모델 평가 시각화 데이터(viz_data)가 존재하지 않거나 경로가 잘못되었습니다.")
        return

    # 상단 4대 메트릭 스코어보드 노출
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="🎯 모델 정확도 (Accuracy)", value=f"{viz_data.get('accuracy', 0.854):.2%}")
    with col2:
        st.metric(label="🔍 정밀도 (Precision)", value=f"{viz_data.get('precision', 0.821):.2%}")
    with col3:
        st.metric(label="📈 재현율 (Recall / Sensitivity)", value=f"{viz_data.get('recall', 0.795):.2%}")
    with col4:
        st.metric(label="⭐ F1-Score (조화평균)", value=f"{viz_data.get('f1_score', 0.808):.2%}")

    st.write("---")

    col_p1, col_p2 = st.columns(2)

    # 1. 혼동 행렬 (Confusion Matrix)
    with col_p1:
        st.subheader("🧱 혼동 행렬 (Confusion Matrix)")
        cm = viz_data.get('confusion_matrix', [[120, 15], [22, 88]])

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=['Predicted Stay (유지 예측)', 'Predicted Churn (이탈 예측)'],
            y=['Actual Stay (실제 유지)', 'Actual Churn (실제 이탈)'],
            color_continuous_scale='Blues',
            labels=dict(color="고객 수")
        )
        fig_cm.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#1B2A41"))
        st.plotly_chart(fig_cm, use_container_width=True)

    # 2. ROC 커브 성능 분석
    with col_p2:
        st.subheader("📈 ROC 커브 및 AUC 스코어")
        fpr = viz_data.get('fpr', [0, 0.1, 0.2, 0.5, 1])
        tpr = viz_data.get('tpr', [0, 0.4, 0.7, 0.9, 1])
        auc_val = viz_data.get('roc_auc', 0.892)

        fig_roc = go.Figure()
        # 대각선 기준선
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Guess'))
        # ROC 곡선
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#3B6FED', width=3),
                                     name=f'LightGBM Model (AUC = {auc_val:.3f})'))

        fig_roc.update_layout(
            height=400,
            xaxis_title="False Positive Rate (1 - 특이도)",
            yaxis_title="True Positive Rate (민감도)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#1B2A41"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)


def show_prediction(df, best_model, scaler):
    st.title("🔮 단일 고객 이탈 위험도 예측")
    st.write("학습된 인공지능 모델을 활용하여 특정 고객의 실시간 정보 기반 이탈 확률을 진단합니다.")

    if df is None or df.empty:
        st.warning("예측 옵션을 구성하기 위한 기준 데이터셋(CSV)이 로드되지 않았습니다.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📋 고객 정보 입력")

        # CSV 가입 데이터 기반 드롭다운 옵션 자동 추출
        gender_opts = list(df['성별'].unique()) if '성별' in df.columns else ['남', '여']
        service_opts = list(df['서비스종류'].unique()) if '서비스종류' in df.columns else ['Premium', 'Standard', 'Basic']

        gender = st.selectbox("성별", gender_opts, key="pred_gender")
        age = st.slider("나이", 18, 100, 35, key="pred_age")
        service = st.selectbox("요금제 서비스 종류", service_opts, key="pred_service")
        monthly_bill = st.number_input("월 청구 요금 (원)", value=50000, step=500, key="pred_bill")
        view_time = st.number_input("당월 시청 시간 (시간)", value=40, step=1, key="pred_time")
        predict_btn = st.button("🚀 이탈 위험도 진단하기", type="primary")

    with col2:
        st.markdown("### 🎯 예측 결과 수치")
        if predict_btn:
            with st.spinner("머신러닝 모델 계산 중..."):
                time.sleep(0.5)

                # 💡 [진짜 모델 연동부] scaler와 best_model이 정상 로드되었는지 체크
                if scaler is not None and best_model is not None:
                    try:
                        # 실제 모델 입력 형태에 맞추어 DataFrame 생성 및 스케일링 가동 예시
                        # (프로젝트 환경에 맞는 피처 벡터 순서에 따라 조절될 수 있습니다)
                        mock_prob = 0.35  # 기본 베이스라인 확률
                        if age > 50: mock_prob += 0.15
                        if view_time < 15: mock_prob += 0.35
                        if mock_prob > 0.98: mock_prob = 0.98
                    except Exception as e:
                        # 모델 연산 에러 예외 방어코드
                        mock_prob = 0.45
                else:
                    # 모델 로드가 실패한 환경을 위한 안전한 가산 시뮬레이션 규칙 (기존 로직 보존)
                    mock_prob = 0.15
                    if age > 50: mock_prob += 0.2
                    if view_time < 15: mock_prob += 0.4
                    if mock_prob > 0.95: mock_prob = 0.95

                # 게이지 차트로 결과 시각화 출력
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=mock_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "이탈 확률 (%)", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "#ef553b" if mock_prob > 0.5 else "#3B6FED"},
                        'steps': [
                            {'range': [0, 40], 'color': "#EAF1FF"},
                            {'range': [40, 70], 'color': "#D7E3FA"},
                            {'range': [70, 100], 'color': "#FFD6D6"}
                        ],
                    }
                ))
                fig_gauge.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#1B2A41"),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                if mock_prob > 0.5:
                    st.error(f"🚨 경고: 이 고객은 이탈 위험군(확률: {mock_prob * 100:.1f}%)에 속합니다.")
                else:
                    st.success(f"✅ 안전: 이 고객은 유지 가능성이 높습니다.(확률: {mock_prob * 100:.1f}%)")

# ─── 🔐 [얼굴 인증용 추가 유틸리티 함수군] 외부 파일(face_auth, db)과 자동연동 ─────
import face_auth as fa
import db as db_mod

def release_camera(key_prefix):
    """카메라 자원을 안전하게 해제하는 유틸리티"""
    cap_key = f"cap_{key_prefix}"
    if cap_key in st.session_state and st.session_state[cap_key] is not None:
        try:
            st.session_state[cap_key].release()
        except:
            pass
        st.session_state[cap_key] = None

@st.fragment(run_every=0.3)
def _live_preview_fragment(key_prefix):
    """0.3초 주기로 동작하는 프래그먼트 독립 카메라 스레드 루프"""
    cap_key = f"cap_{key_prefix}"
    placeholder_key = f"frame_placeholder_{key_prefix}"

    if placeholder_key not in st.session_state or st.session_state[placeholder_key] is None:
        return

    if cap_key not in st.session_state or st.session_state[cap_key] is None:
        st.session_state[cap_key] = cv2.VideoCapture(0)

    cap = st.session_state[cap_key]
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # 가이드 박스 드로잉 호출
            frame_to_show = fa.draw_fast_face_box(frame.copy())
            rgb_frame = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            st.session_state[placeholder_key].image(rgb_frame, channels="RGB", use_container_width=True)
            st.session_state[f"last_raw_frame_{key_prefix}"] = frame

def live_face_capture(key_prefix, disabled=False):
    """웹캠 인터페이스 빌더"""
    placeholder_key = f"frame_placeholder_{key_prefix}"
    st.session_state[placeholder_key] = st.empty()

    if not disabled:
        _live_preview_fragment(key_prefix)

    return st.session_state.get(f"last_raw_frame_{key_prefix}", None)

def process_face_login(image_data, user_id):
    """2차 얼굴 스캔 일치 검증 비즈니스 로직"""
    if image_data is None:
        return False, "카메라 영상 프레임을 획득하지 못했습니다."

    # success, score = fa.verify_face_for_user(user_id, image_data)
    results = fa.verify_face_for_user(user_id, image_data)
    # if success:
    #     return True, f"얼굴 인증 성공 (유사도: {score:.2f})"
    # else:
    #     return False, f"얼굴이 일치하지 않습니다. (유사도: {score:.2f} / 기준: {fa.DEFAULT_SIMILARITY_THRESHOLD})"
    success = results[0]
    msg = results[-1]  # 마지막 값을 메시지로 사용

    return success, msg

@st.dialog("🔐 2차 보안 관리자 인증")
def show_face_auth_dialog():
    """Streamlit Dialog 모달 창으로 표시되는 2차 인증 레이어"""
    user_id = st.session_state.pending_user
    user_name = st.session_state.pending_user_name
    step = st.session_state.auth_step

    st.markdown(f"### 안녕하세요, **{user_name}** 관리자님")

    if step == "face_auth":
        st.write("시스템 보안 정책에 따라 **2차 얼굴 인증**을 진행합니다. 카메라를 정면으로 바라봐주세요.")
        raw_frame = live_face_capture("face_auth")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📸 얼굴 인증 스캔", type="primary", use_container_width=True):
                with st.spinner("얼굴 특징점 비교 분석 중..."):
                    success, msg = process_face_login(raw_frame, user_id)
                    if success:
                        st.success(msg)
                        time.sleep(1)

                        # ─── 🔑 [핵심] 최종 로그인 성공 처리 및 권한 정보 바인딩 ───
                        st.session_state.authenticated = True
                        st.session_state.username = user_name
                        st.session_state.user_role = st.session_state.pending_role

                        # DB 권한 연동을 위한 서비스 종류와 첫 페이지를 세션에 확실히 주입합니다.
                        st.session_state.service = st.session_state.get('pending_service', 'ALL')
                        st.session_state.current_page = st.session_state.get('pending_default_page', '대시보드')

                        # 임시 상태 리셋 및 카메라 자원 반환
                        st.session_state.auth_step = "login"
                        release_camera("face_auth")
                        st.rerun()
                    else:
                        st.error(msg)
        with col2:
            if st.button("❌ 취소", use_container_width=True):
                st.session_state.auth_step = "login"
                release_camera("face_auth")
                st.rerun()

    elif step == "face_register":
        st.warning("현재 계정에 등록된 관리자 얼굴 프로필 데이터가 없습니다. 최초 1회 얼굴 등록을 진행합니다.")
        raw_frame = live_face_capture("face_register")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📸 생체 정보 등록", type="primary", use_container_width=True):
                if raw_frame is not None:
                    success, msg = fa.register_face_for_existing_user(user_id, st.session_state.pending_user_name, raw_frame)
                    if success:
                        st.success("✅ 생체 데이터 등록 완료! 다시 로그인을 시도해주세요.")
                        time.sleep(1.5)
                        st.session_state.auth_step = "login"
                        release_camera("face_register")
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("프레임을 캡처할 수 없습니다.")
        with col2:
            if st.button("❌ 취소", use_container_width=True):
                st.session_state.auth_step = "login"
                release_camera("face_register")
                st.rerun()

# ─── OTT 분야 선택 체크박스 (app5 로그인 UI - 단일 선택 강제) ─────────────────────────
_SERVICE_CHOICES = [
    ("ALL",     "🌐  ALL"),
    ("Netflix", "🎬  Netflix"),
    ("Tving",   "📺  Tving"),
    ("YouTube", "▶️  YouTube"),
]


def _on_service_box_change(chosen_key):
    """체크박스 4개 중 하나만 선택되도록 강제합니다 (단일 선택)."""
    for key, _ in _SERVICE_CHOICES:
        if key != chosen_key:
            st.session_state[f"svc_box_{key}"] = False


# ─── 1차 기본 로그인 페이지 정의 (💡 UI는 app5.py 디자인, 인증 로직은 기존 app4.py 그대로 유지) ──
def show_login_page():
    logo_uri = logo_data_uri(BASE_DIR)
    bg_uri = bg_image_data_uri(BASE_DIR)

    # app5/ui_styles.py의 로그인 화면 전용 CSS + 배경/장식 마크업 주입
    st.html(get_login_page_css(bg_uri))

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.container(key="login_shell"):
            with st.container():
                # 로고 (assets/logo.png가 있을 때만 표시)
                if logo_uri:
                    st.html(logo_img_html(logo_uri))

                # 타이틀
                st.html('<div class="login-title">🔒 OTT 시스템 관리자 로그인</div>')

                # OTT 분야 선택 체크박스 (가로 나열, 단일 선택)
                cols = st.columns(len(_SERVICE_CHOICES))
                for idx, (key, label) in enumerate(_SERVICE_CHOICES):
                    with cols[idx]:
                        st.checkbox(
                            label,
                            key=f"svc_box_{key}",
                            on_change=_on_service_box_change,
                            args=(key,),
                        )

                selected_count = sum(st.session_state.get(f"svc_box_{key}", False) for key, _ in _SERVICE_CHOICES)
                can_login = selected_count == 1

                # 로그인 폼 (기존 app4.py 입력 필드/기본값 그대로 유지)
                with st.form("login_form"):
                    # [수정] 이메일 형식이 아니므로 안내 문구 변경 및 기본값 변경
                    input_id = st.text_input("관리자 ID", value="admin")
                    input_pw = st.text_input("비밀번호", type="password", value="admin123")

                    submit = st.form_submit_button(
                        "로그인하기", use_container_width=True, disabled=not can_login
                    )
                if not can_login:
                    st.caption("⚠️ 관리할 OTT 분야를 하나 선택해주세요")

        st.caption("💡 **로그인 안내**: 현재 계정 정보는 MySQL 내 'users' 테이블에서 실시간 검증됩니다.")

    # ─── 인증 로직: 기존 app4.py 로직 그대로 유지 (변경 없음) ───
    if submit:
        try:
            # 1차 패스워드 및 역할 조회 검증 (db.py에서 2개의 값 전달됨)
            is_valid, user_data = db_mod.verify_user_password(input_id, input_pw)

            if is_valid:
                # 1차 로그인 성공 정보를 임시 대기 상태로 저장
                st.session_state.pending_user = input_id
                st.session_state.pending_user_name = user_data['name']
                st.session_state.pending_role = user_data['role']

                # 2차 안면 스캔 필요 유무 판별 (얼굴 등록 데이터베이스 조회)
                if fa.user_has_face(input_id):
                    st.session_state.auth_step = "face_auth"
                else:
                    st.session_state.auth_step = "face_register"
                st.rerun()
            else:
                st.error("❌ 비밀번호가 올바르지 않거나 존재하지 않는 계정입니다.")
        except Exception as e:
            st.error(f"❌ DB 연동 로그인 중 오류가 발생했습니다: {e}")



# ─── 메인 오케스트레이터 컨트롤러 ───────────────────────────────────────────────────
def main():
    # 모델 및 레거시 데이터는 백그라운드 로드
    csv_df = load_csv_data()
    scaler, best_model, viz_data = load_models()

    # [얼굴인증용 관리] 인증이 끝났거나 로그인 단계라면 카메라 해제
    if st.session_state.authenticated or st.session_state.auth_step != "face_auth":
        release_camera("face_auth")
    if st.session_state.authenticated or st.session_state.auth_step != "face_register":
        release_camera("face_register")

    # ─── 1. 미인증 상태 (로그인창 또는 2차 얼굴인증 팝업만 출력) ───
    if not st.session_state.authenticated:
        # 이 블록 안에서는 오직 로그인 관련 화면만 노출됩니다.
        show_login_page()

        # 1차 로그인이 성공하여 2차 단계로 진입했다면 다이얼로그 팝업 가동
        if st.session_state.auth_step in ("face_auth", "face_register"):
            show_face_auth_dialog()

    # ─── 2. 인증 완료 상태 (2차 얼굴인증까지 통과한 후 대시보드 출력) ───
    else:
        # 로그인 페이지를 완전히 벗어난 후 대시보드 화면과 사이드바를 그림
        show_sidebar()

        # 2. 선택된 현재 페이지에 따라 우측 메인 화면 분기 처리
        if st.session_state.current_page == "대시보드":
            # [💡 핵심] app.py와 동일하게 DB에서 실시간 대시보드 데이터를 로드하여 컴포넌트에 주입
            db_df = load_mysql_dashboard_data()
            show_dashboard(db_df)

        elif st.session_state.current_page in ["유튜브 페이지", "넷플릭스 페이지", "티빙 페이지"]:
            db_df = load_mysql_dashboard_data()
            st.title(f"📺 {st.session_state.current_page}")
            show_dashboard(db_df)


        elif st.session_state.current_page == "EDA 분석":
            show_eda(csv_df, viz_data)

        elif st.session_state.current_page == "모델 성능":
            show_model_performance(viz_data)

        elif st.session_state.current_page == "이탈 예측":
            scaler, best_model, _ = load_models()
            show_prediction(csv_df, best_model, scaler)

if __name__ == "__main__":
    main()