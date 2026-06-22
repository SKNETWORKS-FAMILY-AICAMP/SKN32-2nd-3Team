"""
🎬 OTT 서비스 고객 데이터 통합 대시보드 시스템
얼굴 인식 로그인 및 예측 머신러닝 파이프라인 유지 + 3대 그래프 MySQL 실시간 연동 (결측치 제거 + 벤다이어그램 한글 깨짐 해결)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import cv2
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib_venn import venn3  # 벤 다이어그램 라이브러리
from datetime import datetime
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')

# ─── 🛠️ [핵심 수정] Matplotlib 한글 폰트 깨짐 방지 전역 설정 ──────────────────────────
import platform
from matplotlib import font_manager, rc

# 운영체제별 사용 가능한 한글 폰트 탐색 및 지정
if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin': # Mac OS
    rc('font', family='AppleGothic')
else: # Linux 등 기타 환경
    rc('font', family='sans-serif')

# 그래프 내 마이너스(-) 기호가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False
# ──────────────────────────────────────────────────────────────────────────────

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
if 'login_time' not in st.session_state:
    st.session_state.login_time = None
if 'pending_user' not in st.session_state:
    st.session_state.pending_user = None
if 'pending_user_name' not in st.session_state:
    st.session_state.pending_user_name = None
if 'pending_role' not in st.session_state:
    st.session_state.pending_role = None
if 'auth_step' not in st.session_state:
    st.session_state.auth_step = "login"
if 'service' not in st.session_state:
    st.session_state.service = None

# ─── 얼굴 로그인용 데모 계정 시드 (최초 1회) ───────────────────────────────────────
from db import seed_demo_accounts
seed_demo_accounts()

# ─── OTT 서비스별 관리자 역할 메타 정보 ────────────────────────────────────────────
# DB의 role 컬럼은 서비스 구분자(ALL/YouTube/Netflix/Tving)로 저장되며,
# 로그인 성공 후 이 메타 정보로 표시 라벨/메뉴/기본 페이지를 결정합니다.
ROLE_META = {
    "ALL": {"label": "전체 관리자", "service": "ALL", "default_page": "대시보드"},
    "YouTube": {"label": "유튜브 관리자", "service": "YouTube", "default_page": "유튜브 페이지"},
    "Netflix": {"label": "넷플릭스 관리자", "service": "Netflix", "default_page": "넷플릭스 페이지"},
    "Tving": {"label": "티빙 관리자", "service": "Tving", "default_page": "티빙 페이지"},
}

MENU_BY_SERVICE = {
    "ALL": ["대시보드", "EDA 분석", "모델 성능", "이탈 예측"],
    "YouTube": ["유튜브 페이지", "EDA 분석"],
    "Netflix": ["넷플릭스 페이지", "EDA 분석"],
    "Tving": ["티빙 페이지", "EDA 분석"],
}

# ─── CSS 커스텀 스타일 정의 (디자인 요소는 ui_styles.py로 분리) ───────────────────────────
from ui_styles import (
    GLOBAL_CSS, LOGIN_PAGE_CSS, logo_data_uri, logo_img_html,
    login_left_title_html, login_left_subtitle_html,
    login_right_title_html, login_right_subtitle_html,
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─── [데이터 펑션] MySQL 실시간 연동 및 통합 전처리 ──────────────────────────────────────
@st.cache_data
def load_mysql_dashboard_data():
    """
    MySQL DB에서 실제 설문/이용 데이터를 가져와 3대 그래프용으로 통합 결합하는 함수
    """
    try:
        engine = create_engine("mysql+pymysql://root:mysql80@localhost:3306/ott_db?charset=utf8mb4")

        query = """
        SELECT 
            t.YEAR, 
            t.OPID, 
            t.`Weekday usage` AS weekday_usage, 
            t.`Weekend usage` AS weekend_usage,
            m.svod,
            u.`ott first` AS ott_first,
            u.`ott second` AS ott_second
        FROM ott_time t
        LEFT JOIN ott_money m ON t.OPID = m.OPID AND t.YEAR = m.YEAR
        LEFT JOIN ott_usage u ON t.OPID = u.OPID AND t.YEAR = u.YEAR
        """
        df = pd.read_sql(query, con=engine)

        # [전처리 ①] 이용 시간 결합 (분 단위 합산)
        df['weekday_usage'] = df['weekday_usage'].fillna(0)
        df['weekend_usage'] = df['weekend_usage'].fillna(0)
        df['총이용시간_분'] = df['weekday_usage'] + df['weekend_usage']

        # [전처리 ②] 대시보드 UI 규격에 맞게 한글 컬럼명 매핑
        df = df.rename(columns={
            'OPID': '고객ID',
            'YEAR': '연도',
            'svod': '이용요금'
        })

        return df

    except Exception as e:
        st.error(f"❌ MySQL 실시간 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# ─── 기존 레거시 파일 기반 데이터 로더 ─────────────────────────────────────────
@st.cache_data
def load_legacy_churn_data():
    try:
        return pd.read_csv('korea_telecom_churn.csv', encoding='utf-8-sig')
    except Exception as e:
        return pd.DataFrame()

# ─── 모델 성능/이탈 예측 시각화용 데이터 로더 ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_visualization_data():
    try:
        with open(os.path.join(BASE_DIR, 'data', 'visualization_data.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

# ─── 1. 대시보드 페이지 화면 정의 ──────────────────────────────────────────────────
def show_dashboard(df):
    st.title("📊 실시간 OTT 데이터 통합 모니터링 대시보드")

    if df.empty:
        st.warning("데이터베이스에 연결할 수 없거나 읽어올 데이터가 없습니다.")
        return

    st.success(f"✅ MySQL 실시간 데이터 연동 성공! (조회된 레코드 수: {len(df):,}개)")

    # 3대 핵심 그래프 영역 데이터 시각화 구성
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⏱️ 고객별 평균 총 이용시간 (분)")

        fig_time = px.histogram(
            df,
            x="총이용시간_분",
            nbins=30,
            color_discrete_sequence=['#1f6feb']
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
            font=dict(color="#e6edf3")
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        if '이용요금' in df.columns:
            st.subheader("💰 이용 요금대별 고객 분포")

            # 결측치(Null, 공백) 완벽 차단 필터
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
                    font=dict(color="#e6edf3"),
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

            # 다크 테마 배경에 맞춰 벤다이어그램 캔버스 스타일 생성
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0F1117')
            ax.set_facecolor('#0F1117')

            # 벤 다이어그램 렌더링
            v = venn3(
                subsets=[netflix_set, tving_set, disney_set],
                set_labels=('Netflix', 'Tving', 'Disney+'),
                ax=ax
            )

            # 텍스트 컬러 및 시스템 폰트 상속 활성화 스타일링
            if v:
                for text in v.set_labels:
                    if text:
                        text.set_color('#e6edf3')
                        text.set_fontsize(11)
                for text in v.subset_labels:
                    if text:
                        text.set_color('#ffffff')
                        text.set_fontsize(10)

            # 폰트 전역 셋업이 완료되었으므로 상단 제목 한글이 깨지지 않고 완벽히 표기됩니다.
            plt.title("OTT 구독자 교차 중복 현황 (Venn)", color='#e6edf3', fontsize=13, pad=15)
            st.pyplot(fig)
        else:
            st.info("중복 분석용 데이터(ott_first, ott_second)가 부족합니다.")

    st.dataframe(df.head(50), use_container_width=True)

# ─── 2. EDA 분석 페이지 ────────────────────────────────────────────────────────
def show_eda(df):
    st.title("🔍 이탈 예측 인프라용 EDA 분석 공간")
    st.dataframe(df.head(100), use_container_width=True)

# ─── 3. 모델 성능 페이지 ───────────────────────────────────────────────────────
def _select_model(task_name, model_results, key_prefix):
    """Task에 속한 모델 목록에서 하나를 고르는 selectbox (여러 섹션에서 재사용)."""
    models_in_task = list(model_results[task_name].keys())
    return st.selectbox("모델 선택", models_in_task, key=f"{key_prefix}_model_{task_name}")


def show_model_performance(viz_data):
    st.header("📊 모델 성능 비교")

    if viz_data and 'model_results' in viz_data:
        model_results = viz_data['model_results']

        # 모델 성능 테이블 생성
        performance_data = []

        # 실제 데이터는 영어 키가 아니라 "정확도_Test"처럼 한글+Val/Test 접미사 키로 저장되어 있습니다.
        # Test 셋 값을 우선 사용하고, 없으면 Val 값으로 대체합니다.
        def _get_metric(metrics, korean_key):
            return metrics.get(f"{korean_key}_Test", metrics.get(f"{korean_key}_Val", 0))

        for task_name, task_results in model_results.items():
            for model_name, metrics in task_results.items():
                performance_data.append({
                    "Task": task_name,
                    "Model": model_name,
                    "Accuracy": _get_metric(metrics, '정확도'),
                    "Precision": _get_metric(metrics, '정밀도'),
                    "Recall": _get_metric(metrics, '재현율'),
                    "F1-Score": _get_metric(metrics, 'F1 점수'),
                    "ROC-AUC": _get_metric(metrics, 'AUC-ROC')
                })

        if performance_data:
            performance_df = pd.DataFrame(performance_data)

            st.subheader("모델 성능 테이블")
            st.dataframe(performance_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # 시각화: 모델별 성능 비교
            st.subheader("모델별 성능 비교")

            # Task별로 그룹화하여 시각화
            for task_name in model_results.keys():
                task_df = performance_df[performance_df['Task'] == task_name]

                if not task_df.empty:
                    st.markdown(f"### {task_name}")

                    # 성능 지표 선택
                    metric = st.selectbox(
                        "성능 지표 선택",
                        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                        key=f"metric_{task_name}"
                    )

                    # 바 차트
                    fig = go.Figure(data=[
                        go.Bar(
                            x=task_df['Model'],
                            y=task_df[metric],
                            marker_color='skyblue'
                        )
                    ])

                    fig.update_layout(
                        title=f"{task_name} - {metric} 비교",
                        xaxis_title="모델",
                        yaxis_title=metric,
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")

            # 최고 성능 모델 표시
            st.subheader("최고 성능 모델")

            best_models = []
            for task_name, task_results in model_results.items():
                best_f1 = 0
                best_model = ""
                for model_name, metrics in task_results.items():
                    f1_value = _get_metric(metrics, 'F1 점수')
                    if f1_value > best_f1:
                        best_f1 = f1_value
                        best_model = model_name

                best_models.append({
                    "Task": task_name,
                    "Best Model": best_model,
                    "F1-Score": best_f1
                })

            best_models_df = pd.DataFrame(best_models)
            st.dataframe(best_models_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Confusion Matrix 시각화
            st.subheader("Confusion Matrix")

            # Task별로 Confusion Matrix 표시
            for task_name in model_results.keys():
                st.markdown(f"### {task_name}")

                selected_model = _select_model(task_name, model_results, "cm")

                # Confusion Matrix 데이터가 있는지 확인
                if 'confusion_matrices' in viz_data and viz_data['confusion_matrices']:
                    confusion_matrices = viz_data['confusion_matrices']

                    # 해당 모델의 confusion matrix 가져오기 (test → val → 접미사 없는 키 순으로 탐색)
                    cm_key = next(
                        (k for k in (f"{task_name}_{selected_model}_test", f"{task_name}_{selected_model}_val",
                                     f"{task_name}_{selected_model}")
                         if k in confusion_matrices),
                        None,
                    )
                    if cm_key is not None:
                        cm_data = confusion_matrices[cm_key]
                        cm = np.array(cm_data)
                    else:
                        st.info(f"⚠️ {selected_model}의 Confusion Matrix 데이터가 없습니다.")
                        cm = np.array([[1000, 50], [100, 200]])
                else:
                    st.info("⚠️ Confusion Matrix 데이터가 없습니다. train_models.py를 실행하여 confusion matrix를 저장해야 합니다.")
                    cm = np.array([[1000, 50], [100, 200]])

                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Negative', 'Predicted Positive'],
                    y=['Actual Negative', 'Actual Positive'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 14}
                ))

                fig_cm.update_layout(
                    title=f"{task_name} - {selected_model} Confusion Matrix",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig_cm, use_container_width=True)

                st.markdown("---")

            # ROC Curve 시각화
            st.subheader("ROC Curve")

            # Task별로 ROC Curve 표시
            for task_name in model_results.keys():
                st.markdown(f"### {task_name}")

                selected_model = _select_model(task_name, model_results, "roc")

                # ROC Curve 데이터가 있는지 확인 (roc_curves는 "{task}_{model}" 형태의 평탄한 키로 저장되어 있습니다)
                roc_data = viz_data.get('roc_curves', {}).get(f"{task_name}_{selected_model}")
                if roc_data:
                    fpr = np.array(roc_data['fpr'])
                    tpr = np.array(roc_data['tpr'])
                    auc = roc_data['auc']
                else:
                    st.info(f"⚠️ {selected_model}의 ROC Curve 데이터가 없습니다.")
                    fpr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    tpr = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0])
                    auc = 0.85

                fig_roc = go.Figure()

                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {auc:.3f})',
                    line=dict(color='blue', width=2)
                ))

                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='gray', width=2, dash='dash')
                ))

                fig_roc.update_layout(
                    title=f"{task_name} - {selected_model} ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True
                )

                st.plotly_chart(fig_roc, use_container_width=True)

                st.markdown("---")

            # Precision-Recall Curve 시각화
            st.subheader("Precision-Recall Curve")

            # Task별로 Precision-Recall Curve 표시
            for task_name in model_results.keys():
                st.markdown(f"### {task_name}")

                selected_model = _select_model(task_name, model_results, "pr")

                # Precision-Recall Curve 데이터가 있는지 확인 (pr_curves도 "{task}_{model}" 형태의 평탄한 키)
                pr_data = viz_data.get('pr_curves', {}).get(f"{task_name}_{selected_model}")
                if pr_data:
                    precision = np.array(pr_data['precision'])
                    recall = np.array(pr_data['recall'])
                    ap = pr_data['ap']
                else:
                    st.info(f"⚠️ {selected_model}의 Precision-Recall Curve 데이터가 없습니다.")
                    precision = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5])
                    recall = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    ap = 0.75

                fig_pr = go.Figure()

                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'PR Curve (AP = {ap:.3f})',
                    line=dict(color='green', width=2)
                ))

                fig_pr.update_layout(
                    title=f"{task_name} - {selected_model} Precision-Recall Curve",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True
                )

                st.plotly_chart(fig_pr, use_container_width=True)

                st.markdown("---")

            # Feature Importance 시각화
            st.subheader("Feature Importance")

            # Feature Importance 데이터가 있는지 확인
            if 'feature_importance' in viz_data and viz_data['feature_importance']:
                feature_importance = viz_data['feature_importance']

                # Feature Importance 테이블 생성
                fi_data = []
                for feature, importance in feature_importance.items():
                    fi_data.append({"Feature": feature, "Importance": importance})

                if fi_data:
                    fi_df = pd.DataFrame(fi_data)
                    fi_df = fi_df.sort_values('Importance', ascending=False)

                    # TOP20 표시
                    fi_top20 = fi_df.head(20)

                    st.markdown("### TOP20 Feature Importance")
                    st.dataframe(fi_top20, use_container_width=True, hide_index=True)

                    # Feature Importance 바 차트
                    fig_fi = go.Figure(data=[
                        go.Bar(
                            x=fi_top20['Importance'],
                            y=fi_top20['Feature'],
                            orientation='h',
                            marker_color='orange'
                        )
                    ])

                    fig_fi.update_layout(
                        title="TOP20 Feature Importance",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=600,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )

                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.warning("Feature Importance 데이터가 없습니다.")
            else:
                st.info("⚠️ Feature Importance 데이터가 없습니다. train_models.py를 실행하여 feature importance를 저장해야 합니다.")

                # 예시 Feature Importance
                example_features = [
                    {"Feature": "ott_time_recent_mean", "Importance": 0.15},
                    {"Feature": "ott_usage_01_decline_count", "Importance": 0.12},
                    {"Feature": "ott_time_cv", "Importance": 0.10},
                    {"Feature": "ott_usage_02_recent_mean", "Importance": 0.08},
                    {"Feature": "ott_time_std", "Importance": 0.07}
                ]

                example_df = pd.DataFrame(example_features)

                st.markdown("### 예시 Feature Importance")
                st.dataframe(example_df, use_container_width=True, hide_index=True)

                fig_fi = go.Figure(data=[
                    go.Bar(
                        x=example_df['Importance'],
                        y=example_df['Feature'],
                        orientation='h',
                        marker_color='orange'
                    )
                ])

                fig_fi.update_layout(
                    title="예시 Feature Importance",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig_fi, use_container_width=True)

                st.markdown("---")

            # SHAP Summary 시각화
            st.subheader("SHAP Summary Plot")

            # SHAP 데이터가 있는지 확인
            if 'shap_values' in viz_data and viz_data['shap_values'] is not None:
                shap_values = viz_data['shap_values']
                features = viz_data.get('features', [])

                st.info("⚠️ SHAP Summary Plot은 현재 데이터 형식으로 표시하기 어렵습니다. train_models.py에서 SHAP 이미지를 저장하거나 데이터 형식을 변경해야 합니다.")

                # SHAP Summary Plot 설명
                st.markdown("""
                **SHAP Summary Plot 설명:**
                - 각 점은 하나의 데이터 샘플을 나타냅니다
                - X축: SHAP 값 (피처가 예측에 미치는 영향)
                - Y축: 피처 (중요도 순으로 정렬)
                - 색상: 피처 값 (높은 값은 빨간색, 낮은 값은 파란색)
                - 오른쪽에 위치할수록 이탈 예측에 긍정적인 영향
                - 왼쪽에 위치할수록 이탈 예측에 부정적인 영향
                """)
            else:
                st.info("⚠️ SHAP 데이터가 없습니다. train_models.py를 실행하여 SHAP 값을 계산해야 합니다.")

                # SHAP Summary Plot 설명
                st.markdown("""
                **SHAP Summary Plot 설명:**
                - 각 점은 하나의 데이터 샘플을 나타냅니다
                - X축: SHAP 값 (피처가 예측에 미치는 영향)
                - Y축: 피처 (중요도 순으로 정렬)
                - 색상: 피처 값 (높은 값은 빨간색, 낮은 값은 파란색)
                - 오른쪽에 위치할수록 이탈 예측에 긍정적인 영향
                - 왼쪽에 위치할수록 이탈 예측에 부정적인 영향
                """)

                st.markdown("**참고:** SHAP Summary Plot은 train_models.py에서 SHAP 값을 계산하고 저장해야 표시할 수 있습니다.")

                st.markdown("---")

            # Learning Curve 시각화
            st.subheader("Learning Curve")

            # Learning Curve 데이터가 있는지 확인
            if 'learning_curves' in viz_data and viz_data['learning_curves']:
                learning_curves = viz_data['learning_curves']

                # Task별로 Learning Curve 표시
                for task_name in learning_curves.keys():
                    st.markdown(f"### {task_name}")

                    # 모델 선택
                    models_in_task = list(learning_curves[task_name].keys())
                    selected_model = st.selectbox(
                        "모델 선택",
                        models_in_task,
                        key=f"lc_model_{task_name}"
                    )

                    lc_data = learning_curves[task_name][selected_model]

                    # Learning Curve 그래프
                    fig_lc = go.Figure()

                    fig_lc.add_trace(go.Scatter(
                        x=lc_data['train_sizes'],
                        y=lc_data['train_scores_mean'],
                        mode='lines+markers',
                        name='Training Score',
                        line=dict(color='blue')
                    ))

                    fig_lc.add_trace(go.Scatter(
                        x=lc_data['train_sizes'],
                        y=lc_data['val_scores_mean'],
                        mode='lines+markers',
                        name='Validation Score',
                        line=dict(color='red')
                    ))

                    fig_lc.update_layout(
                        title=f"{task_name} - {selected_model} Learning Curve",
                        xaxis_title="Training Set Size",
                        yaxis_title="Score",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=True
                    )

                    st.plotly_chart(fig_lc, use_container_width=True)

                    st.markdown("---")
            else:
                st.info("⚠️ Learning Curve 데이터가 없습니다. train_models.py를 실행하여 Learning Curve를 계산해야 합니다.")

                # Learning Curve 설명
                st.markdown("""
                **Learning Curve 설명:**
                - 훈련 데이터 크기에 따른 모델 성능 변화를 시각화
                - 파란색: 훈련 데이터 점수
                - 빨간색: 검증 데이터 점수
                - 훈련 점수와 검증 점수의 간격이 크면 과적합(Overfitting)
                - 두 점수 모두 낮으면 과소적합(Underfitting)
                - 두 점수가 높고 간격이 좁으면 적절한 모델
                """)

                # 예시 Learning Curve
                train_sizes = np.array([100, 500, 1000, 2000, 5000])
                train_scores = np.array([0.95, 0.92, 0.90, 0.88, 0.85])
                val_scores = np.array([0.75, 0.80, 0.82, 0.84, 0.85])

                fig_lc = go.Figure()

                fig_lc.add_trace(go.Scatter(
                    x=train_sizes,
                    y=train_scores,
                    mode='lines+markers',
                    name='Training Score (예시)',
                    line=dict(color='blue')
                ))

                fig_lc.add_trace(go.Scatter(
                    x=train_sizes,
                    y=val_scores,
                    mode='lines+markers',
                    name='Validation Score (예시)',
                    line=dict(color='red')
                ))

                fig_lc.update_layout(
                    title="예시 Learning Curve",
                    xaxis_title="Training Set Size",
                    yaxis_title="Score",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True
                )

                st.plotly_chart(fig_lc, use_container_width=True)

                st.markdown("---")

            # Threshold-F1 Curve 시각화
            st.subheader("Threshold-F1 Curve")

            # Threshold-F1 Curve 데이터가 있는지 확인
            # (threshold_f1_curves는 "{task}_{model}" 형태의 평탄한 키로 저장되어 있습니다)
            if 'threshold_f1_curves' in viz_data and viz_data['threshold_f1_curves']:
                threshold_f1_curves = viz_data['threshold_f1_curves']

                # Task별로 Threshold-F1 Curve 표시
                for task_name in model_results.keys():
                    st.markdown(f"### {task_name}")

                    selected_model = _select_model(task_name, model_results, "tf")

                    tf_data = threshold_f1_curves.get(f"{task_name}_{selected_model}")
                    if tf_data is None:
                        st.info(f"⚠️ {selected_model}의 Threshold-F1 Curve 데이터가 없습니다.")
                        st.markdown("---")
                        continue

                    # Threshold-F1 Curve 그래프
                    fig_tf = go.Figure()

                    fig_tf.add_trace(go.Scatter(
                        x=tf_data['thresholds'],
                        y=tf_data['f1_scores'],
                        mode='lines+markers',
                        name='F1 Score',
                        line=dict(color='purple')
                    ))

                    # 최적 threshold 표시
                    best_threshold = tf_data['best_threshold']
                    best_f1 = tf_data['best_f1']

                    fig_tf.add_vline(
                        x=best_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Best Threshold: {best_threshold:.3f}",
                        annotation_position="top"
                    )

                    fig_tf.update_layout(
                        title=f"{task_name} - {selected_model} Threshold-F1 Curve (Best F1: {best_f1:.3f})",
                        xaxis_title="Threshold",
                        yaxis_title="F1 Score",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=True
                    )

                    st.plotly_chart(fig_tf, use_container_width=True)

                    st.markdown("---")
            else:
                st.info("⚠️ Threshold-F1 Curve 데이터가 없습니다. train_models.py를 실행하여 Threshold-F1 Curve를 계산해야 합니다.")

                # Threshold-F1 Curve 설명
                st.markdown("""
                **Threshold-F1 Curve 설명:**
                - 다양한 threshold에 따른 F1 score 변화를 시각화
                - X축: Threshold (0~1)
                - Y축: F1 Score
                - 빨간색 점선: 최적 threshold
                - threshold가 낮으면 재현율(Recall)이 높아지고 정밀도(Precision)가 낮아짐
                - threshold가 높으면 정밀도(Precision)가 높아지고 재현율(Recall)이 낮아짐
                - 최적 threshold는 F1 score가 가장 높은 지점
                """)

                # 예시 Threshold-F1 Curve
                thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                f1_scores = np.array([0.65, 0.72, 0.78, 0.82, 0.85, 0.83, 0.79, 0.74, 0.68])
                best_threshold = 0.5
                best_f1 = 0.85

                fig_tf = go.Figure()

                fig_tf.add_trace(go.Scatter(
                    x=thresholds,
                    y=f1_scores,
                    mode='lines+markers',
                    name='F1 Score (예시)',
                    line=dict(color='purple')
                ))

                fig_tf.add_vline(
                    x=best_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Best Threshold: {best_threshold:.3f}",
                    annotation_position="top"
                )

                fig_tf.update_layout(
                    title="예시 Threshold-F1 Curve (Best F1: 0.85)",
                    xaxis_title="Threshold",
                    yaxis_title="F1 Score",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True
                )

                st.plotly_chart(fig_tf, use_container_width=True)

                st.markdown("---")

            # Top10 High Risk Customer 시각화
            st.subheader("Top10 High Risk Customer")

            # Top10 데이터가 있는지 확인 (top10_customers는 task/model로 중첩되지 않은 평탄한 리스트입니다)
            top10_customers = viz_data.get('top10_customers')
            if top10_customers:
                top10_df = pd.DataFrame(top10_customers)
                if 'churn_prob' in top10_df.columns:
                    top10_df = top10_df.sort_values('churn_prob', ascending=False).head(10)

                if {'user_id', 'churn_prob'}.issubset(top10_df.columns):
                    st.markdown("#### Top10 High Risk Customers")
                    st.dataframe(
                        top10_df[[c for c in ['user_id', 'churn_prob', 'churn'] if c in top10_df.columns]],
                        use_container_width=True, hide_index=True,
                    )

                    # Top10 바 차트
                    fig_top10 = go.Figure(data=[
                        go.Bar(
                            x=top10_df['churn_prob'],
                            y=top10_df['user_id'],
                            orientation='h',
                            marker_color='red'
                        )
                    ])

                    fig_top10.update_layout(
                        title="Top10 High Risk Customers",
                        xaxis_title="Churn Probability",
                        yaxis_title="User ID",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )

                    st.plotly_chart(fig_top10, use_container_width=True)

                    if 'churn' in top10_df.columns:
                        actual_churn_count = int(top10_df['churn'].sum())
                        st.markdown(f"**실제 이탈:** {actual_churn_count}/{len(top10_df)} ({actual_churn_count * 10}%)")

                    st.markdown("---")
                else:
                    st.warning("Top10 데이터 형식이 올바르지 않습니다.")
            else:
                st.info("⚠️ Top10 High Risk Customer 데이터가 없습니다. train_models.py를 실행하여 Top10 데이터를 저장해야 합니다.")

                # Top10 설명
                st.markdown("""
                **Top10 High Risk Customer 설명:**
                - 이탈 확률이 가장 높은 상위 10명의 고객을 시각화
                - X축: 이탈 확률 (Churn Probability)
                - Y축: 사용자 ID (User ID)
                - 빨간색 막대: 이탈 확률
                - 실제 이탈 여부를 확인하여 모델의 예측 정확도 평가
                - Precision@10 지표와 연관됨
                """)

                # 예시 Top10
                example_top10 = [
                    {"user_id": "user_001", "churn_prob": 0.95, "churn": 1},
                    {"user_id": "user_002", "churn_prob": 0.92, "churn": 1},
                    {"user_id": "user_003", "churn_prob": 0.88, "churn": 1},
                    {"user_id": "user_004", "churn_prob": 0.85, "churn": 0},
                    {"user_id": "user_005", "churn_prob": 0.82, "churn": 1},
                    {"user_id": "user_006", "churn_prob": 0.80, "churn": 1},
                    {"user_id": "user_007", "churn_prob": 0.78, "churn": 0},
                    {"user_id": "user_008", "churn_prob": 0.75, "churn": 1},
                    {"user_id": "user_009", "churn_prob": 0.72, "churn": 1},
                    {"user_id": "user_010", "churn_prob": 0.70, "churn": 1}
                ]

                example_df = pd.DataFrame(example_top10)

                st.markdown("#### 예시 Top10 High Risk Customers")
                st.dataframe(example_df, use_container_width=True, hide_index=True)

                fig_top10 = go.Figure(data=[
                    go.Bar(
                        x=example_df['churn_prob'],
                        y=example_df['user_id'],
                        orientation='h',
                        marker_color='red'
                    )
                ])

                fig_top10.update_layout(
                    title="예시 Top10 High Risk Customers",
                    xaxis_title="Churn Probability",
                    yaxis_title="User ID",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig_top10, use_container_width=True)

                actual_churn_count = example_df['churn'].sum()
                st.markdown(f"**실제 이탈:** {actual_churn_count}/10 ({actual_churn_count*10}%)")

                st.markdown("---")

            # Risk Distribution 시각화
            st.subheader("Risk Distribution")

            # Risk Distribution 데이터가 있는지 확인
            # (risk_distribution도 "{task}_{model}" 형태의 평탄한 키로 저장되어 있습니다)
            if 'risk_distribution' in viz_data and viz_data['risk_distribution']:
                risk_distribution = viz_data['risk_distribution']

                # Task별로 Risk Distribution 표시
                for task_name in model_results.keys():
                    st.markdown(f"### {task_name}")

                    selected_model = _select_model(task_name, model_results, "rd")

                    rd_data = risk_distribution.get(f"{task_name}_{selected_model}")
                    if rd_data is None:
                        st.info(f"⚠️ {selected_model}의 Risk Distribution 데이터가 없습니다.")
                        st.markdown("---")
                        continue

                    # Risk Distribution 히스토그램
                    fig_rd = go.Figure(data=[
                        go.Histogram(
                            x=rd_data['churn_probs'],
                            nbinsx=50,
                            name='Churn Probability Distribution',
                            marker_color='orange'
                        )
                    ])

                    fig_rd.update_layout(
                        title=f"{task_name} - {selected_model} Risk Distribution",
                        xaxis_title="Churn Probability",
                        yaxis_title="Count",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=True
                    )

                    st.plotly_chart(fig_rd, use_container_width=True)

                    # Risk Category 분포
                    risk_categories = rd_data.get('risk_categories', {})
                    if risk_categories:
                        st.markdown("#### Risk Category Distribution")

                        risk_df = pd.DataFrame([
                            {"Category": "Low Risk (0-0.3)", "Count": risk_categories.get('low', 0)},
                            {"Category": "Medium Risk (0.3-0.7)", "Count": risk_categories.get('medium', 0)},
                            {"Category": "High Risk (0.7-1.0)", "Count": risk_categories.get('high', 0)}
                        ])

                        st.dataframe(risk_df, use_container_width=True, hide_index=True)

                        # Risk Category 파이 차트
                        fig_pie = go.Figure(data=[
                            go.Pie(
                                labels=risk_df['Category'],
                                values=risk_df['Count'],
                                hole=0.3
                            )
                        ])

                        fig_pie.update_layout(
                            title=f"{task_name} - {selected_model} Risk Category Distribution",
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )

                        st.plotly_chart(fig_pie, use_container_width=True)

                    st.markdown("---")
            else:
                st.info("⚠️ Risk Distribution 데이터가 없습니다. train_models.py를 실행하여 Risk Distribution을 계산해야 합니다.")

                # Risk Distribution 설명
                st.markdown("""
                **Risk Distribution 설명:**
                - 전체 고객의 이탈 확률 분포를 시각화
                - X축: 이탈 확률 (Churn Probability)
                - Y축: 고객 수 (Count)
                - Low Risk (0-0.3): 이탈 확률이 낮은 고객
                - Medium Risk (0.3-0.7): 이탈 확률이 중간인 고객
                - High Risk (0.7-1.0): 이탈 확률이 높은 고객
                - 분포를 통해 전체 고객의 이탈 위험도 패턴 파악 가능
                """)

                # 예시 Risk Distribution
                np.random.seed(42)
                example_probs = np.concatenate([
                    np.random.beta(2, 5, 5000),  # Low risk
                    np.random.beta(5, 5, 3000),  # Medium risk
                    np.random.beta(5, 2, 2000)   # High risk
                ])

                fig_rd = go.Figure(data=[
                    go.Histogram(
                        x=example_probs,
                        nbinsx=50,
                        name='Churn Probability Distribution (예시)',
                        marker_color='orange'
                    )
                ])

                fig_rd.update_layout(
                    title="예시 Risk Distribution",
                    xaxis_title="Churn Probability",
                    yaxis_title="Count",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True
                )

                st.plotly_chart(fig_rd, use_container_width=True)

                # 예시 Risk Category 분포
                example_risk_df = pd.DataFrame([
                    {"Category": "Low Risk (0-0.3)", "Count": 5000},
                    {"Category": "Medium Risk (0.3-0.7)", "Count": 3000},
                    {"Category": "High Risk (0.7-1.0)", "Count": 2000}
                ])

                st.markdown("#### 예시 Risk Category Distribution")
                st.dataframe(example_risk_df, use_container_width=True, hide_index=True)

                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=example_risk_df['Category'],
                        values=example_risk_df['Count'],
                        hole=0.3
                    )
                ])

                fig_pie.update_layout(
                    title="예시 Risk Category Distribution",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("---")
        else:
            st.warning("모델 결과 데이터가 없습니다.")
    else:
        st.warning("모델 결과 데이터를 찾을 수 없습니다. train_models.py를 실행하여 모델 결과를 생성하세요.")

# ─── 4. 이탈 예측 인퍼런스 페이지 ──────────────────────────────────────────────────
def show_churn_prediction(viz_data):
    st.header("🔮 이탈 예측")

    st.markdown("### 이탈 예측 기능")
    st.markdown("이 페이지에서는 학습된 모델을 사용하여 이탈 예측을 수행합니다.")

    # 모델 선택
    st.subheader("모델 선택")
    model_options = ["XGBoost_Tuned", "RandomForest", "LightGBM", "CatBoost", "Ensemble", "DeepLearning_Transfer"]
    selected_model = st.selectbox("예측에 사용할 모델 선택", model_options)

    st.markdown("---")

    # 이탈자 예측 Top 10
    st.subheader("이탈자 예측 Top 10")

    # 실제 데이터 로드
    top10_data = viz_data.get('top10_customers', [])

    if top10_data and len(top10_data) > 0:
        # 실제 데이터 사용
        top10_df = pd.DataFrame(top10_data)

        # 필요한 컬럼만 선택
        if 'user_id' in top10_df.columns and 'churn_prob' in top10_df.columns:
            display_df = top10_df[['user_id', 'churn_prob']].copy()
            if 'churn' in top10_df.columns:
                display_df['actual_churn'] = top10_df['churn']
            display_df['risk_level'] = display_df['churn_prob'].apply(lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.3 else 'Low')
            display_df = display_df.head(10)

            st.markdown("#### 이탈 예측 Top 10 고객")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # 이탈 예측 바 차트
            fig_churn = go.Figure(data=[
                go.Bar(
                    x=display_df['churn_prob'],
                    y=display_df['user_id'],
                    orientation='h',
                    marker=dict(color=display_df['churn_prob'], colorscale='Reds'),
                    text=display_df['churn_prob'].apply(lambda x: f'{x:.2%}'),
                    textposition='outside'
                )
            ])
            fig_churn.update_layout(
                title='이탈 예측 Top 10 고객 확률',
                xaxis_title='이탈 확률',
                yaxis_title='사용자 ID',
                height=400
            )
            st.plotly_chart(fig_churn, use_container_width=True)
        else:
            st.warning("Top10 데이터 형식이 올바르지 않습니다.")
    else:
        st.warning("Top10 데이터가 없습니다. train_models.py를 실행하여 데이터를 생성하세요.")

    st.markdown("---")

    # 미래 년도 이탈 예측
    st.subheader("미래 년도 이탈 예측")

    st.markdown("#### 미래 년도 이탈률 예측")

    # 예시 미래 년도 이탈 예측
    future_years = [2024, 2025, 2026, 2027, 2028]
    future_churn_rates = [0.15, 0.18, 0.22, 0.25, 0.28]

    future_df = pd.DataFrame({
        "년도": future_years,
        "예상 이탈률": future_churn_rates
    })

    st.dataframe(future_df, use_container_width=True, hide_index=True)

    # 미래 년도 이탈률 라인 차트
    fig_future = go.Figure(data=[
        go.Scatter(
            x=future_years,
            y=future_churn_rates,
            mode='lines+markers',
            name='예상 이탈률',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        )
    ])

    fig_future.update_layout(
        title="미래 년도 이탈률 예측",
        xaxis_title="년도",
        yaxis_title="예상 이탈률",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )

    st.plotly_chart(fig_future, use_container_width=True)

    # 미래 년도 이탈자 수 예측
    st.markdown("#### 미래 년도 이탈자 수 예측")

    total_customers = 11040
    future_churn_counts = [int(total_customers * rate) for rate in future_churn_rates]

    future_count_df = pd.DataFrame({
        "년도": future_years,
        "총 고객 수": [total_customers] * len(future_years),
        "예상 이탈자 수": future_churn_counts,
        "예상 잔존 고객 수": [total_customers - count for count in future_churn_counts]
    })

    st.dataframe(future_count_df, use_container_width=True, hide_index=True)

    # 미래 년도 이탈자 수 바 차트
    fig_future_count = go.Figure()

    fig_future_count.add_trace(go.Bar(
        x=future_years,
        y=future_churn_counts,
        name='예상 이탈자 수',
        marker_color='red'
    ))

    fig_future_count.add_trace(go.Bar(
        x=future_years,
        y=[total_customers - count for count in future_churn_counts],
        name='예상 잔존 고객 수',
        marker_color='green'
    ))

    fig_future_count.update_layout(
        title="미래 년도 이탈자 수 예측",
        xaxis_title="년도",
        yaxis_title="고객 수",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        barmode='stack',
        showlegend=True
    )

    st.plotly_chart(fig_future_count, use_container_width=True)

    st.markdown("---")

    # 설명
    st.markdown("""
    **이탈 예측 설명:**
    - **이탈자 예측 Top 10:** 현재 고객 중 이탈 확률이 가장 높은 상위 10명을 예측
    - **미래 년도 이탈률 예측:** 과거 데이터를 기반으로 미래 년도의 이탈률을 예측
    - **미래 년도 이탈자 수 예측:** 예상 이탈률을 적용하여 미래 년도의 이탈자 수를 예측

    **참고:**
    - 실제 예측을 수행하려면 train_models.py에서 학습된 모델을 저장하고 로드해야 합니다
    - 미래 년도 예측은 시계열 모델이나 추세 분석을 통해 더 정확하게 수행할 수 있습니다
    - 현재는 예시 데이터를 표시하며, 실제 데이터를 적용하면 더 정확한 예측이 가능합니다
    """)

# ─── 얼굴 인식 로그인용 유틸리티 ────────────────────────────────────────────────────
def release_camera(key_prefix: str) -> None:
    """해당 key_prefix로 열려 있는 카메라를 닫고, 이전에 찍힌 사진 기록도 함께 지웁니다."""
    cap_key = f"{key_prefix}_cap"
    cap = st.session_state.get(cap_key)
    if cap is not None:
        cap.release()
        st.session_state[cap_key] = None
    st.session_state[f"{key_prefix}_previewing"] = False
    st.session_state[f"{key_prefix}_captured_frame"] = None


_MAX_CONSECUTIVE_FRAME_FAILS = 5


@st.fragment(run_every=0.3)
def _live_preview_fragment(key_prefix: str):
    """0.3초마다 한 프레임씩 읽어 윤곽선을 그려 보여주고, '촬영하기' 클릭 시 그 프레임을 캡처합니다."""
    from face_auth import draw_fast_face_box

    cap_key = f"{key_prefix}_cap"
    previewing_key = f"{key_prefix}_previewing"
    captured_key = f"{key_prefix}_captured_frame"
    fail_count_key = f"{key_prefix}_fail_count"

    cap = st.session_state.get(cap_key)
    if cap is None:
        st.session_state[previewing_key] = False
        st.rerun()
        return

    ret, frame = cap.read()
    if ret:
        st.session_state[fail_count_key] = 0
        display = draw_fast_face_box(frame)
        st.image(display, channels="BGR", width=360)
        st.session_state[f"{key_prefix}_last_frame"] = frame
    else:
        fail_count = st.session_state.get(fail_count_key, 0) + 1
        st.session_state[fail_count_key] = fail_count
        if fail_count >= _MAX_CONSECUTIVE_FRAME_FAILS:
            st.error("카메라 연결이 끊겼습니다. 다시 시작해주세요.")
            release_camera(key_prefix)
            st.rerun()
            return
        st.warning("프레임을 가져오지 못했습니다. 계속 시도 중...")

    if st.button("📸 촬영하기", key=f"{key_prefix}_capture", use_container_width=True):
        last_frame = st.session_state.get(f"{key_prefix}_last_frame")
        release_camera(key_prefix)
        st.session_state[captured_key] = last_frame
        st.rerun()


def live_face_capture(key_prefix: str, disabled: bool = False):
    """페이지에 들어오면 자동으로 카메라를 열어 실시간 얼굴 윤곽선 미리보기를 보여주고,
    사용자가 원하는 순간 '촬영하기' 버튼을 눌러 그 프레임을 캡처합니다."""
    captured_key = f"{key_prefix}_captured_frame"
    previewing_key = f"{key_prefix}_previewing"
    cap_key = f"{key_prefix}_cap"

    captured = st.session_state.get(captured_key)
    if captured is not None:
        st.image(captured, channels="BGR", caption="촬영된 이미지", use_container_width=True)
        if st.button("🔄 다시 촬영", key=f"{key_prefix}_retry", disabled=disabled):
            st.session_state[captured_key] = None
            st.rerun()
        return captured

    if disabled:
        return None

    if not st.session_state.get(previewing_key, False):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows에서 더 빠르게 열리는 백엔드
        if not cap.isOpened():
            cap.release()
            st.error("카메라를 열 수 없습니다. 다른 프로그램이 카메라를 사용 중인지 확인해주세요.")
            return None
        st.session_state[cap_key] = cap
        st.session_state[previewing_key] = True
        st.rerun()
        return None

    _live_preview_fragment(key_prefix)
    return st.session_state.get(captured_key)


def process_face_login(image_data, user_id):
    """1차 인증으로 식별된 계정(user_id)에 대해 얼굴 인증을 수행합니다."""
    try:
        from face_auth import verify_face_for_user, DEFAULT_SIMILARITY_THRESHOLD

        if not isinstance(image_data, np.ndarray):
            return False, 0.0, "이미지 형식 오류"

        success, similarity, message = verify_face_for_user(
            user_id, image_data, user_name=st.session_state.get("pending_user"),
            threshold=DEFAULT_SIMILARITY_THRESHOLD,
        )
        return success, similarity, message
    except Exception as e:
        return False, 0.0, f"인증 오류: {str(e)}"


# ─── 사이드바 및 로그인 화면 정의 ──────────────────────────────────────────────────
def show_sidebar():
    st.sidebar.title(f"🎬 {st.session_state.username} 님")
    st.sidebar.write(f"권한: `{st.session_state.user_role}`")
    if st.session_state.login_time:
        st.sidebar.caption(f"최근 로그인: {st.session_state.login_time.strftime('%Y-%m-%d %H:%M:%S')}")

    menu_list = MENU_BY_SERVICE.get(st.session_state.service, ["대시보드"])
    if st.session_state.current_page not in menu_list:
        st.session_state.current_page = menu_list[0]

    page = st.sidebar.radio("메뉴 이동", menu_list, index=menu_list.index(st.session_state.current_page))
    st.session_state.current_page = page

    if st.sidebar.button("로그아웃"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.service = None
        st.session_state.login_time = None
        st.session_state.current_page = "대시보드"
        # 이전 로그인에서 찍었던 얼굴 사진이 다음 로그인에 그대로 남지 않도록 초기화합니다.
        release_camera("face_auth")
        release_camera("face_register")
        st.rerun()


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


def show_login_page():
    """1차 인증(ID/PW) 화면. 성공하면 얼굴 인증(또는 얼굴 등록) 단계로 넘어갑니다.
    좌측의 OTT 분야 선택은 로그인 버튼을 활성화하기 위한 UX 단계일 뿐이며,
    로그인 후 실제 권한/메뉴는 인증된 계정의 DB role(ROLE_META)로 결정됩니다."""

    logo_uri = logo_data_uri(BASE_DIR)

    st.html(LOGIN_PAGE_CSS)

    with st.container(key="login_shell"):
        left, right = st.columns([1, 1])

        with left:
            if logo_uri:
                st.html(logo_img_html(logo_uri))

            st.html(login_left_title_html("어떤 OTT 관리자이신가요?"))
            st.html(login_left_subtitle_html("관리할 플랫폼을 선택해주세요"))

            # 체크박스 세로 일렬 배치 (디자인 가이드: 수직 스택)
            for key, label in _SERVICE_CHOICES:
                st.checkbox(
                    label,
                    key=f"svc_box_{key}",
                    on_change=_on_service_box_change,
                    args=(key,),
                )

            selected_count = sum(st.session_state.get(f"svc_box_{key}", False) for key, _ in _SERVICE_CHOICES)
            can_login = selected_count == 1

        with right:
            st.html(login_right_title_html("🔐 OTT Analytics 관제 시스템"))
            st.html(login_right_subtitle_html("관리자 계정으로 로그인하세요"))

            with st.form("login_form"):
                username = st.text_input("아이디", placeholder="아이디를 입력하세요", icon="📧")
                password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요", icon="🔒")
                submit = st.form_submit_button(
                    "로그인", use_container_width=True, disabled=not can_login
                )
                if not can_login:
                    st.caption("⚠️ 왼쪽에서 관리할 OTT 분야를 하나 선택해주세요.")

        if submit:
            from db import verify_user_password, user_has_face, update_login_attempt, log_login_attempt
            from face_auth import check_lock_status

            is_locked, lock_message = check_lock_status(username)
            if is_locked:
                st.error(lock_message)
            else:
                ok, user_name, role, message = verify_user_password(username, password)

                if ok:
                    update_login_attempt(username, success=True)
                    log_login_attempt(username, user_name, success=True)

                    st.session_state.pending_user = username
                    st.session_state.pending_user_name = user_name
                    st.session_state.pending_role = role
                    st.session_state.auth_step = "face_auth" if user_has_face(username) else "face_register"

                    st.success("1차 인증 성공")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    update_login_attempt(username, success=False)
                    log_login_attempt(username, None, success=False)

                    from db import get_user_lock_info
                    lock_info = get_user_lock_info(username)
                    if lock_info:
                        remaining = max(0, 5 - (lock_info["fail_count"] or 0))
                        st.error(f"{message} (남은 시도: {remaining}회)")
                    else:
                        st.error(message)


@st.dialog("🔐 얼굴 인증")
def show_face_auth_dialog():
    """2차 인증(얼굴 인식/얼굴 등록)을 하나의 팝업으로 처리합니다.
    얼굴 미등록 계정은 '얼굴 등록하기'가 먼저 뜨고, 등록을 마치면 같은 팝업 자리에서
    '얼굴 인증하기'로 전환됩니다 (등록 → 인증 순서)."""
    st.write(f"1차 인증 계정: **{st.session_state.pending_user}**")

    if st.session_state.auth_step == "face_register":
        camera_img = live_face_capture("face_register")

        reg_col1, reg_col2 = st.columns(2)

        with reg_col1:
            if st.button("얼굴 등록하기", use_container_width=True):
                if camera_img is None:
                    st.error("얼굴 이미지를 촬영해주세요.")
                else:
                    from face_auth import register_face_for_existing_user

                    with st.spinner("얼굴 등록 중..."):
                        success, message = register_face_for_existing_user(
                            st.session_state.pending_user,
                            st.session_state.get("pending_user_name", st.session_state.pending_user),
                            camera_img,
                        )

                    if success:
                        st.success(message)
                        st.session_state.auth_step = "face_auth"
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(message)

        with reg_col2:
            if st.button("이전으로", use_container_width=True, key="face_register_back"):
                release_camera("face_register")
                st.session_state.pending_user = None
                st.session_state.pending_role = None
                st.session_state.auth_step = "login"
                st.rerun()
        return

    from face_auth import check_lock_status
    is_locked, lock_message = check_lock_status(st.session_state.pending_user)
    if is_locked:
        st.error(lock_message)

    camera_img = live_face_capture("face_auth", disabled=is_locked)

    auth_col1, auth_col2 = st.columns(2)

    with auth_col1:
        if st.button("얼굴 인증하기", use_container_width=True, disabled=is_locked):
            if camera_img is None:
                st.error("얼굴 이미지를 촬영해주세요.")
            else:
                with st.spinner("얼굴 인증 중..."):
                    success, similarity, message = process_face_login(camera_img, st.session_state.pending_user)

                if success:
                    user_name = st.session_state.get("pending_user_name") or st.session_state.pending_user
                    role_meta = ROLE_META.get(
                        st.session_state.pending_role,
                        {"label": st.session_state.pending_role, "service": "ALL", "default_page": "대시보드"},
                    )

                    st.session_state.authenticated = True
                    st.session_state.username = st.session_state.pending_user
                    st.session_state.user_role = role_meta["label"]
                    st.session_state.service = role_meta["service"]
                    st.session_state.current_page = role_meta["default_page"]
                    st.session_state.login_time = datetime.now()

                    st.session_state.pending_user = None
                    st.session_state.pending_role = None
                    st.session_state.auth_step = "login"

                    st.success(f"확인되었습니다 {user_name}님!")
                    st.caption(f"유사도: {similarity:.2%}")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    from db import get_user_lock_info
                    lock_info = get_user_lock_info(st.session_state.pending_user)
                    if lock_info:
                        remaining = max(0, 5 - (lock_info["fail_count"] or 0))
                        st.error(f"얼굴 인증 실패: {message} (남은 시도: {remaining}회)")
                    else:
                        st.error(f"얼굴 인증 실패: {message}")
                    st.caption(f"유사도: {similarity:.2%}")

    with auth_col2:
        if st.button("이전으로", use_container_width=True):
            release_camera("face_auth")
            st.session_state.pending_user = None
            st.session_state.pending_role = None
            st.session_state.auth_step = "login"
            st.rerun()


# ─── 메인 오케스트레이터 컨트롤러 ───────────────────────────────────────────────────
def main():
    # 현재 화면이 아닌 다른 단계에 카메라가 열려 있으면 안전하게 정리합니다.
    if st.session_state.authenticated or st.session_state.auth_step != "face_auth":
        release_camera("face_auth")
    if st.session_state.authenticated or st.session_state.auth_step != "face_register":
        release_camera("face_register")

    if not st.session_state.authenticated:
        # 1차 로그인 화면을 그대로 두고, 얼굴 등록/2차 얼굴 인증은 그 위에 팝업(모달)으로 띄웁니다.
        show_login_page()
        if st.session_state.auth_step in ("face_auth", "face_register"):
            show_face_auth_dialog()
    else:
        show_sidebar()

        if st.session_state.current_page == "대시보드":
            db_df = load_mysql_dashboard_data()
            show_dashboard(db_df)

        elif st.session_state.current_page == "유튜브 페이지":
            db_df = load_mysql_dashboard_data()
            st.title("▶️ 유튜브 관리자 페이지")
            show_dashboard(db_df)

        elif st.session_state.current_page == "넷플릭스 페이지":
            db_df = load_mysql_dashboard_data()
            st.title("🎬 넷플릭스 관리자 페이지")
            show_dashboard(db_df)

        elif st.session_state.current_page == "티빙 페이지":
            db_df = load_mysql_dashboard_data()
            st.title("📺 티빙 관리자 페이지")
            show_dashboard(db_df)

        elif st.session_state.current_page == "EDA 분석":
            legacy_df = load_legacy_churn_data()
            show_eda(legacy_df)

        elif st.session_state.current_page == "모델 성능":
            viz_data = load_visualization_data()
            show_model_performance(viz_data)

        elif st.session_state.current_page == "이탈 예측":
            viz_data = load_visualization_data()
            show_churn_prediction(viz_data)

if __name__ == "__main__":
    main()