"""
OTT 고객 이탈 예측 시스템 시각화 대시보드
데이터 분석 시각화
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="OTT 데이터 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_visualization_data():
    try:
        with open(os.path.join(BASE_DIR, 'data', 'visualization_data.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def show_preprocessing_a(viz_data):
    st.header("🔧 전처리 - a: OTT 데이터 전처리")
    
    st.markdown("""
    ### 📊 OTT 데이터 전처리 및 특성 추출
    
    **목적:** 원본 OTT 데이터를 사용자(User) 레벨로 집계하고 시계열 기반 이탈 정의를 적용하여 특성을 추출합니다.
    
    **처리 과정:**
    1. **데이터 로드:** ott_money, ott_time, ott_usage_01, ott_usage_02 데이터 로드
    2. **사용자별 집계:** 각 사용자별로 시계열 데이터를 집계
    3. **이탈 정의:** 최근 3개월 사용량이 전체 평균의 50% 미만일 경우 이탈로 정의
    4. **특성 추출:** 시계열 통계량 (mean, std, cv, trend, max, min 등) 추출
    5. **데이터 누수 방지:** 미래 정보를 포함하는 특성 제거 또는 안전한 버전 사용
    
    **생성된 특성:**
    - ott_time 관련: mean, std, cv, trend, max, min, change_last, decline_count, max_drop, recent_mean, usage_years_count, consecutive_decline_count, max_increase, diff_std, recent_to_overall_ratio, last_year_ratio_safe, two_year_ratio_safe, longest_decline_streak, volatility, recent_2yr_consecutive_decline_safe, recent_3yr_consecutive_decline_safe, recent_decline_score, recent_decline_rate
    - ott_usage_01 관련: mean, std, cv, trend, change_last, decline_count, max_drop, recent_mean, usage_years_count, consecutive_decline_count, max_increase, diff_std, recent_to_overall_ratio
    - ott_usage_02 관련: mean, std, cv, trend, change_last, decline_count, max_drop, recent_mean, usage_years_count, consecutive_decline_count, max_increase, diff_std, recent_to_overall_ratio
    
    **데이터 손실:** 원본 데이터에서 결측치나 부적절한 데이터를 제거하여 일부 사용자가 제외될 수 있습니다.
    """)
    
    st.markdown("---")
    
    # 원본 데이터셋 정보
    st.subheader("원본 데이터셋")
    
    original_data_info = [
        {"파일": "data/ott_money.csv", "설명": "OTT 금액 데이터", "주요 컬럼": ["pid", "YEAR", "ott_money"]},
        {"파일": "data/ott_time.csv", "설명": "OTT 시간 데이터", "주요 컬럼": ["pid", "YEAR", "ott_time"]},
        {"파일": "data/ott_usage_01.csv", "설명": "OTT 사용량 데이터 (Type 01)", "주요 컬럼": ["pid", "YEAR", "ott_usage_01"]},
        {"파일": "data/ott_usage_02.csv", "설명": "OTT 사용량 데이터 (Type 02)", "주요 컬럼": ["pid", "YEAR", "ott_usage_02"]}
    ]
    
    st.dataframe(pd.DataFrame(original_data_info), use_container_width=True)
    
    st.markdown("---")
    
    # 전처리 결과 요약
    if viz_data and 'data_analysis' in viz_data:
        data_analysis = viz_data['data_analysis']
        
        st.subheader("전처리 결과 요약")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 샘플 수", f"{data_analysis['total_samples']:,}")
        
        with col2:
            st.metric("총 피처 수", data_analysis['total_features'])
        
        with col3:
            st.metric("이탈 비율", f"{(data_analysis.get('churn_rate', 0) * 100):.1f}%")

def show_preprocessing_b(viz_data):
    st.header("🔧 전처리 - b: 데이터 전처리 (결측치 처리, 스케일링, 특성 선택)")
    
    st.markdown("""
    ### 📊 데이터 전처리: 결측치 처리, 스케일링, 특성 선택
    
    **목적:** 전처리 a에서 생성된 데이터를 모델 학습에 적합한 형태로 변환합니다.
    
    **처리 과정:**
    1. **결측치 처리:** SimpleImputer를 사용하여 결측치를 중앙값으로 채움
    2. **스케일링:** StandardScaler를 사용하여 특성을 표준화 (평균 0, 표준편차 1)
    3. **특성 선택:** SelectKBest와 f_classif를 사용하여 중요한 특성 선택
    4. **데이터 분할:** Train 60%, Validation 20%, Test 20%로 분할
    
    **사용된 라이브러리:**
    - sklearn.impute.SimpleImputer: 결측치 처리
    - sklearn.preprocessing.StandardScaler: 특성 스케일링
    - sklearn.feature_selection.SelectKBest: 특성 선택
    - sklearn.feature_selection.f_classif: 특성 중요도 평가 (F-score)
    
    **출력:**
    - X_train_selected, X_val_selected, X_test_selected: 전처리된 데이터
    - selected_features: 선택된 특성 리스트
    - scaler: 스케일러 객체 (저장됨)
    - selector: 특성 선택기 객체 (저장됨)
    """)
    
    st.markdown("---")
    
    # 전처리 파라미터 정보
    st.subheader("전처리 파라미터")
    
    preprocessing_params = [
        {"파라미터": "결측치 처리 방법", "값": "중앙값 (median)"},
        {"파라미터": "스케일링 방법", "값": "표준화 (StandardScaler)"},
        {"파라미터": "특성 선택 방법", "값": "SelectKBest (f_classif)"},
        {"파라미터": "데이터 분할", "값": "Train 60%, Val 20%, Test 20%"},
        {"파라미터": "오버샘플링", "값": "SMOTETomek (클래스 불균형 처리)"}
    ]
    
    st.dataframe(pd.DataFrame(preprocessing_params), use_container_width=True)
    
    st.markdown("---")
    
    # 선택된 특성 정보
    if viz_data and 'features' in viz_data:
        st.subheader("선택된 특성")
        
        features = viz_data['features']
        
        st.markdown(f"**총 {len(features)}개 특성이 선택되었습니다.**")
        
        # 특성을 카테고리별로 분류
        time_features = [f for f in features if 'ott_time' in f]
        usage01_features = [f for f in features if 'ott_usage_01' in f]
        usage02_features = [f for f in features if 'ott_usage_02' in f]
        money_features = [f for f in features if 'ott_money' in f]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 시간 관련 특성")
            for f in time_features[:10]:
                st.markdown(f"- {f}")
            if len(time_features) > 10:
                st.markdown(f"- ... 외 {len(time_features) - 10}개")
            
            st.markdown("### 💰 금액 관련 특성")
            for f in money_features:
                st.markdown(f"- {f}")
        
        with col2:
            st.markdown("### 📈 사용량 01 관련 특성")
            for f in usage01_features[:10]:
                st.markdown(f"- {f}")
            if len(usage01_features) > 10:
                st.markdown(f"- ... 외 {len(usage01_features) - 10}개")
            
            st.markdown("### 📈 사용량 02 관련 특성")
            for f in usage02_features[:10]:
                st.markdown(f"- {f}")
            if len(usage02_features) > 10:
                st.markdown(f"- ... 외 {len(usage02_features) - 10}개")

def show_data_analysis(viz_data):
    st.header("📊 데이터 분석")
    
    if viz_data is None or 'data_analysis' not in viz_data:
        st.error("❌ 데이터 분석 정보를 찾을 수 없습니다. 먼저 train_models.py를 실행해주세요.")
        return
    
    data_analysis = viz_data['data_analysis']
    
    # 1. 데이터 기본 정보
    st.subheader("1. 데이터 기본 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 샘플 수", f"{data_analysis['total_samples']:,}")
    
    with col2:
        st.metric("총 피처 수", data_analysis['total_features'])
    
    with col3:
        st.metric("수치형 피처", data_analysis['numeric_features'])
    
    with col4:
        st.metric("범주형 피처", data_analysis['categorical_features'])
    
    st.markdown("---")
    
    # 2. 데이터셋 파일 정보
    st.subheader("2. 데이터셋 파일 정보")
    
    # 기본 제공 데이터셋
    original_datasets = [
        "data/ott_money.csv",
        "data/ott_time.csv",
        "data/ott_usage_01.csv",
        "data/ott_usage_02.csv"
    ]
    
    # 생성된 데이터셋
    generated_datasets = [
        "data/ott_model_results.json",
        "data/visualization_data.json",
        "data/ott_data_v.0/",
        "data/ott_data_v.2/",
        "data/before/sns_time.csv",
        "data/before/sns_usage_01.csv",
        "data/before/sns_usage_02.csv",
        "data/before/sns_usage_03.csv"
    ]
    
    # 수령 자료 저장 폴더 (데이터 제공 아님)
    storage_folders = [
        "data/before/",
        "data/xlsx/",
        "data/zip/"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 기본 제공 데이터셋")
        for dataset in original_datasets:
            st.markdown(f"- {dataset}")
    
    with col2:
        st.markdown("### 🆕 생성된 데이터셋")
        for dataset in generated_datasets:
            st.markdown(f"- {dataset}")
    
    st.markdown("---")
    
    # 수령 자료 저장 폴더
    st.subheader("수령 자료 저장 폴더 (데이터 제공 아님)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📂 before/")
        st.markdown("수령받은 원본 자료 저장")
    
    with col2:
        st.markdown("### 📂 xlsx/")
        st.markdown("수령받은 엑셀 자료 저장")
    
    with col3:
        st.markdown("### 📂 zip/")
        st.markdown("수령받은 압축 자료 저장")
    
    st.markdown("---")
    
    # 3. 피처 분류 정보
    st.subheader("3. 피처 분류 정보")
    
    # 기본 제공 피처 (원본 데이터셋)
    original_features = [
        "ott_money",
        "ott_time",
        "ott_usage_01",
        "ott_usage_02"
    ]
    
    # 생성된 피처 (피처 엔지니어링)
    generated_features = [
        "ott_time_mean", "ott_time_std", "ott_time_cv", "ott_time_trend",
        "ott_time_max", "ott_time_min", "ott_time_change_last", "ott_time_decline_count",
        "ott_time_max_drop", "ott_time_recent_mean", "ott_time_usage_years_count",
        "ott_time_consecutive_decline_count", "ott_time_max_increase", "ott_time_diff_std",
        "ott_time_recent_to_overall_ratio", "ott_time_last_year_ratio_safe",
        "ott_time_two_year_ratio_safe", "ott_time_longest_decline_streak",
        "ott_time_volatility", "ott_time_recent_2yr_consecutive_decline_safe",
        "ott_time_recent_3yr_consecutive_decline_safe", "ott_time_recent_decline_score",
        "ott_time_recent_decline_rate",
        "ott_usage_01_mean", "ott_usage_01_std", "ott_usage_01_cv", "ott_usage_01_trend",
        "ott_usage_01_change_last", "ott_usage_01_decline_count", "ott_usage_01_max_drop",
        "ott_usage_01_recent_mean", "ott_usage_01_usage_years_count",
        "ott_usage_01_consecutive_decline_count", "ott_usage_01_max_increase",
        "ott_usage_01_diff_std", "ott_usage_01_recent_to_overall_ratio",
        "ott_usage_02_mean", "ott_usage_02_std", "ott_usage_02_cv", "ott_usage_02_trend",
        "ott_usage_02_change_last", "ott_usage_02_decline_count", "ott_usage_02_max_drop",
        "ott_usage_02_recent_mean", "ott_usage_02_usage_years_count",
        "ott_usage_02_consecutive_decline_count", "ott_usage_02_max_increase",
        "ott_usage_02_diff_std", "ott_usage_02_recent_to_overall_ratio"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 기본 제공 피처")
        st.markdown(f"**총 {len(original_features)}개**")
        for feature in original_features:
            st.markdown(f"- {feature}")
    
    with col2:
        st.markdown("### 🆕 생성된 피처 (엔지니어링)")
        st.markdown(f"**총 {len(generated_features)}개**")
        for feature in generated_features:
            st.markdown(f"- {feature}")
    
    st.markdown("---")
    
    # 4. 결측치 분석
    st.subheader("4. 결측치 분석")
    
    missing_values = data_analysis.get('missing_values', {})
    missing_percentage = data_analysis.get('missing_percentage', {})
    
    # 결측치가 있는 피처만 필터링
    missing_df = pd.DataFrame({
        'Feature': list(missing_values.keys()),
        'Missing Count': list(missing_values.values()),
        'Missing %': list(missing_percentage.values())
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        st.markdown(f"**결측치가 있는 피처: {len(missing_df)}개**")
        
        # 결측치 표
        st.dataframe(missing_df, use_container_width=True)
        
        # 결측치 시각화
        fig_missing = px.bar(
            missing_df.head(20),
            x='Feature',
            y='Missing %',
            title='상위 20개 피처 결측치 비율',
            color='Missing %',
            color_continuous_scale='Reds'
        )
        fig_missing.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("✅ 결측치가 없습니다.")
    
    st.markdown("---")
    
    # 5. 피처별 통계 정보 (있는 경우)
    if 'feature_stats' in data_analysis:
        st.subheader("5. 피처별 통계 정보")
        
        feature_stats = data_analysis['feature_stats']
        
        # 통계 정보를 데이터프레임으로 변환
        stats_df = pd.DataFrame(feature_stats).T
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")

# 메인 함수
def main():
    st.title("📊 OTT 데이터 분석 대시보드")
    st.markdown("---")
    
    # 데이터 로드
    viz_data = load_visualization_data()
    
    # 사이드바
    with st.sidebar:
        st.header("메뉴")
        page = st.radio(
            "시각화 선택",
            ["0. 전처리 - a", "1. 전처리 - b", "2. 데이터 분석"]
        )
    
    # 페이지별 시각화
    if page == "0. 전처리 - a":
        show_preprocessing_a(viz_data)
    elif page == "1. 전처리 - b":
        show_preprocessing_b(viz_data)
    elif page == "2. 데이터 분석":
        show_data_analysis(viz_data)

if __name__ == "__main__":
    main()
