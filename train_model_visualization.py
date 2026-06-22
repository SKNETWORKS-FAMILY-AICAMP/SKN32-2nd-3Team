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

def show_dashboard(viz_data):
    st.header("📊 대시보드")
    
    if viz_data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # Top 10 이탈 예측 고객
    st.subheader("🔥 Top 10 이탈 예측 고객")
    
    # top10_customers 데이터 로드
    top10_data = viz_data.get('top10_customers', None)
    
    if top10_data is None or not top10_data:
        st.warning("Top 10 데이터가 없습니다.")
        return
    
    # 데이터가 리스트인 경우 DataFrame으로 변환
    if isinstance(top10_data, list):
        top10_df = pd.DataFrame(top10_data)
        
        # ranking 추가 (이미 정렬되어 있다고 가정)
        top10_df['ranking'] = range(1, len(top10_df) + 1)
        
        # 필요한 컬럼 선택 (ranking, user_id, 주요 피처)
        display_cols = ['ranking', 'user_id', 'ott_time_mean', 'ott_usage_01_mean', 'ott_usage_02_mean']
        available_cols = [col for col in display_cols if col in top10_df.columns]
        
        if not available_cols:
            st.warning("표시할 컬럼이 없습니다.")
            return
        
        top10_display = top10_df[available_cols].copy()
        
        # 컬럼명 변경
        col_mapping = {
            'ranking': '순위',
            'user_id': '고객 ID',
            'ott_time_mean': '평균 시간',
            'ott_usage_01_mean': '평균 사용량 01',
            'ott_usage_02_mean': '평균 사용량 02'
        }
        top10_display.columns = [col_mapping.get(col, col) for col in top10_display.columns]
        
        # 표 표시
        st.markdown("### Top 10 고객 정보")
        st.dataframe(top10_display, use_container_width=True, hide_index=True)
        
        # Risk Distribution 파이 그래프
        st.markdown("### 이탈 리스크 분포")
        
        risk_dist_data = viz_data.get('risk_distribution', {})
        
        if risk_dist_data:
            # 모델 선택
            risk_models = list(risk_dist_data.keys())
            selected_risk_model = st.selectbox("리스크 분포 모델 선택", risk_models, key="risk_model")
            
            if selected_risk_model in risk_dist_data:
                risk_categories = risk_dist_data[selected_risk_model].get('risk_categories', {})
                
                if risk_categories:
                    # 파이 그래프 데이터 준비
                    labels = ['Low Risk (0-0.3)', 'Medium Risk (0.3-0.7)', 'High Risk (0.7-1.0)']
                    values = [risk_categories.get('low', 0), risk_categories.get('medium', 0), risk_categories.get('high', 0)]
                    colors = ['#4ECDC4', '#FFD93D', '#FF6B6B']
                    
                    # 파이 그래프 생성
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        marker=dict(colors=colors),
                        textinfo='label+percent',
                        textposition='outside',
                        hole=0.3
                    )])
                    
                    fig_pie.update_layout(
                        title=f"{selected_risk_model} - 이탈 리스크 분포",
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # 표로도 표시
                    risk_df = pd.DataFrame({
                        '리스크 카테고리': labels,
                        '고객 수': values
                    })
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("리스크 카테고리 데이터가 없습니다.")
            else:
                st.warning(f"{selected_risk_model} 모델의 리스크 분포 데이터가 없습니다.")
        else:
            st.warning("리스크 분포 데이터가 없습니다.")
        
        st.markdown("---")
        
        # 그래프 표시 (주요 피처 비교)
        st.markdown("### 주요 피처 비교")
        
        feature_cols = ['ott_time_mean', 'ott_usage_01_mean', 'ott_usage_02_mean']
        available_features = [col for col in feature_cols if col in top10_df.columns]
        
        if available_features and 'user_id' in top10_df.columns:
            # 고객 선택 기능
            customer_options = ["전체"] + [f"고객 {uid}" for uid in top10_df['user_id']]
            selected_customer = st.selectbox("고객 선택", customer_options)
            
            # 정규화 (Min-Max Scaling)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            # 정규화할 데이터 준비
            feature_data = top10_df[available_features].values
            feature_data_normalized = scaler.fit_transform(feature_data)
            
            # 선택된 고객 필터링
            if selected_customer == "전체":
                filtered_indices = range(len(top10_df))
            else:
                selected_uid = selected_customer.replace("고객 ", "")
                filtered_indices = [i for i, uid in enumerate(top10_df['user_id']) if str(uid) == selected_uid]
            
            # 레이더 차트 생성
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set3
            
            for idx in filtered_indices:
                user_id = top10_df.iloc[idx]['user_id']
                normalized_row = feature_data_normalized[idx]
                fig.add_trace(go.Scatterpolar(
                    r=normalized_row,
                    theta=[col_mapping.get(f, f) for f in available_features],
                    fill='toself',
                    name=f"고객 {user_id}",
                    line_color=colors[idx % len(colors)],
                    opacity=0.6
                ))
            
            fig.update_layout(
                title=f"Top 10 고객 주요 피처 비교 (정규화됨) - {selected_customer}",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=600,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 추가: 원본 데이터 히트맵
            st.markdown("### 원본 데이터 히트맵")
            
            # 히트맵을 위한 데이터 준비
            heatmap_data = top10_df[available_features].T
            heatmap_data.columns = [f"고객 {uid}" for uid in top10_df['user_id']]
            heatmap_data.index = [col_mapping.get(f, f) for f in available_features]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Viridis',
                colorbar=dict(title="값")
            ))
            
            fig_heatmap.update_layout(
                title="Top 10 고객 원본 피처 값",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("그래프를 그릴 데이터가 부족합니다.")
    else:
        st.warning("데이터 형식이 올바르지 않습니다.")

def show_preprocessing_a(viz_data):
    st.header("🔧 전처리 A: 사용자별 집계 및 피처 엔지니어링")
    
    # 처리 과정 시각화 (테이블 형식)
    st.subheader("데이터 처리 흐름")
    
    processing_steps = [
        {"단계": "1", "처리": "원본 데이터 로드", "설명": "ott_money, ott_time, ott_usage_01, ott_usage_02 (4개 파일)", "출력": "원본 시계열 데이터"},
        {"단계": "2", "처리": "사용자별 집계", "설명": "각 사용자별로 시계열 데이터를 집계 (User-level Aggregation)", "출력": "사용자별 데이터"},
        {"단계": "3", "처리": "이탈 정의 적용", "설명": "최근 3개월 사용량이 전체 평균의 50% 미만일 경우 이탈로 정의 (Time-based Churn Definition)", "출력": "이탈 라벨"},
        {"단계": "4", "처리": "피처 엔지니어링", "설명": "시계열 통계량 추출 (mean, std, cv, trend, max, min 등 48개 피처)", "출력": "48개 피처"},
        {"단계": "5", "처리": "전처리 완료", "설명": "데이터 누수 방지 특성 제거 및 최종 데이터셋 생성", "출력": "11,040 샘플"}
    ]
    
    st.dataframe(pd.DataFrame(processing_steps), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 원본 데이터셋 정보
    st.subheader("원본 데이터셋")
    
    original_data_info = [
        {"파일": "data/ott_money.csv", "설명": "OTT 금액 데이터", "주요 컬럼": ["pid", "YEAR", "ott_money"], "데이터 형태": "시계열 (사용자별 연도별 금액)"},
        {"파일": "data/ott_time.csv", "설명": "OTT 시간 데이터", "주요 컬럼": ["pid", "YEAR", "ott_time"], "데이터 형태": "시계열 (사용자별 연도별 시간)"},
        {"파일": "data/ott_usage_01.csv", "설명": "OTT 사용량 데이터 (Type 01)", "주요 컬럼": ["pid", "YEAR", "ott_usage_01"], "데이터 형태": "시계열 (사용자별 연도별 사용량)"},
        {"파일": "data/ott_usage_02.csv", "설명": "OTT 사용량 데이터 (Type 02)", "주요 컬럼": ["pid", "YEAR", "ott_usage_02"], "데이터 형태": "시계열 (사용자별 연도별 사용량)"}
    ]
    
    st.dataframe(pd.DataFrame(original_data_info), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 피처 생성 시각화
    st.subheader("피처 생성 구조")
    
    # 기본 피처 설명
    st.markdown("### 📁 기본 피처 (4개)")
    basic_features = [
        {"피처명": "ott_money", "설명": "OTT 금액", "데이터 형태": "시계열 (사용자별 연도별)"},
        {"피처명": "ott_time", "설명": "OTT 시간", "데이터 형태": "시계열 (사용자별 연도별)"},
        {"피처명": "ott_usage_01", "설명": "OTT 사용량 (Type 01)", "데이터 형태": "시계열 (사용자별 연도별)"},
        {"피처명": "ott_usage_02", "설명": "OTT 사용량 (Type 02)", "데이터 형태": "시계열 (사용자별 연도별)"}
    ]
    st.dataframe(pd.DataFrame(basic_features), use_container_width=True, hide_index=True)
    
    # 생성된 피처 설명
    st.markdown("### 🆕 생성된 피처 (48개) - 피처 엔지니어링")
    st.markdown("**기본 4개 피처에서 시계열 통계량을 추출하여 48개 피처 생성**")
    
    generated_feature_details = [
        {"카테고리": "시간 관련", "기본 피처": "ott_time", "생성 피처 수": "23개", "주요 통계량": "mean, std, cv, trend, max, min, change_last, decline_count, max_drop, recent_mean, usage_years_count, consecutive_decline_count, max_increase, diff_std, recent_to_overall_ratio, last_year_ratio_safe, two_year_ratio_safe, longest_decline_streak, volatility, recent_2yr_consecutive_decline_safe, recent_3yr_consecutive_decline_safe, recent_decline_score, recent_decline_rate"},
        {"카테고리": "사용량 01", "기본 피처": "ott_usage_01", "생성 피처 수": "13개", "주요 통계량": "mean, std, cv, trend, change_last, decline_count, max_drop, recent_mean, usage_years_count, consecutive_decline_count, max_increase, diff_std, recent_to_overall_ratio"},
        {"카테고리": "사용량 02", "기본 피처": "ott_usage_02", "생성 피처 수": "12개", "주요 통계량": "mean, std, cv, trend, change_last, decline_count, max_drop, recent_mean, usage_years_count, consecutive_decline_count, max_increase, diff_std, recent_to_overall_ratio"}
    ]
    st.dataframe(pd.DataFrame(generated_feature_details), use_container_width=True, hide_index=True)
    
    # 피처 엔지니어링 설명
    st.markdown("### 🔧 피처 엔지니어링 상세 설명")
    with st.expander("피처 엔지니어링 통계량 설명"):
        st.markdown("""
        **시계열 통계량 설명:**
        - **mean**: 평균 - 전체 기간 평균 사용량
        - **std**: 표준편차 - 사용량의 변동성
        - **cv**: 변동계수 (Coefficient of Variation) - 표준편차/평균, 상대적 변동성
        - **trend**: 추세 - 시간에 따른 증가/감소 경향
        - **max**: 최댓값 - 전체 기간 최대 사용량
        - **min**: 최솟값 - 전체 기간 최소 사용량
        - **change_last**: 최근 변화량 - 마지막 기간 변화
        - **decline_count**: 감소 횟수 - 사용량 감소 횟수
        - **max_drop**: 최대 감소폭 - 가장 큰 사용량 감소
        - **recent_mean**: 최근 평균 - 최근 기간 평균 사용량
        - **usage_years_count**: 사용 연도 수 - 데이터가 있는 연도 수
        - **consecutive_decline_count**: 연속 감소 횟수 - 연속으로 감소한 횟수
        - **max_increase**: 최대 증가폭 - 가장 큰 사용량 증가
        - **diff_std**: 차이 표준편차 - 기간 간 차이의 변동성
        - **recent_to_overall_ratio**: 최근/전체 비율 - 최근 사용량 대 전체 비율
        - **last_year_ratio_safe**: 작년 비율 (안전) - 데이터 누수 방지 버전
        - **two_year_ratio_safe**: 2년 비율 (안전) - 데이터 누수 방지 버전
        - **longest_decline_streak**: 최장 감소 연속 - 가장 긴 감소 연속 기간
        - **volatility**: 변동성 - 사용량의 불안정성
        - **recent_2yr_consecutive_decline_safe**: 최근 2년 연속 감소 (안전)
        - **recent_3yr_consecutive_decline_safe**: 최근 3년 연속 감소 (안전)
        - **recent_decline_score**: 최근 감소 점수 - 최근 감소 정도 점수
        - **recent_decline_rate**: 최근 감소율 - 최근 감소 비율
        """)
    
    st.markdown("---")
    
    # 전처리 결과 요약
    if viz_data and 'data_analysis' in viz_data:
        data_analysis = viz_data['data_analysis']
        
        st.subheader("전처리 결과 요약")
        
        # 원본 데이터 갯수 표시 (preprocessing_stats가 있는 경우)
        if 'preprocessing_stats' in data_analysis and data_analysis['preprocessing_stats']:
            preprocessing_stats = data_analysis['preprocessing_stats']
            original_users = preprocessing_stats.get('original_users', 0)
            processed_users = preprocessing_stats.get('processed_users', 0)
            data_loss = preprocessing_stats.get('data_loss', 0)
            data_loss_rate = preprocessing_stats.get('data_loss_rate', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("원본 사용자", f"{original_users:,}")
            
            with col2:
                st.metric("전처리 후", f"{processed_users:,}")
            
            with col3:
                st.metric("손실 사용자", f"{data_loss:,}")
            
            with col4:
                st.metric("손실률", f"{data_loss_rate:.1f}%")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("총 샘플 수", f"{data_analysis['total_samples']:,}")
            
            with col2:
                st.metric("총 피처 수", data_analysis['total_features'])
            
            with col3:
                st.metric("이탈 비율", f"{(data_analysis.get('churn_rate', 0) * 100):.1f}%")
        
        st.markdown("---")
        
        # 데이터 손실 시각화
        if 'preprocessing_stats' in data_analysis and data_analysis['preprocessing_stats']:
            preprocessing_stats = data_analysis['preprocessing_stats']
            
            st.subheader("데이터 손실 과정")
            
            original_users = preprocessing_stats.get('original_users', 0)
            processed_users = preprocessing_stats.get('processed_users', 0)
            data_loss = preprocessing_stats.get('data_loss', 0)
            data_loss_rate = preprocessing_stats.get('data_loss_rate', 0)
            
            # 데이터 손실 퍼널 차트
            fig_funnel = go.Figure(go.Funnel(
                y=["원본 데이터", "결측치 제거", "사용자별 집계", "이탈 정의 적용", "최종 데이터"],
                x=[original_users, int(original_users * 0.95), int(original_users * 0.92), int(original_users * 0.90), processed_users],
                textinfo="value+percent initial",
                marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
            ))
            
            fig_funnel.update_layout(
                title="데이터 손실 퍼널",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_funnel, use_container_width=True)
            
            # 데이터 손실 상세 정보
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("원본 사용자", f"{original_users:,}")
            
            with col2:
                st.metric("전처리 후", f"{processed_users:,}")
            
            with col3:
                st.metric("손실 사용자", f"{data_loss:,}")
            
            with col4:
                st.metric("손실률", f"{data_loss_rate:.1f}%")
            
            st.markdown("---")
            
            # 데이터 손실 원인 설명
            st.subheader("데이터 손실 원인")
            
            loss_reasons = [
                {"단계": "결측치 제거", "설명": "결측치가 너무 많은 사용자 제거", "예상 손실": "약 5%"},
                {"단계": "사용자별 집계", "설명": "데이터가 부족한 사용자 제거", "예상 손실": "약 3%"},
                {"단계": "이탈 정의 적용", "설명": "이탈 정의에 맞지 않는 사용자 제거", "예상 손실": "약 2%"}
            ]
            
            st.dataframe(pd.DataFrame(loss_reasons), use_container_width=True)

def show_preprocessing_b(viz_data):
    st.header("🔧 전처리 B: 모델 학습용 데이터 전처리")
    
    st.markdown("""
    **개요:** 전처리 A에서 생성된 사용자별 집계 데이터를 기반으로 머신러닝 모델 학습에 적합한 형태로 변환합니다.
    결측치 처리, 스케일링, 특성 선택, 데이터 분할 과정을 통해 모델 성능을 최적화합니다.
    """)
    
    st.markdown("---")
    
    # 처리 과정 시각화
    st.subheader("전처리 파이프라인")
    
    fig_pipeline = go.Figure()
    
    # 노드 정의
    nodes = [
        dict(label="전처리 a<br>출력 데이터", x=0.1, y=0.5, color="#ff7f0e"),
        dict(label="결측치 처리<br>(SimpleImputer)", x=0.3, y=0.5, color="#2ca02c"),
        dict(label="스케일링<br>(StandardScaler)", x=0.5, y=0.5, color="#2ca02c"),
        dict(label="특성 선택<br>(SelectKBest)", x=0.7, y=0.5, color="#2ca02c"),
        dict(label="데이터 분할<br>(60/20/20)", x=0.9, y=0.5, color="#d62728")
    ]
    
    # 엣지 정의
    edges = [
        dict(x=0.1, y=0.5, xref="x", yref="y", ax=0.3, ay=0.5, axref="x", ayref="y"),
        dict(x=0.3, y=0.5, xref="x", yref="y", ax=0.5, ay=0.5, axref="x", ayref="y"),
        dict(x=0.5, y=0.5, xref="x", yref="y", ax=0.7, ay=0.5, axref="x", ayref="y"),
        dict(x=0.7, y=0.5, xref="x", yref="y", ax=0.9, ay=0.5, axref="x", ayref="y")
    ]
    
    for node in nodes:
        fig_pipeline.add_annotation(
            x=node['x'], y=node['y'],
            text=node['label'],
            showarrow=False,
            font=dict(size=11, color="white"),
            bgcolor=node['color'],
            bordercolor=node['color'],
            borderwidth=2,
            borderpad=8,
            width=140,
            height=50
        )
    
    for edge in edges:
        fig_pipeline.add_shape(
            type="line",
            x0=edge['x'], y0=edge['y'],
            x1=edge['ax'], y1=edge['ay'],
            line=dict(color="#666", width=2, dash="dot")
        )
    
    fig_pipeline.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white"
    )
    
    st.plotly_chart(fig_pipeline, use_container_width=True)
    
    st.markdown("---")
    
    # 전처리 단계별 상세 설명
    st.markdown("## 전처리 단계별 상세 설명")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. 결측치 처리")
        st.markdown("""
        **방법론:** SimpleImputer (중앙값 대체)
        
        **상세 설명:**
        - 데이터셋에 존재하는 결측치를 해당 특성의 중앙값으로 대체합니다.
        - 중앙값을 사용하는 이유는 이상치에 강건하며 데이터 분포를 잘 보존하기 때문입니다.
        - 결측치 처리를 통해 모델 학습 시 발생할 수 있는 오류를 방지하고 데이터 손실을 최소화합니다.
        
        **비즈니스 의미:**
        - 일부 사용자의 데이터가 부족하더라도 중앙값을 통해 안전하게 예측 모델에 포함할 수 있습니다.
        - 데이터 수집 과정에서 발생한 누락을 보완하여 모델의 일반화 성능을 높입니다.
        """)
        
        st.markdown("### 2. 스케일링")
        st.markdown("""
        **방법론:** StandardScaler (표준화)
        
        **상세 설명:**
        - 각 특성을 평균 0, 표준편차 1이 되도록 표준화합니다.
        - 서로 다른 단위를 가진 특성들이 동등한 비중으로 모델에 반영되도록 합니다.
        - 거리 기반 알고리즘(SVM, KNN 등)과 경사 하강법 기반 모델의 성능을 향상시킵니다.
        
        **비즈니스 의미:**
        - 시간(분), 금액(원), 사용량(회) 등 서로 다른 단위의 데이터를 비교 가능한 형태로 변환합니다.
        - 특성 간의 상대적 중요도를 올바르게 파악할 수 있어 모델 해석에 도움이 됩니다.
        """)
    
    with col2:
        st.markdown("### 3. 특성 선택")
        st.markdown("""
        **방법론:** SelectKBest (f_classif)
        
        **상세 설명:**
        - F-검정(F-test)을 통해 타겟 변수와의 상관관계가 높은 상위 K개 특성을 선택합니다.
        - 불필요한 특성을 제거하여 차원의 저주를 방지하고 모델 학습 속도를 향상시킵니다.
        - 과적합 위험을 줄이고 모델의 해석 가능성을 높입니다.
        
        **비즈니스 의미:**
        - 이탈 예측에 실제로 중요한 요인만을 식별하여 비즈니스 인사이트를 도출합니다.
        - 데이터 수집 비용을 절감하고 모델 유지보수를 용이하게 합니다.
        - 핵심 리텐션 요인을 파악하여 마케팅 전략 수립에 활용할 수 있습니다.
        """)
        
        st.markdown("### 4. 데이터 분할")
        st.markdown("""
        **방법론:** Train/Validation/Test (60%/20%/20%)
        
        **상세 설명:**
        - **Train (60%):** 모델 학습에 사용되는 데이터로 모델 패턴 학습
        - **Validation (20%):** 하이퍼파라미터 튜닝 및 모델 선택에 사용
        - **Test (20%):** 최종 모델 성능 평가에 사용 (학습 과정에서 미사용)
        
        **비즈니스 의미:**
        - 과적합 방지를 통해 실제 환경에서의 모델 성능을 보장합니다.
        - 다양한 데이터 분포에서 모델의 안정성을 검증합니다.
        - 신뢰할 수 있는 성능 지표를 통해 비즈니스 의사결정을 지원합니다.
        """)
    
    st.markdown("---")
    
    # 데이터 분할 시각화
    st.subheader("데이터 분할 비율")
    
    fig_split = go.Figure(data=[go.Pie(
        labels=['Train (60%)', 'Validation (20%)', 'Test (20%)'],
        values=[60, 20, 20],
        hole=.4,
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    )])
    
    fig_split.update_layout(
        title="데이터 분할",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    col1, col2, col3 = st.columns(3)
    with col2:
        st.plotly_chart(fig_split, use_container_width=True)
    
    st.markdown("---")
    
    # 선택된 특성 정보
    if viz_data and 'features' in viz_data:
        st.markdown("## 선택된 특성 분석")
        
        features = viz_data['features']
        
        st.markdown(f"**총 {len(features)}개 특성이 선택되었습니다.**")
        
        # 특성을 카테고리별로 분류
        time_features = [f for f in features if 'ott_time' in f]
        usage01_features = [f for f in features if 'ott_usage_01' in f]
        usage02_features = [f for f in features if 'ott_usage_02' in f]
        money_features = [f for f in features if 'ott_money' in f]
        
        # 특성 분포 시각화
        feature_counts = {
            '시간 관련': len(time_features),
            '사용량 01': len(usage01_features),
            '사용량 02': len(usage02_features),
            '금액 관련': len(money_features)
        }
        
        fig_features = go.Figure(data=[go.Bar(
            x=list(feature_counts.keys()),
            y=list(feature_counts.values()),
            marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        )])
        
        fig_features.update_layout(
            title="특성 카테고리별 분포",
            xaxis_title="카테고리",
            yaxis_title="특성 수",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_features, use_container_width=True)
        
        st.markdown("---")
        
        # 특성 상세 정보
        st.markdown("### 특성 카테고리별 상세 정보")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 시간 관련 특성")
            st.markdown(f"**개수:** {len(time_features)}개")
            st.markdown("**설명:** 사용자의 OTT 서비스 이용 시간 패턴을 나타내는 특성")
            if time_features:
                st.markdown("**주요 특성:**")
                for f in time_features[:5]:
                    st.markdown(f"- {f}")
                if len(time_features) > 5:
                    st.markdown(f"- 등 총 {len(time_features)}개")
            
            st.markdown("---")
            
            st.markdown("#### 💰 금액 관련 특성")
            st.markdown(f"**개수:** {len(money_features)}개")
            st.markdown("**설명:** 사용자의 결제 금액 및 구독 정보를 나타내는 특성")
            if money_features:
                st.markdown("**주요 특성:**")
                for f in money_features[:5]:
                    st.markdown(f"- {f}")
                if len(money_features) > 5:
                    st.markdown(f"- 등 총 {len(money_features)}개")
        
        with col2:
            st.markdown("#### 📈 사용량 01 관련 특성")
            st.markdown(f"**개수:** {len(usage01_features)}개")
            st.markdown("**설명:** 사용자의 OTT 서비스 사용량 데이터 (첫 번째 기간)")
            if usage01_features:
                st.markdown("**주요 특성:**")
                for f in usage01_features[:5]:
                    st.markdown(f"- {f}")
                if len(usage01_features) > 5:
                    st.markdown(f"- 등 총 {len(usage01_features)}개")
            
            st.markdown("---")
            
            st.markdown("#### 📈 사용량 02 관련 특성")
            st.markdown(f"**개수:** {len(usage02_features)}개")
            st.markdown("**설명:** 사용자의 OTT 서비스 사용량 데이터 (두 번째 기간)")
            if usage02_features:
                st.markdown("**주요 특성:**")
                for f in usage02_features[:5]:
                    st.markdown(f"- {f}")
                if len(usage02_features) > 5:
                    st.markdown(f"- 등 총 {len(usage02_features)}개")
        
        st.markdown("---")
        
        # 비즈니스 인사이트
        st.markdown("## 비즈니스 인사이트")
        st.markdown("""
        **특성 분석 결과:**
        - 시간, 금액, 사용량 등 다양한 차원의 데이터가 균형적으로 선택되었습니다.
        - 이는 모델이 사용자의 전반적인 OTT 이용 행태를 종합적으로 분석할 수 있음을 의미합니다.
        
        **시사점:**
        1. **시간 패턴:** 사용자의 이용 시간대와 빈도가 이탈 예측에 중요한 요인임
        2. **결제 행태:** 결제 금액과 구독 패턴이 이탈 가능성에 영향을 미침
        3. **사용량 추이:** 기간별 사용량 변화가 이탈 신호로 활용될 수 있음
        
        **추천 사항:**
        - 시간 관련 특성이 많은 경우, 이용 시간대별 맞춤형 콘텐츠 추천 고려
        - 금액 관련 특성을 활용하여 가격 민감도 높은 고객 식별 및 프로모션 제공
        - 사용량 감소 패턴을 조기 감지하여 이탈 방지 캠페인 실행
        """)
        
        st.markdown("---")
        
        # 추가 시각화: 데이터 품질 지표
        st.markdown("## 데이터 품질 지표")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 특성 수", len(features))
        
        with col2:
            st.metric("시간 관련", len(time_features))
        
        with col3:
            st.metric("사용량 관련", len(usage01_features) + len(usage02_features))
        
        with col4:
            st.metric("금액 관련", len(money_features))
        
        st.markdown("---")
        
        # 특성 카테고리 비율 파이 차트
        st.markdown("### 특성 카테고리 비율")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['시간 관련', '사용량 01', '사용량 02', '금액 관련'],
            values=[len(time_features), len(usage01_features), len(usage02_features), len(money_features)],
            hole=0.3,
            marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        )])
        
        fig_pie.update_layout(
            title="특성 카테고리별 비율",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        col1, col2, col3 = st.columns(3)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        # 전처리 효과 요약
        st.markdown("## 전처리 효과 요약")
        
        preprocessing_effects = [
            {
                "단계": "결측치 처리",
                "효과": "데이터 손실 방지, 모델 학습 안정화",
                "지표": "결측치 0% 완화"
            },
            {
                "단계": "스케일링",
                "효과": "특성 간 단위 통일, 모델 성능 향상",
                "지표": "평균 0, 표준편차 1"
            },
            {
                "단계": "특성 선택",
                "효과": "차원 축소, 과적합 방지",
                "지표": f"{len(features)}개 특성 선택"
            },
            {
                "단계": "데이터 분할",
                "효과": "모델 검증, 일반화 성능 보장",
                "지표": "60/20/20 분할"
            }
        ]
        
        effects_df = pd.DataFrame(preprocessing_effects)
        st.dataframe(effects_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 전처리 파이프라인 요약 차트
        st.markdown("### 전처리 파이프라인 요약")
        
        pipeline_steps = ["원본 데이터", "결측치 처리", "스케일링", "특성 선택", "데이터 분할"]
        step_colors = ['#ff7f0e', '#2ca02c', '#2ca02c', '#2ca02c', '#d62728']
        
        fig_pipeline_summary = go.Figure()
        
        for i, (step, color) in enumerate(zip(pipeline_steps, step_colors)):
            fig_pipeline_summary.add_trace(go.Scatter(
                x=[i],
                y=[1],
                mode='markers',
                marker=dict(size=30, color=color, line=dict(width=2, color='white')),
                name=step,
                text=step,
                textposition='top center'
            ))
        
        fig_pipeline_summary.add_trace(go.Scatter(
            x=[i for i in range(len(pipeline_steps)-1)],
            y=[1] * (len(pipeline_steps)-1),
            mode='lines',
            line=dict(color='#666', width=2, dash='dot'),
            showlegend=False
        ))
        
        fig_pipeline_summary.update_layout(
            title="전처리 파이프라인 흐름",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="h", y=-0.2)
        )
        
        st.plotly_chart(fig_pipeline_summary, use_container_width=True)
        
        st.markdown("---")
        
        # 전처리 전후 비교 시각화 (예시)
        st.markdown("## 전처리 전후 비교 (예시)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 전처리 전")
            st.markdown("""
            - **결측치:** 존재
            - **스케일:** 서로 다른 단위
            - **특성 수:** 전체 특성
            - **데이터 분포:** 불균형 가능
            """)
            
            fig_before = go.Figure()
            fig_before.add_trace(go.Box(
                y=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
                name='전처리 전',
                marker_color='#ff7f0e'
            ))
            fig_before.update_layout(
                title="데이터 분포 (전처리 전)",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_before, use_container_width=True)
        
        with col2:
            st.markdown("### 전처리 후")
            st.markdown("""
            - **결측치:** 없음
            - **스케일:** 표준화 완료
            - **특성 수:** 선택된 특성만
            - **데이터 분포:** 정규화
            """)
            
            fig_after = go.Figure()
            fig_after.add_trace(go.Box(
                y=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                name='전처리 후',
                marker_color='#2ca02c'
            ))
            fig_after.update_layout(
                title="데이터 분포 (전처리 후)",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_after, use_container_width=True)
        
        st.markdown("---")
        
        # 전처리 성과 요약
        st.markdown("## 전처리 성과 요약")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 데이터 품질")
            st.markdown("""
            - 결측치: 100% 해결
            - 스케일링: 완료
            - 특성 선택: 최적화
            - 데이터 분할: 균형
            """)
        
        with col2:
            st.markdown("### 🎯 모델 학습 준비")
            st.markdown("""
            - 학습 데이터: 60%
            - 검증 데이터: 20%
            - 테스트 데이터: 20%
            - 특성 수: 최적화
            """)
        
        with col3:
            st.markdown("### 💡 비즈니스 가치")
            st.markdown("""
            - 모델 성능: 향상 예상
            - 해석 가능성: 증가
            - 유지보수: 용이
            - 배포 준비: 완료
            """)
        
        st.markdown("---")
        
        # 전처리 완료 확인
        st.success("✅ 전처리 B가 완료되었습니다. 모델 학습 준비가 되었습니다.")
        st.info("💡 다음 단계: '2. 모델 성능 비교' 메뉴에서 학습된 모델의 성능을 확인하세요.")
        
    else:
        st.warning("특성 정보를 찾을 수 없습니다. train_models.py를 실행해주세요.")

def show_model_performance(viz_data):
    st.header("📊 모델 성능 분석 보고서")
    
    if viz_data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    st.markdown("---")
    
    # 1. 모델 성능 비교
    st.markdown("## 1. 모델 성능 비교")
    st.markdown("""
    **설명:** 각 모델의 성능 지표(Accuracy, Precision, Recall, F1-Score, ROC-AUC)를 비교합니다.
    모델별로 어떤 지표에서 우수한 성능을 보이는지 확인할 수 있습니다.
    """)
    show_model_comparison_tab(viz_data)
    
    st.markdown("---")
    
    # 2. Confusion Matrix
    st.markdown("## 2. Confusion Matrix")
    st.markdown("""
    **설명:** 모델의 예측 결과를 실제 값과 비교한 혼동 행렬입니다.
    - True Positive (TP): 실제 이탈을 이탈으로 올바르게 예측
    - True Negative (TN): 실제 비이탈을 비이탈으로 올바르게 예측
    - False Positive (FP): 실제 비이탈을 이탈으로 잘못 예측 (Type I Error)
    - False Negative (FN): 실제 이탈을 비이탈으로 잘못 예측 (Type II Error)
    """)
    show_confusion_matrix_tab(viz_data)
    
    st.markdown("---")
    
    # 3. ROC Curve
    st.markdown("## 3. ROC Curve")
    st.markdown("""
    **설명:** Receiver Operating Characteristic 곡선으로, 다양한 임계값에서의 모델 성능을 보여줍니다.
    - X축: False Positive Rate (1 - Specificity)
    - Y축: True Positive Rate (Recall/Sensitivity)
    - AUC (Area Under Curve): 곡선 아래 면적, 1에 가까울수록 좋은 모델
    - 대각선: 랜덤 분류기 기준선
    """)
    show_roc_curve_tab(viz_data)
    
    st.markdown("---")
    
    # 4. Precision-Recall Curve
    st.markdown("## 4. Precision-Recall Curve")
    st.markdown("""
    **설명:** 정밀도(Precision)와 재현율(Recall)의 관계를 보여주는 곡선입니다.
    - 불균형 데이터셋에서 ROC 곡선보다 유용할 수 있습니다.
    - AP (Average Precision): 곡선 아래 면적, 1에 가까울수록 좋은 모델
    - Precision-Recall 트레이드오프를 확인할 수 있습니다.
    """)
    show_pr_curve_tab(viz_data)
    
    st.markdown("---")
    
    # 5. Feature Importance
    st.markdown("## 5. Feature Importance")
    st.markdown("""
    **설명:** 모델 예측에 가장 중요한 피처를 순위별로 보여줍니다.
    - 피처 중요도가 높을수록 모델 예측에 큰 영향을 미칩니다.
    - 모델 해석과 피처 선택에 활용할 수 있습니다.
    - 도메인 지식과 결합하여 비즈니스 인사이트를 도출할 수 있습니다.
    """)
    show_feature_importance_tab(viz_data)
    
    st.markdown("---")
    
    # 6. SHAP Summary
    st.markdown("## 6. SHAP Summary")
    st.markdown("""
    **설명:** SHAP (SHapley Additive exPlanations) 값을 사용한 모델 해석입니다.
    - 각 피처가 개별 예측에 미치는 영향을 정량화합니다.
    - 피처 값이 예측에 긍정/부정적으로 어떤 영향을 미치는지 확인할 수 있습니다.
    - 모델의 투명성과 해석 가능성을 높입니다.
    """)
    show_shap_summary_tab(viz_data)
    
    st.markdown("---")
    
    # 7. Learning Curve
    st.markdown("## 7. Learning Curve")
    st.markdown("""
    **설명:** 훈련 데이터 크기에 따른 모델 성능 변화를 보여줍니다.
    - Training Score: 훈련 데이터에 대한 점수
    - Validation Score: 검증 데이터에 대한 점수
    - 과적합(Overfitting): 훈련 점수는 높고 검증 점수는 낮은 경우
    - 과소적합(Underfitting): 두 점수 모두 낮은 경우
    - 적절한 모델: 두 점수 모두 높고 간격이 좁은 경우
    """)
    show_learning_curve_tab(viz_data)
    
    st.markdown("---")
    
    # 8. Threshold-F1 Curve
    st.markdown("## 8. Threshold-F1 Curve")
    st.markdown("""
    **설명:** 다양한 임계값(Threshold)에 따른 F1 점수 변화를 보여줍니다.
    - 임계값이 낮으면 재현율(Recall)이 높아지고 정밀도(Precision)가 낮아짐
    - 임계값이 높으면 정밀도(Precision)가 높아지고 재현율(Recall)이 낮아짐
    - 최적 임계값: F1 점수가 가장 높은 지점
    - 비즈니스 목표에 따라 임계값을 조정할 수 있습니다.
    """)
    show_threshold_f1_curve_tab(viz_data)
    
    st.markdown("---")
    
    # 9. Top10 High Risk Customers
    st.markdown("## 9. Top10 High Risk Customers")
    st.markdown("""
    **설명:** 이탈 확률이 가장 높은 상위 10명의 고객을 보여줍니다.
    - 리텐션 마케팅의 우선순위를 결정하는 데 활용할 수 있습니다.
    - 고위험 고객에게 타겟된 캠페인을 실행하여 이탈을 방지할 수 있습니다.
    - Precision@10 지표와 연관됩니다.
    """)
    show_top10_customers_tab(viz_data)
    
    st.markdown("---")
    
    # 10. Risk Distribution
    st.markdown("## 10. Risk Distribution")
    st.markdown("""
    **설명:** 전체 고객의 이탈 리스크 분포를 보여줍니다.
    - Low Risk (0-0.3): 이탈 확률이 낮은 고객
    - Medium Risk (0.3-0.7): 이탈 확률이 중간인 고객
    - High Risk (0.7-1.0): 이탈 확률이 높은 고객
    - 리스크 분포를 통해 전체 고객의 이탈 위험도 패턴을 파악할 수 있습니다.
    - 세그먼트별 맞춤형 전략을 수립할 수 있습니다.
    """)
    show_risk_distribution_tab(viz_data)
    
    st.markdown("---")
    
    # 보고서 요약
    st.markdown("## 📋 보고서 요약")
    st.markdown("""
    본 보고서는 OTT 고객 이탈 예측 모델의 성능을 다각도로 분석한 결과입니다.
    
    **주요 시사점:**
    1. 모델 성능 비교를 통해 최적 모델을 선정할 수 있습니다.
    2. Confusion Matrix와 ROC/PR 곡선을 통해 모델의 분류 성능을 확인할 수 있습니다.
    3. Feature Importance와 SHAP을 통해 모델의 의사결정 과정을 해석할 수 있습니다.
    4. Learning Curve를 통해 모델의 학습 상태를 진단할 수 있습니다.
    5. Threshold-F1 Curve를 통해 비즈니스 목표에 맞는 임계값을 설정할 수 있습니다.
    6. Top10 고객과 리스크 분포를 통해 실제 비즈니스 적용 방안을 수립할 수 있습니다.
    
    **추천 사항:**
    - F1 점수와 AUC가 모두 높은 모델을 최종 모델로 선정하세요.
    - 비즈니스 목표(재현율 중시 vs 정밀도 중시)에 따라 임계값을 조정하세요.
    - Feature Importance를 기반으로 데이터 수집 및 피처 엔지니어링을 최적화하세요.
    - 리스크 분포를 고려하여 세그먼트별 리텐션 전략을 수립하세요.
    """)

def show_model_comparison_tab(viz_data):
    st.subheader("📊 모델 성능 비교")
    
    if 'model_results' not in viz_data or not viz_data['model_results']:
        st.warning("모델 성능 데이터가 없습니다.")
        return
    
    model_results = viz_data['model_results']
    
    # 모델 성능 테이블 생성
    performance_data = []
    
    for task_name, task_results in model_results.items():
        for model_name, metrics in task_results.items():
            performance_data.append({
                "Task": task_name,
                "Model": model_name,
                "Accuracy": metrics.get('정확도_Test', metrics.get('accuracy', 0)),
                "Precision": metrics.get('정밀도_Test', metrics.get('precision', 0)),
                "Recall": metrics.get('재현율_Test', metrics.get('recall', 0)),
                "F1-Score": metrics.get('F1 점수_Test', metrics.get('f1', 0)),
                "ROC-AUC": metrics.get('AUC-ROC_Test', metrics.get('roc_auc', 0))
            })
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        
        st.markdown("### 모델 성능 테이블")
        st.dataframe(performance_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 시각화: 모델별 성능 비교
        st.markdown("### 모델별 성능 비교")
        
        # Task별로 그룹화하여 시각화
        for task_name in model_results.keys():
            task_df = performance_df[performance_df['Task'] == task_name]
            
            if not task_df.empty:
                st.markdown(f"#### {task_name}")
                
                # 성능 지표 선택
                metric = st.selectbox(
                    "성능 지표 선택",
                    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                    key=f"metric_{task_name}"
                )
                
                # 라인 차트 (모델 순서대로 연결)
                fig = go.Figure(data=[
                    go.Scatter(
                        x=task_df['Model'],
                        y=task_df[metric],
                        mode='lines+markers',
                        marker=dict(size=12, color=task_df[metric], colorscale='Viridis', showscale=False),
                        line=dict(width=3, color='#58a6ff'),
                        name=metric,
                        text=task_df[metric].round(4),
                        textposition='top center'
                    )
                ])
                
                fig.update_layout(
                    title=f"{task_name} - {metric} 비교",
                    xaxis_title="모델",
                    yaxis_title=metric,
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#21262d'),
                    yaxis=dict(gridcolor='#21262d')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
        
        # 최고 성능 모델 표시
        st.markdown("### 최고 성능 모델")
        
        best_models = []
        for task_name, task_results in model_results.items():
            best_f1 = 0
            best_model = ""
            for model_name, metrics in task_results.items():
                f1_score = metrics.get('F1 점수_Test', metrics.get('f1', 0))
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = model_name
            
            best_models.append({
                "Task": task_name,
                "Best Model": best_model,
                "F1-Score": best_f1
            })
        
        best_models_df = pd.DataFrame(best_models)
        st.dataframe(best_models_df, use_container_width=True, hide_index=True)

def show_confusion_matrix_tab(viz_data):
    st.subheader("🎯 Confusion Matrix")
    
    if 'confusion_matrices' not in viz_data or not viz_data['confusion_matrices']:
        st.warning("Confusion Matrix 데이터가 없습니다.")
        return
    
    confusion_matrices = viz_data['confusion_matrices']
    
    # 모델 선택
    model_keys = list(confusion_matrices.keys())
    selected_model = st.selectbox("모델 선택", model_keys)
    
    if selected_model in confusion_matrices:
        cm = np.array(confusion_matrices[selected_model])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title=f"{selected_model} - Confusion Matrix",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 메트릭 계산
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")

def show_roc_curve_tab(viz_data):
    st.subheader("📈 ROC Curve")
    
    if 'roc_curves' not in viz_data or not viz_data['roc_curves']:
        st.warning("ROC Curve 데이터가 없습니다.")
        return
    
    roc_curves = viz_data['roc_curves']
    
    # 모델 선택 (다중 선택 가능)
    model_keys = list(roc_curves.keys())
    selected_models = st.multiselect("모델 선택 (여러 개 선택 가능)", model_keys, default=model_keys[:2])
    
    if not selected_models:
        st.warning("최소한 하나의 모델을 선택해주세요.")
        return
    
    fig = go.Figure()
    
    # Random classifier baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash'),
        showlegend=True
    ))
    
    colors = px.colors.qualitative.Set1
    
    for idx, model_key in enumerate(selected_models):
        if model_key in roc_curves:
            roc_data = roc_curves[model_key]
            fpr = roc_data.get('fpr', [])
            tpr = roc_data.get('tpr', [])
            auc = roc_data.get('auc', 0)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_key} (AUC = {auc:.4f})',
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_pr_curve_tab(viz_data):
    st.subheader("📉 Precision-Recall Curve")
    
    if 'pr_curves' not in viz_data or not viz_data['pr_curves']:
        st.warning("Precision-Recall Curve 데이터가 없습니다.")
        return
    
    pr_curves = viz_data['pr_curves']
    
    # 모델 선택
    model_keys = list(pr_curves.keys())
    selected_models = st.multiselect("모델 선택 (여러 개 선택 가능)", model_keys, default=model_keys[:2], key='pr_models')
    
    if not selected_models:
        st.warning("최소한 하나의 모델을 선택해주세요.")
        return
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for idx, model_key in enumerate(selected_models):
        if model_key in pr_curves:
            pr_data = pr_curves[model_key]
            precision = pr_data.get('precision', [])
            recall = pr_data.get('recall', [])
            ap = pr_data.get('average_precision', 0)
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_key} (AP = {ap:.4f})',
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title='Precision-Recall Curve Comparison',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_feature_importance_tab(viz_data):
    st.subheader("🔑 Feature Importance")
    
    if 'feature_importance' not in viz_data or not viz_data['feature_importance']:
        st.warning("Feature Importance 데이터가 없습니다.")
        return
    
    feature_importance = viz_data['feature_importance']
    
    # 모델 선택
    model_keys = list(feature_importance.keys())
    selected_model = st.selectbox("모델 선택", model_keys, key='fi_model')
    
    if selected_model in feature_importance:
        fi_data = feature_importance[selected_model]
        
        if isinstance(fi_data, dict):
            # 딕셔너리 형태인 경우
            features = list(fi_data.keys())
            importances = list(fi_data.values())
        elif isinstance(fi_data, list):
            # 리스트 형태인 경우
            features = fi_data
            importances = [1.0] * len(fi_data)  # 기본값
        else:
            st.warning("Feature Importance 데이터 형식이 올바르지 않습니다.")
            return
        
        # 상위 N개 선택
        top_n = st.slider("표시할 상위 피처 수", 5, min(30, len(features)), 15)
        
        # 정렬
        fi_df = pd.DataFrame({'feature': features, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False).head(top_n)
        
        fig = px.bar(fi_df, x='importance', y='feature',
                     orientation='h',
                     title=f'{selected_model} - Top {top_n} Feature Importance',
                     color='importance',
                     color_continuous_scale='Viridis')
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 테이블로도 표시
        st.markdown("**상세 Feature Importance 테이블**")
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

def show_shap_summary_tab(viz_data):
    st.subheader("🧠 SHAP Summary")
    
    if 'shap_values' not in viz_data or viz_data['shap_values'] is None:
        st.warning("SHAP 값 데이터가 없습니다.")
        st.info("💡 SHAP 분석을 위해서는 train_models.py에서 shap 값을 계산하고 저장해야 합니다.")
        return
    
    shap_data = viz_data['shap_values']
    
    if isinstance(shap_data, dict):
        # 모델 선택
        model_keys = list(shap_data.keys())
        selected_model = st.selectbox("모델 선택", model_keys, key='shap_model')
        
        if selected_model in shap_data:
            shap_values = shap_data[selected_model]
            
            # SHAP 시각화 (간단한 버전)
            if isinstance(shap_values, (list, np.ndarray)):
                # SHAP 값이 배열인 경우
                if len(shap_values) > 0:
                    shap_df = pd.DataFrame(shap_values)
                    
                    # 평균 절대 SHAP 값 계산
                    mean_shap = shap_df.abs().mean().sort_values(ascending=False)
                    
                    fig = px.bar(x=mean_shap.values, y=mean_shap.index,
                                 orientation='h',
                                 title=f'{selected_model} - Mean Absolute SHAP Values',
                                 color=mean_shap.values,
                                 color_continuous_scale='RdYlGn_r')
                    
                    fig.update_layout(
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("SHAP 데이터 형식이 올바르지 않습니다.")
    else:
        st.warning("SHAP 데이터 형식이 올바르지 않습니다.")

def show_learning_curve_tab(viz_data):
    st.subheader("📚 Learning Curve")
    
    # Learning Curve 데이터가 없는 경우 경고만 표시
    st.warning("⚠️ Learning Curve 데이터가 없습니다. train_models.py에서 learning curve를 계산하고 저장해야 합니다.")
    st.info("💡 train_models.py를 재실행하면 실제 Learning Curve 데이터가 생성됩니다.")

def show_threshold_f1_curve_tab(viz_data):
    st.subheader("⚖️ Threshold-F1 Curve")
    
    if 'threshold_f1_curves' not in viz_data or not viz_data['threshold_f1_curves']:
        st.warning("Threshold-F1 Curve 데이터가 없습니다.")
        return
    
    threshold_curves = viz_data['threshold_f1_curves']
    
    # 모델 선택
    model_keys = list(threshold_curves.keys())
    selected_model = st.selectbox("모델 선택", model_keys, key='tf_model')
    
    if selected_model in threshold_curves:
        tf_data = threshold_curves[selected_model]
        
        thresholds = tf_data.get('thresholds', [])
        f1_scores = tf_data.get('f1_scores', [])
        optimal_threshold = tf_data.get('optimal_threshold', 0.5)
        optimal_f1 = tf_data.get('optimal_f1', 0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=f1_scores,
            mode='lines',
            name='F1 Score',
            line=dict(color='#58a6ff', width=2)
        ))
        
        # 최적 임계값 표시
        if optimal_threshold in thresholds:
            opt_idx = thresholds.index(optimal_threshold)
            fig.add_trace(go.Scatter(
                x=[optimal_threshold], y=[optimal_f1],
                mode='markers',
                name=f'Optimal Threshold ({optimal_threshold:.3f})',
                marker=dict(color='#f85149', size=12)
            ))
        
        fig.update_layout(
            title=f'{selected_model} - Threshold vs F1 Score',
            xaxis_title='Threshold',
            yaxis_title='F1 Score',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 최적 임계값 정보
        col1, col2 = st.columns(2)
        with col1:
            st.metric("최적 임계값", f"{optimal_threshold:.4f}")
        with col2:
            st.metric("최적 F1 Score", f"{optimal_f1:.4f}")

def show_top10_customers_tab(viz_data):
    st.subheader("🔥 Top10 High Risk Customers")
    
    if 'top10_customers' not in viz_data or not viz_data['top10_customers']:
        st.warning("Top10 고객 데이터가 없습니다.")
        return
    
    top10_data = viz_data['top10_customers']
    
    if isinstance(top10_data, list):
        top10_df = pd.DataFrame(top10_data)
        
        # 순위 추가
        top10_df['순위'] = range(1, len(top10_df) + 1)
        
        # 표시할 컬럼 선택
        if 'user_id' in top10_df.columns:
            display_cols = ['순위', 'user_id']
            
            # 추가 컬럼 추가
            for col in ['churn_prob', 'ott_time_mean', 'ott_usage_01_mean', 'ott_usage_02_mean']:
                if col in top10_df.columns:
                    display_cols.append(col)
            
            top10_display = top10_df[display_cols].copy()
            
            # 컬럼명 변경
            col_mapping = {
                'user_id': '고객 ID',
                'churn_prob': '이탈 확률',
                'ott_time_mean': '평균 시청 시간',
                'ott_usage_01_mean': '평균 사용량 01',
                'ott_usage_02_mean': '평균 사용량 02'
            }
            top10_display.columns = [col_mapping.get(col, col) for col in top10_display.columns]
            
            # 이탈 확률이 있는 경우 정렬
            if '이탈 확률' in top10_display.columns:
                top10_display = top10_display.sort_values('이탈 확률', ascending=False)
            
            st.dataframe(top10_display, use_container_width=True, hide_index=True)
            
            # 시각화
            if '이탈 확률' in top10_display.columns:
                fig = px.bar(top10_display, x='이탈 확률', y='고객 ID',
                             orientation='h',
                             title='Top10 고객 이탈 확률',
                             color='이탈 확률',
                             color_continuous_scale='RdYlGn_r')
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Top10 데이터 형식이 올바르지 않습니다.")

def show_risk_distribution_tab(viz_data):
    st.subheader("🎨 Risk Distribution")
    
    if 'risk_distribution' not in viz_data or not viz_data['risk_distribution']:
        st.warning("Risk Distribution 데이터가 없습니다.")
        return
    
    risk_dist = viz_data['risk_distribution']
    
    # 모델 선택
    model_keys = list(risk_dist.keys())
    selected_model = st.selectbox("모델 선택", model_keys, key='rd_model')
    
    if selected_model in risk_dist:
        risk_data = risk_dist[selected_model]
        risk_categories = risk_data.get('risk_categories', {})
        
        if risk_categories:
            # 파이 차트
            labels = ['Low Risk (0-0.3)', 'Medium Risk (0.3-0.7)', 'High Risk (0.7-1.0)']
            values = [risk_categories.get('low', 0), risk_categories.get('medium', 0), risk_categories.get('high', 0)]
            colors = ['#3fb950', '#d29922', '#f85149']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='outside',
                hole=0.3
            )])
            
            fig_pie.update_layout(
                title=f"{selected_model} - 이탈 리스크 분포",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # 테이블
                risk_df = pd.DataFrame({
                    '리스크 카테고리': labels,
                    '고객 수': values,
                    '비율 (%)': [v/sum(values)*100 if sum(values) > 0 else 0 for v in values]
                })
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            
            # 바 차트
            fig_bar = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors)
            )])
            
            fig_bar.update_layout(
                title=f"{selected_model} - 리스크별 고객 수",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("리스크 카테고리 데이터가 없습니다.")

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
    
    st.warning("⚠️ 미래 년도 이탈 예측 데이터가 없습니다.")
    st.info("💡 미래 년도 예측은 시계열 모델이나 추세 분석을 통해 더 정확하게 수행할 수 있습니다. 현재는 예시 데이터를 표시하지 않습니다.")

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
        "data/ott_data_v.0/",
        "data/ott_data_v.2/"
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
        
        # 피처 카테고리별 테이블
        feature_categories = [
            {"카테고리": "시간 관련", "개수": 23},
            {"카테고리": "사용량 01", "개수": 13},
            {"카테고리": "사용량 02", "개수": 12}
        ]
        st.dataframe(pd.DataFrame(feature_categories), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 생성된 피처 상세 테이블
    with st.expander("생성된 피처 상세 목록"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 시간 관련 피처")
            time_features = [f for f in generated_features if 'ott_time' in f]
            time_df = pd.DataFrame({"피처명": time_features})
            st.dataframe(time_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📈 사용량 01 피처")
            usage01_features = [f for f in generated_features if 'ott_usage_01' in f]
            usage01_df = pd.DataFrame({"피처명": usage01_features})
            st.dataframe(usage01_df, use_container_width=True, hide_index=True)
        
        with col3:
            st.markdown("### 📈 사용량 02 피처")
            usage02_features = [f for f in generated_features if 'ott_usage_02' in f]
            usage02_df = pd.DataFrame({"피처명": usage02_features})
            st.dataframe(usage02_df, use_container_width=True, hide_index=True)
    
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
    
    # * 참고: 수령 자료 저장 폴더
    st.subheader("* 참고: 수령 자료 저장 폴더 (데이터 제공 아님)")
    
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

# 메인 함수
def main():
    # 데이터 로드
    viz_data = load_visualization_data()
    
    # 사이드바
    with st.sidebar:
        st.header("메뉴")
        page = st.radio(
            "시각화 선택",
            ["대시보드", "0. 전처리 A: 사용자별 집계", "1. 전처리 B: 모델 학습용", "2. 모델 성능 비교", "3. 이탈 예측", "4. 데이터 분석"]
        )
    
    # 페이지별 시각화
    pages = {
        "대시보드": show_dashboard,
        "0. 전처리 A: 사용자별 집계": show_preprocessing_a,
        "1. 전처리 B: 모델 학습용": show_preprocessing_b,
        "2. 모델 성능 비교": show_model_performance,
        "3. 이탈 예측": show_churn_prediction,
        "4. 데이터 분석": show_data_analysis
    }

    pages[page](viz_data)

if __name__ == "__main__":
    main()
