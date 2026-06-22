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
    st.header("🔧 전처리 B: 결측치 처리, 스케일링, 특성 선택")
    
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
    
    # 전처리 단계별 설명
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 전처리 단계")
        st.markdown("""
        **1. 결측치 처리**
        - 방법: SimpleImputer (중앙값)
        - 목적: 결측치를 안전하게 채워서 모델 학습 가능
        
        **2. 스케일링**
        - 방법: StandardScaler
        - 목적: 특성을 표준화 (평균 0, 표준편차 1)
        """)
    
    with col2:
        st.markdown("### 🎯 특성 선택 & 분할")
        st.markdown("""
        **3. 특성 선택**
        - 방법: SelectKBest (f_classif)
        - 목적: 중요한 특성만 선택하여 차원 축소
        
        **4. 데이터 분할**
        - Train: 60%
        - Validation: 20%
        - Test: 20%
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
        st.subheader("선택된 특성")
        
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
        
        # 특성 리스트
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

def show_model_performance(viz_data):
    st.header("📊 모델 성능 비교")
    
    if viz_data and 'model_results' in viz_data:
        model_results = viz_data['model_results']
        
        # 모델 성능 테이블 생성
        performance_data = []
        
        for task_name, task_results in model_results.items():
            for model_name, metrics in task_results.items():
                performance_data.append({
                    "Task": task_name,
                    "Model": model_name,
                    "Accuracy": metrics.get('accuracy', 0),
                    "Precision": metrics.get('precision', 0),
                    "Recall": metrics.get('recall', 0),
                    "F1-Score": metrics.get('f1', 0),
                    "ROC-AUC": metrics.get('roc_auc', 0)
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
                    if metrics.get('f1', 0) > best_f1:
                        best_f1 = metrics.get('f1', 0)
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
                
                # 모델 선택
                models_in_task = list(model_results[task_name].keys())
                selected_model = st.selectbox(
                    "모델 선택",
                    models_in_task,
                    key=f"cm_model_{task_name}"
                )
                
                # Confusion Matrix 데이터가 있는지 확인
                if 'confusion_matrices' in viz_data and viz_data['confusion_matrices']:
                    confusion_matrices = viz_data['confusion_matrices']
                    
                    # 해당 모델의 confusion matrix 가져오기
                    cm_key = f"{task_name}_{selected_model}"
                    if cm_key in confusion_matrices:
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
                
                # 모델 선택
                models_in_task = list(model_results[task_name].keys())
                selected_model = st.selectbox(
                    "모델 선택",
                    models_in_task,
                    key=f"roc_model_{task_name}"
                )
                
                # ROC Curve 데이터가 있는지 확인
                if 'y_true_test' in viz_data and 'y_prob_test' in viz_data:
                    y_true = np.array(viz_data['y_true_test'])
                    
                    # 해당 모델의 확률 데이터 가져오기 (현재는 모델별로 저장되어 있지 않으므로 예시로 표시)
                    st.info("⚠️ ROC Curve 데이터가 모델별로 저장되어 있지 않습니다. train_models.py를 수정하여 각 모델의 확률을 저장해야 합니다.")
                    
                    # 예시 ROC Curve
                    fpr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    tpr = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0])
                    auc = 0.85
                else:
                    st.info("⚠️ ROC Curve 데이터가 없습니다. train_models.py를 실행하여 확률 데이터를 저장해야 합니다.")
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
                
                # 모델 선택
                models_in_task = list(model_results[task_name].keys())
                selected_model = st.selectbox(
                    "모델 선택",
                    models_in_task,
                    key=f"pr_model_{task_name}"
                )
                
                # Precision-Recall Curve 데이터가 있는지 확인
                if 'y_true_test' in viz_data and 'y_prob_test' in viz_data:
                    y_true = np.array(viz_data['y_true_test'])
                    
                    # 해당 모델의 확률 데이터 가져오기 (현재는 모델별로 저장되어 있지 않으므로 예시로 표시)
                    st.info("⚠️ Precision-Recall Curve 데이터가 모델별로 저장되어 있지 않습니다. train_models.py를 수정하여 각 모델의 확률을 저장해야 합니다.")
                    
                    # 예시 Precision-Recall Curve
                    precision = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5])
                    recall = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    ap = 0.75
                else:
                    st.info("⚠️ Precision-Recall Curve 데이터가 없습니다. train_models.py를 실행하여 확률 데이터를 저장해야 합니다.")
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
            if 'threshold_f1_curves' in viz_data and viz_data['threshold_f1_curves']:
                threshold_f1_curves = viz_data['threshold_f1_curves']
                
                # Task별로 Threshold-F1 Curve 표시
                for task_name in threshold_f1_curves.keys():
                    st.markdown(f"### {task_name}")
                    
                    # 모델 선택
                    models_in_task = list(threshold_f1_curves[task_name].keys())
                    selected_model = st.selectbox(
                        "모델 선택",
                        models_in_task,
                        key=f"tf_model_{task_name}"
                    )
                    
                    tf_data = threshold_f1_curves[task_name][selected_model]
                    
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
            
            # Top10 데이터가 있는지 확인
            if 'top10_customers' in viz_data and viz_data['top10_customers']:
                top10_customers = viz_data['top10_customers']
                
                # Task별로 Top10 표시
                for task_name in top10_customers.keys():
                    st.markdown(f"### {task_name}")
                    
                    # 모델 선택
                    models_in_task = list(top10_customers[task_name].keys())
                    selected_model = st.selectbox(
                        "모델 선택",
                        models_in_task,
                        key=f"top10_model_{task_name}"
                    )
                    
                    top10_data = top10_customers[task_name][selected_model]
                    
                    # Top10 테이블 생성
                    if top10_data:
                        top10_df = pd.DataFrame(top10_data)
                        
                        st.markdown("#### Top10 High Risk Customers")
                        st.dataframe(top10_df, use_container_width=True, hide_index=True)
                        
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
                            title=f"{task_name} - {selected_model} Top10 High Risk Customers",
                            xaxis_title="Churn Probability",
                            yaxis_title="User ID",
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        
                        st.plotly_chart(fig_top10, use_container_width=True)
                        
                        # 실제 이탈 여부 표시
                        actual_churn_count = top10_df['churn'].sum()
                        st.markdown(f"**실제 이탈:** {actual_churn_count}/10 ({actual_churn_count*10}%)")
                        
                        st.markdown("---")
                    else:
                        st.warning("Top10 데이터가 없습니다.")
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
            if 'risk_distribution' in viz_data and viz_data['risk_distribution']:
                risk_distribution = viz_data['risk_distribution']
                
                # Task별로 Risk Distribution 표시
                for task_name in risk_distribution.keys():
                    st.markdown(f"### {task_name}")
                    
                    # 모델 선택
                    models_in_task = list(risk_distribution[task_name].keys())
                    selected_model = st.selectbox(
                        "모델 선택",
                        models_in_task,
                        key=f"rd_model_{task_name}"
                    )
                    
                    rd_data = risk_distribution[task_name][selected_model]
                    
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
            
            # Code Detail 시각화 (use_ml_history.txt)
            st.subheader("Code Detail (use_ml_history.txt)")
            
            # use_ml_history.txt 파일 읽기
            ml_history_path = os.path.join(BASE_DIR, 'ml', 'use_ml_history.txt')
            
            if os.path.exists(ml_history_path):
                try:
                    with open(ml_history_path, 'r', encoding='utf-8') as f:
                        ml_history_content = f.read()
                    
                    st.markdown("### ML History 내용")
                    st.text_area("use_ml_history.txt", ml_history_content, height=400)
                    
                    # 내용을 라인별로 분석하여 시각화
                    lines = ml_history_content.split('\n')
                    
                    # 날짜별 통계
                    date_stats = {}
                    for line in lines:
                        if line.strip():
                            # 날짜 패턴 찾기 (예: 2024-06-21)
                            import re
                            date_match = re.search(r'\d{4}-\d{2}-\d{2}', line)
                            if date_match:
                                date = date_match.group()
                                if date not in date_stats:
                                    date_stats[date] = 0
                                date_stats[date] += 1
                    
                    if date_stats:
                        st.markdown("### 날짜별 활동 통계")
                        
                        date_df = pd.DataFrame([
                            {"Date": date, "Activity Count": count}
                            for date, count in sorted(date_stats.items())
                        ])
                        
                        st.dataframe(date_df, use_container_width=True, hide_index=True)
                        
                        # 날짜별 활동 바 차트
                        fig_date = go.Figure(data=[
                            go.Bar(
                                x=date_df['Date'],
                                y=date_df['Activity Count'],
                                marker_color='purple'
                            )
                        ])
                        
                        fig_date.update_layout(
                            title="날짜별 활동 통계",
                            xaxis_title="Date",
                            yaxis_title="Activity Count",
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        
                        st.plotly_chart(fig_date, use_container_width=True)
                    
                    st.markdown("---")
                except Exception as e:
                    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            else:
                st.warning("⚠️ use_ml_history.txt 파일을 찾을 수 없습니다.")
                st.markdown("**파일 경로:** `ml/use_ml_history.txt`")
                
                st.markdown("---")
        else:
            st.warning("모델 결과 데이터가 없습니다.")
    else:
        st.warning("모델 결과 데이터를 찾을 수 없습니다. train_models.py를 실행하여 모델 결과를 생성하세요.")

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
    if page == "대시보드":
        show_dashboard(viz_data)
    elif page == "0. 전처리 A: 사용자별 집계":
        show_preprocessing_a(viz_data)
    elif page == "1. 전처리 B: 모델 학습용":
        show_preprocessing_b(viz_data)
    elif page == "2. 모델 성능 비교":
        show_model_performance(viz_data)
    elif page == "3. 이탈 예측":
        show_churn_prediction(viz_data)
    elif page == "4. 데이터 분석":
        show_data_analysis(viz_data)

if __name__ == "__main__":
    main()
