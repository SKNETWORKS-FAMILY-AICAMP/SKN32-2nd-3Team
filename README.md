# 🎬 OTT 서비스 고객 이탈 예측 시스템

> **OTT Analytics Platform v2.0** | AI 기반 고객 이탈 예측 대시보드

---

## 프로젝트 개요

글로벌 및 국내 OTT 서비스(Netflix, Tving, Disney+) 고객 데이터를 기반으로 **머신러닝/딥러닝 모델**을 활용하여 고객 이탈을 예측하고, **페이스 로그인(얼굴 인식)** 기능과 **현업 수준의 Streamlit 대시보드**를 통해 분석 결과를 시각화하는 시스템입니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **페이스 로그인** | OpenCV 기반 얼굴 인식 인증 (카메라 촬영 → 얼굴 감지 → 임베딩 비교) |
| **계정 로그인** | ID/PW 기반 로그인 (admin/analyst/viewer 역할 구분) |
| **대시보드** | KPI 카드, 월별 트렌드, 서비스별/요금제별/연령대별 이탈 현황 |
| **EDA 분석** | 분포 분석, 상관관계 히트맵, 세그먼트 분석, 데이터 요약 |
| **모델 성능** | 7개 모델 비교, ROC 곡선, 레이더 차트, 특성 중요도 |
| **이탈 예측** | 개별 고객 예측 (게이지 차트 + 맞춤형 전략), 배치 예측 |
| **고객 관리** | 검색/필터링, 위험도 분류, 고위험 고객 목록 |
| **시스템 설정** | 사용자 관리, 모델 파라미터, 시스템 정보 |

---

## 데이터셋

- **규모**: 5,000명 고객 데이터
- **특성 수**: 28개 (인구통계, 요금제, 시청패턴, 구독정보, 부가서비스, CS이력)
- **주요 서비스**: Netflix, Tving, Disney+
- **요금제**: 프리미엄 4K, 스탠다드 HD, 베이직, 광고형 스탠다드, 연간 멤버십

### 주요 변수

| 변수 | 설명 |
|------|------|
| 서비스 | Netflix, Tving, Disney+ |
| 요금제유형 | 프리미엄 4K, 스탠다드 HD, 베이직, 광고형 스탠다드, 연간 멤버십 |
| 구독유형 | 연간 구독, 6개월 구독, 월간 구독 |
| 만족도점수 | 1~5점 (이탈 예측 최고 중요 변수) |
| 월시청시간 | 월간 총 시청 시간 (시간 단위) |
| 부가서비스수 | 오프라인 저장/키즈 프로필/추가 프로필/독점 콘텐츠 이용 수 |

---

## 실행 방법

### 1. 환경 설치

```bash
pip install streamlit scikit-learn xgboost lightgbm imbalanced-learn \
            shap plotly opencv-python-headless tensorflow sqlalchemy PyMySQL matplotlib-venn
```

### 2. 데이터 생성 및 전처리

```bash
python3 generate_data.py
python3 preprocess_eda.py
```

### 3. 모델 학습

```bash
python3 train_models.py
```

### 4. 앱 실행

```bash
streamlit run app.py
```

---

*© 2024 OTT Analytics Platform | 부트캠프 프로젝트*
