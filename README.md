# SKN31-2nd-4Team

## 팀 및 팀원 소개

### 팀 명
- <b>S-Board</b>

### 팀원
| 이태혁 | 박상희 | 송채영 | 이호원 |
| :---: | :---: | :---: | :---: |
| <b>팀장</b>, 데이터 정규화 + 전처리 + 테이블 + SQLite DB 설계 | 데이터 전처리 + 이탈라벨 + 모델학습 | Face recognition (InsightFace), 로그인 시스템, 권한 관리 (admin/user)| 데이터 전처리, Decision Tree 모델링 | UI + 발표 + 정리 |

---

## 프로젝트 개요

### 프로젝트명
- 사용 패턴 붕괴 신호를 완전히 복원한 상태 기반 churn 모델을 활용한 OTT 이탈 위험도 랭킹 시스템

### 프로젝트 배경
최근 OTT 시장 경쟁이 심화되면서 신규 고객 확보보다 기존 고객 유지가 더욱 중요해지고 있습니다. 하지만 고객 수가 많아질수록 어떤 고객이 이탈할 가능성이 높은지 파악하기 어렵다는 문제가 있습니다.

이를 해결하기 위해 저희는 고객 데이터를 분석하여 이탈 가능성을 예측하고, 위험 고객을 사전에 관리할 수 있는 AI 기반 고객 이탈 예측 시스템인 S-Board를 개발하였습니다.

### 프로젝트 목표
<img src="data\pr_goal.png">

### 기능요약
가입 고객 이탈 예측 AI 시스템은 관리자 계정 및 얼굴 인증 로그인을 통해 안전한 접근 환경을 제공한다.
YouTube, Netflix, Tving 플랫폼의 이용 현황을 실시간으로 모니터링하고 고객 데이터를 분석하여 주요 지표를 시각화한다.
머신러닝 기반 고객 이탈 예측 기능을 통해 이탈 위험 고객을 선별한다.
또한 관리자 권한 분리, 고객 관리, 시스템 운영 관리 기능을 제공하여 효율적인 서비스 운영 환경을 구축한다.


### 프로젝트 구조
```
SKN32-2nd-3Team
churn_project_ott/
├── app.py                  # Streamlit 메인 앱
├── assets/                 # 로고/차트 이미지
├── data/                   # 데이터
│   ├── before/, xlsx/      # 원본 CSV, 전처리 결과, 엑셀본
│   └── *.csv, *.json       # ott 사용량, visualization_data.json
├── models/                 # train_models.py로 생성된 학습 산출물(.pkl/.keras)
├── face_data/              # 얼굴 임베딩 + admin_face.jpg + users.json (생체정보/계정정보)
├── db.py                   # .env에서 MYSQL_USER/PASSWORD, FACE_EMBEDDING_KEY 로드
├── face_auth.py / ui_styles.py / train_model_visualization.py / train_models.py / preprocess_eda.py / generate_data.py
├── ott_db.sql
├── requirements.txt, README.md
```

---

## 🛠 기술 스택

| Category | Stack |
|----------|-------|
| Language | ![Python](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| Data Processing | ![Pandas](https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white) |
| Machine Learning | ![Scikit-Learn](https://img.shields.io/badge/SCIKIT--LEARN-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBOOST-228B22?style=for-the-badge&logo=xgboost&logoColor=white) ![LightGBM](https://img.shields.io/badge/LIGHTGBM-9ACD32?style=for-the-badge) ![RandomForest](https://img.shields.io/badge/RANDOM_FOREST-228B22?style=for-the-badge) 
| Visualization | ![Matplotlib](https://img.shields.io/badge/MATPLOTLIB-11557C?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |

---

## 🏗️ 시스템 아키텍처 (System Architecture)

> [!NOTE]
> 사용자 얼굴 인증부터 데이터 분석, ML 모델 서빙까지의 전체 파이프라인입니다.

<img src="data\faceID.png">

---

## 🧠 핵심 기술 및 데이터 전처리 (Feature Engineering)

본 프로젝트의 핵심은 **고객의 사용 패턴 붕괴 신호를 감지하고 이를 복원하여 상태(State) 기반의 이탈 징후를 정의**한 점입니다.

### 1) 데이터 전처리 및 분석 정의
* **사용 패턴 붕괴 신호 정의:** OTT 플랫폼(YouTube, Netflix, Tving)의 주간/월간 접속 빈도 및 시청 시간의 급격한 감소율을 정량화 (예: 전월 대비 시청 시간 $30\%$ 이상 감소 시 붕괴 신호로 판단).
* **상태 기반(State-based) Churn 라벨링:** 단순히 '탈퇴 여부'가 아닌, '로그인 빈도 감소 상태', '결제 수단 만료 임박 상태' 등 고객의 현재 행동 상태를 다각도로 결합하여 이탈 위험군 정의.

### 2) 주요 Feature Engineering
* `Trend_Index`: 최근 2주간 이용량 변동 추이 지표.
* `Platform_Dependency`: 특정 OTT 플랫폼에 편중된 사용도 계산.
* `Engagement_Score`: 총 시청 시간, 접속 일수, 인터랙션 요소를 결합한 종합 활성 점수 생성.

---

## 🤖 모델링 및 성능 평가 (Modeling & Evaluation)

이탈 위험도를 정확하게 랭킹화하기 위해 다형성의 머신러닝 알고리즘을 학습시키고 비교 검증하였습니다.

### 1) 모델 성능 비교
| Model | Accuracy | F1-Score | AUC-ROC | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **Decision Tree** | 0.81 | 0.78 | 0.80 | Baseline |
| **Random Forest** | 0.86 | 0.84 | 0.87 | 오버피팅 경향 있음 |
| **XGBoost** | 0.89 | 0.88 | 0.91 | 우수한 예측력 |
| **LightGBM** | 0.90 | 0.89 | 0.92 | 빠른 연산 속도 |
| **CatBoost** (최종) | **0.92** | **0.91** | **0.94** | **최종 채택 (범주형 변수 최적 처리)** |

### 2) 최종 모델 선정 이유
* **CatBoost** 모델이 데이터셋 내 범주형 변수(선호 플랫폼, 주 사용 시간대 등)를 가장 효과적으로 학습하여 과적합 없이 가장 높은 F1-Score를 기록했습니다. 
* 최종 산출된 이탈 확률값을 기반으로 Streamlit 대시보드 내 **'이탈 위험도 Top N 랭킹 시스템'**을 구현하였습니다.

---

## 수행 결과
### 1) 메인 페이지 - 
<img src="data\face_login.png">
<img src="data\dashboard.png">
<img src="data\EDA.png">
<img src="data\model.png">
<img src="data\churn.png">

---

## 한 줄 회고
#### 이태혁


#### 박상희


#### 송채영


#### 이호원




---
