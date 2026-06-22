"""
데이터 전처리 및 EDA (탐색적 데이터 분석)
OTT 서비스 고객 이탈 예측 프로젝트
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import os
font_path = None
for f in fm.findSystemFonts():
    if 'Nanum' in f or 'nanum' in f:
        font_path = f
        break

if font_path:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

<<<<<<< HEAD
# Get project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 1. 데이터 로드 ───────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'korea_telecom_churn.csv'), encoding='utf-8-sig')
=======
# ─── 1. 데이터 로드 ───────────────────────────────────────────────────────────
# df = pd.read_csv('C:/project_file/churn_project_ott_v2/churn_project/data/korea_telecom_churn.csv', encoding='utf-8-sig')
from sqlalchemy import create_engine

# ─── 1. 데이터 로드 (MySQL 연동) ───────────────────────────────────────────────
# 본인의 MySQL 비밀번호와 포트번호(기본 3306)를 입력하세요.
engine = create_engine("mysql+pymysql://root:mysql80@localhost:3306/ott_db")

query = """
SELECT 
    u.OPID AS `고객ID`, 
    u.YEAR,
    u.ott_first, 
    u.ott_second,
    t.`Weekday usage`, 
    t.`Weekend usage`,
    m.svod
FROM ott_usage u
LEFT JOIN ott_time t ON u.OPID = t.OPID AND u.YEAR = t.YEAR
LEFT JOIN ott_money m ON u.OPID = m.OPID AND u.YEAR = m.YEAR
"""

df = pd.read_sql(query, con=engine)
>>>>>>> 74d49c4 (feat: 로컬 프로젝트 초기 커밋)
print("=" * 60)
print("1. 데이터 기본 정보")
print("=" * 60)
print(f"데이터 크기: {df.shape}")
print(f"이탈 고객 수: {df['이탈여부'].sum()} ({df['이탈여부'].mean():.2%})")

# ─── 2. 결측치 처리 ───────────────────────────────────────────────────────────
# OTT 데이터 컬럼명에 맞춰 수정
target_cols = ['월시청시간', '만족도점수', '월평균접속횟수']
for col in target_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# ─── 3. 파생변수 생성 ─────────────────────────────────────────────────────────
df['월평균요금'] = df['월정액'] * (1 - df['할인율'] / 100)
df['시청시간비율'] = df['월시청시간'] / (df['월시청시간'].max() + 1)
df['고객가치점수'] = (df['가입기간_개월'] * 0.5 + df['부가서비스수'] * 15 + df['만족도점수'] * 10)
df['CS민감도'] = df['CS문의횟수_6개월'] + df['불만접수건수'] * 3
df['연령대'] = pd.cut(df['나이'], bins=[0, 29, 39, 49, 59, 100],
                    labels=['20대', '30대', '40대', '50대', '60대이상'])

# ─── 4. EDA 시각화 ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('OTT 서비스 고객 이탈 예측 - 탐색적 데이터 분석 (EDA)', fontsize=16, fontweight='bold', y=0.98)

colors_churn = ['#2196F3', '#F44336']

# 1) 이탈 여부 분포
ax = axes[0, 0]
churn_counts = df['이탈여부'].value_counts()
ax.bar(['유지', '이탈'], churn_counts.values, color=colors_churn)
ax.set_title('고객 이탈 분포', fontweight='bold')

# 2) 요금제별 이탈률
ax = axes[0, 1]
plan_churn = df.groupby('요금제유형')['이탈여부'].mean().sort_values(ascending=False)
ax.bar(plan_churn.index, plan_churn.values * 100, color='#FF7043')
ax.set_title('요금제별 이탈률', fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

# 3) 만족도별 이탈률
ax = axes[0, 2]
sat_churn = df.groupby('만족도점수')['이탈여부'].mean()
ax.plot(sat_churn.index, sat_churn.values * 100, 'o-', color='#9C27B0')
ax.set_title('만족도 점수별 이탈률', fontweight='bold')

# 4) 가입기간 분포
ax = axes[1, 0]
df[df['이탈여부']==0]['가입기간_개월'].hist(ax=ax, bins=30, alpha=0.5, label='유지')
df[df['이탈여부']==1]['가입기간_개월'].hist(ax=ax, bins=30, alpha=0.5, label='이탈')
ax.set_title('가입기간별 이탈 분포', fontweight='bold')
ax.legend()

# 5) 서비스별 이탈률
ax = axes[1, 1]
carrier_churn = df.groupby('서비스')['이탈여부'].mean().sort_values(ascending=False)
ax.bar(carrier_churn.index, carrier_churn.values * 100, color=['#E50914', '#00A0E9', '#0063E5'])
ax.set_title('서비스별 이탈률', fontweight='bold')

# 6) 구독유형별 이탈률
ax = axes[1, 2]
contract_churn = df.groupby('구독유형')['이탈여부'].mean().sort_values(ascending=False)
ax.pie(contract_churn.values, labels=contract_churn.index, autopct='%1.1f%%')
ax.set_title('구독유형별 이탈 비율', fontweight='bold')

# 7) 연령대별 이탈률
ax = axes[2, 0]
age_churn = df.groupby('연령대', observed=True)['이탈여부'].mean()
ax.bar(age_churn.index, age_churn.values * 100, color='#00BCD4')
ax.set_title('연령대별 이탈률', fontweight='bold')

# 8) 월정액 분포
ax = axes[2, 1]
df[df['이탈여부']==0]['월정액'].hist(ax=ax, bins=30, alpha=0.5, label='유지')
df[df['이탈여부']==1]['월정액'].hist(ax=ax, bins=30, alpha=0.5, label='이탈')
ax.set_title('월정액 분포', fontweight='bold')
ax.legend()

# 9) 부가서비스 수별 이탈률
ax = axes[2, 2]
addon_churn = df.groupby('부가서비스수')['이탈여부'].mean()
ax.bar(addon_churn.index, addon_churn.values * 100, color='#8BC34A')
ax.set_title('부가서비스 수별 이탈률', fontweight='bold')

plt.tight_layout()
<<<<<<< HEAD
plt.savefig(os.path.join(BASE_DIR, 'assets', 'eda_overview.png'))
=======
plt.savefig('C:/project_file/churn_project_ott_v2/churn_project/assets/eda_overview.png')
>>>>>>> 74d49c4 (feat: 로컬 프로젝트 초기 커밋)
plt.close()

# ─── 5. 상관관계 히트맵 ──────────────────────────────────────────────────────
# 숫자 컬럼 자동 선택
numeric_df = df.select_dtypes(include=[np.number])
corr_df = numeric_df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
<<<<<<< HEAD
plt.savefig(os.path.join(BASE_DIR, 'assets', 'correlation_heatmap.png'))
=======
plt.savefig('C:/project_file/churn_project_ott_v2/churn_project/assets/correlation_heatmap.png')
>>>>>>> 74d49c4 (feat: 로컬 프로젝트 초기 커밋)
plt.close()

# ─── 6. 인코딩 및 스케일링 ────────────────────────────────────────────────────
df_model = df.copy()
df_model.drop(columns=['고객ID', '연령대'], inplace=True)

cat_cols = df_model.select_dtypes(include=['object']).columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

<<<<<<< HEAD
joblib.dump(le_dict, os.path.join(BASE_DIR, 'models', 'label_encoders.pkl'))
=======
joblib.dump(le_dict, 'C:/project_file/churn_project_ott_v2/churn_project/models/label_encoders.pkl')
>>>>>>> 74d49c4 (feat: 로컬 프로젝트 초기 커밋)

feature_cols = [c for c in df_model.columns if c != '이탈여부']
X = df_model[feature_cols]
y = df_model['이탈여부']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

<<<<<<< HEAD
joblib.dump(scaler, os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
joblib.dump(feature_cols, os.path.join(BASE_DIR, 'models', 'feature_cols.pkl'))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(BASE_DIR, 'models', 'train_test_split.pkl'))

df_model.to_csv(os.path.join(BASE_DIR, 'data', 'processed_data.csv'), index=False, encoding='utf-8-sig')
=======
joblib.dump(scaler, 'C:/project_file/churn_project_ott_v2/churn_project/models/scaler.pkl')
joblib.dump(feature_cols, 'C:/project_file/churn_project_ott_v2/churn_project/models/feature_cols.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
joblib.dump((X_train, X_test, y_train, y_test), 'C:/project_file/churn_project_ott_v2/churn_project/models/train_test_split.pkl')

df_model.to_csv('C:/project_file/churn_project_ott_v2/churn_project/data/processed_data.csv', index=False, encoding='utf-8-sig')
>>>>>>> 74d49c4 (feat: 로컬 프로젝트 초기 커밋)
print("전처리 및 EDA 완료")
