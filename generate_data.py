"""
OTT 서비스 고객 이탈 예측 데이터셋 생성
- OTT 서비스(Netflix, Tving, Disney+) 환경 기반 가상 데이터
- 실제 OTT 서비스 요금제, 부가서비스 반영
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
N = 5000

# 기본 인구통계 정보
age = np.random.randint(18, 75, N)
gender = np.random.choice(['남성', '여성'], N, p=[0.52, 0.48])
region = np.random.choice(
    ['서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '세종', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'],
    N,
    p=[0.20, 0.24, 0.06, 0.07, 0.05, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.05, 0.07]
)

# OTT 서비스 및 요금제
carrier = np.random.choice(['Netflix', 'Tving', 'Disney+'], N, p=[0.45, 0.30, 0.25])
plan_type = np.random.choice(['프리미엄 4K', '스탠다드 HD', '베이직', '광고형 스탠다드', '연간 멤버십'], N, p=[0.15, 0.25, 0.30, 0.20, 0.10])

# 요금제별 월정액
plan_price_map = {
    '프리미엄 4K': np.random.normal(17000, 500, N),
    '스탠다드 HD': np.random.normal(13500, 400, N),
    '베이직': np.random.normal(9500, 300, N),
    '광고형 스탠다드': np.random.normal(5500, 300, N),
    '연간 멤버십': np.random.normal(12000, 1000, N)
}
monthly_fee = np.array([plan_price_map[p][i] for i, p in enumerate(plan_type)])
monthly_fee = np.clip(monthly_fee, 5000, 20000).astype(int)

# 사용 패턴
data_usage_gb = np.random.exponential(50, N)  # 월 시청 시간 (시간)
call_minutes = np.random.normal(20, 10, N)    # 월 평균 접속 횟수
sms_count = np.random.normal(5, 3, N)        # 프로필 수
roaming_usage = np.random.choice([0, 1], N, p=[0.70, 0.30])  # 동시 접속 사용

# 계약 및 가입 정보
contract_type = np.random.choice(['연간 구독', '6개월 구독', '월간 구독'], N, p=[0.30, 0.20, 0.50])
tenure_months = np.random.randint(1, 60, N)   # 가입 기간 (개월)
num_lines = np.random.choice([1, 2, 3, 4], N, p=[0.60, 0.25, 0.10, 0.05])  # 등록 기기 수

# 부가서비스 (OTT 맥락에 맞게 변경)
has_internet = np.random.choice([0, 1], N, p=[0.40, 0.60])      # 오프라인 저장 이용
has_iptv = np.random.choice([0, 1], N, p=[0.70, 0.30])          # 키즈 프로필 이용
has_insurance = np.random.choice([0, 1], N, p=[0.80, 0.20])     # 추가 프로필 구매
has_content = np.random.choice([0, 1], N, p=[0.50, 0.50])       # 독점 콘텐츠 시청
num_addons = has_internet + has_iptv + has_insurance + has_content

# 타 OTT 사용 여부 (벤다이어그램용)
# 주 서비스가 Netflix인 경우 Tving, Disney+ 사용 확률
other_netflix = np.random.choice([0, 1], N, p=[0.6, 0.4])
other_tving = np.random.choice([0, 1], N, p=[0.7, 0.3])
other_disney = np.random.choice([0, 1], N, p=[0.8, 0.2])

# 고객 서비스 이용
cs_calls_6m = np.random.poisson(1, N)          # 최근 6개월 고객센터 문의 횟수
complaint_count = np.random.poisson(0.3, N)    # 불만 접수 건수
satisfaction_score = np.random.randint(1, 6, N)  # 만족도 (1~5)

# 결제 정보
payment_method = np.random.choice(['간편결제', '신용카드', '휴대폰결제', '정기결제'], N, p=[0.50, 0.35, 0.10, 0.05])
late_payment = np.random.choice([0, 1], N, p=[0.95, 0.05])      # 결제 실패 경험
discount_rate = np.random.choice([0, 5, 10, 15, 20, 30], N, p=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05])  # 할인율(%)

# 단말기 정보 (주로 사용하는 기기)
device_age_months = np.random.randint(0, 36, N)  # 기기 사용 기간
device_type = np.random.choice(['스마트TV', '스마트폰', '태블릿', 'PC'], N, p=[0.40, 0.35, 0.15, 0.10])

# 이탈 여부 (비즈니스 로직 반영)
churn_prob = np.zeros(N)

# 이탈 요인 반영
churn_prob += (satisfaction_score <= 2) * 0.35
churn_prob += (complaint_count >= 2) * 0.25
churn_prob += (contract_type == '월간 구독') * 0.20
churn_prob += (tenure_months < 6) * 0.15
churn_prob += (monthly_fee > 15000) * 0.10
churn_prob += (late_payment == 1) * 0.15
churn_prob += (cs_calls_6m >= 3) * 0.15
churn_prob += (num_addons == 0) * 0.10
churn_prob += (discount_rate == 0) * 0.05
churn_prob -= (contract_type == '연간 구독') * 0.20
churn_prob -= (num_addons >= 3) * 0.12
churn_prob -= (tenure_months > 24) * 0.15
churn_prob -= (satisfaction_score >= 4) * 0.20

churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn = (np.random.rand(N) < churn_prob).astype(int)

# 데이터프레임 생성
df = pd.DataFrame({
    '고객ID': [f'OTT{str(i+1).zfill(5)}' for i in range(N)],
    '나이': age,
    '성별': gender,
    '지역': region,
    '서비스': carrier,
    '요금제유형': plan_type,
    '월정액': monthly_fee,
    '월시청시간': np.round(data_usage_gb, 2),
    '월평균접속횟수': np.clip(call_minutes, 0, 100).astype(int),
    '프로필수': np.clip(sms_count, 1, 5).astype(int),
    '동시접속이용': roaming_usage,
    '구독유형': contract_type,
    '가입기간_개월': tenure_months,
    '등록기기수': num_lines,
    '오프라인저장': has_internet,
    '키즈프로필': has_iptv,
    '추가프로필구매': has_insurance,
    '독점콘텐츠시청': has_content,
    '부가서비스수': num_addons,
    'Netflix_사용': 0,
    'Tving_사용': 0,
    'Disney_사용': 0,
    'CS문의횟수_6개월': cs_calls_6m,
    '불만접수건수': complaint_count,
    '만족도점수': satisfaction_score,
    '결제방법': payment_method,
    '결제실패경험': late_payment,
    '할인율': discount_rate,
    '주사용기기기간_개월': device_age_months,
    '주사용기기': device_type,
    '이탈여부': churn
})

# 다중 사용 여부 로직 보완
for i in range(N):
    s = df.loc[i, '서비스']
    if s == 'Netflix':
        df.loc[i, 'Netflix_사용'] = 1
        df.loc[i, 'Tving_사용'] = other_tving[i]
        df.loc[i, 'Disney_사용'] = other_disney[i]
    elif s == 'Tving':
        df.loc[i, 'Tving_사용'] = 1
        df.loc[i, 'Netflix_사용'] = other_netflix[i]
        df.loc[i, 'Disney_사용'] = other_disney[i]
    elif s == 'Disney+':
        df.loc[i, 'Disney_사용'] = 1
        df.loc[i, 'Netflix_사용'] = other_netflix[i]
        df.loc[i, 'Tving_사용'] = other_tving[i]

# 저장 (기존 파일명 유지)
df.to_csv('D:/Personal/P-PJT/Team-PJT_/churn_project_new/churn_project/data/korea_telecom_churn.csv', index=False, encoding='utf-8-sig')
print(f"데이터 생성 완료: {df.shape}")
print(f"이탈률: {df['이탈여부'].mean():.2%}")
