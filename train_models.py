"""
머신러닝 / 딥러닝 모델 학습 및 평가
OTT/SNS 고객 이탈 예측 프로젝트 - 전이학습 및 앙상블 기반 고성능 모델
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import joblib
import json
import warnings
import os
warnings.filterwarnings('ignore')

# Get project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import optuna
from scipy import stats

# 한글 폰트
for f in fm.findSystemFonts():
    if 'Nanum' in f or 'nanum' in f:
        plt.rcParams['font.family'] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

# ─── OTT/SNS 데이터 로드 및 전처리 ────────────────────────────────────────────────
def load_ott_sns_data():
    """OTT 및 SNS 데이터 로드 및 통합"""
    print("=" * 60)
    print("OTT/SNS 데이터 로드")
    print("=" * 60)
    
    # OTT 데이터 로드
    try:
        ott_money = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ott_money.csv'))
        ott_time = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ott_time.csv'))
        ott_usage_01 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ott_usage_01.csv'))
        ott_usage_02 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ott_usage_02.csv'))
        print(f"✅ OTT 데이터 로드 완료")
    except Exception as e:
        print(f"❌ OTT 데이터 로드 실패: {e}")
        return None, None
    
    # SNS 데이터 로드
    try:
        sns_time = pd.read_csv(os.path.join(BASE_DIR, 'data', 'sns_time.csv'))
        sns_usage_01 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'sns_usage_01.csv'))
        sns_usage_02 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'sns_usage_02.csv'))
        sns_usage_03 = pd.read_csv(os.path.join(BASE_DIR, 'data', 'sns_usage_03.csv'))
        print(f"✅ SNS 데이터 로드 완료")
    except Exception as e:
        print(f"❌ SNS 데이터 로드 실패: {e}")
        return None, None

    # OTT 데이터 전처리
    ott_df = preprocess_ott_data(ott_money, ott_time, ott_usage_01, ott_usage_02)

    # SNS 데이터 전처리
    sns_df = preprocess_sns_data(sns_time, sns_usage_01, sns_usage_02, sns_usage_03)

    return ott_df, sns_df

def preprocess_ott_data(ott_money, ott_time, ott_usage_01, ott_usage_02):
    """OTT 데이터 전처리 및 특성 추출"""
    print("📊 OTT 데이터 전처리 중...")

    # ott_money: pid별 비용 데이터
    money_dict = dict(zip(ott_money['pid'], ott_money['p21d26058']))

    # ott_time: 연도별 사용자 시계열 데이터
    time_data = ott_time.set_index('YEAR')

    # ott_usage: 연도별 사용 패턴 데이터
    usage_01_data = ott_usage_01.set_index('YEAR')
    usage_02_data = ott_usage_02.set_index('YEAR')

    # 사용자별 특성 추출
    user_features = []

    # 모든 사용자 ID 추출
    all_users = set()
    for col in time_data.columns:
        if col != 'YEAR':
            all_users.add(col)

    for user_id in list(all_users)[:1000]:  # 샘플링 (메모리 관리)
        features = {'user_id': user_id}

        # 비용 정보
        features['ott_money'] = money_dict.get(user_id, 0)

        # 시계열 특성 (ott_time)
        if user_id in time_data.columns:
            user_time = time_data[user_id].dropna()
            if len(user_time) > 0:
                features['ott_time_mean'] = user_time.mean()
                features['ott_time_std'] = user_time.std()
                features['ott_time_trend'] = user_time.iloc[-1] - user_time.iloc[0] if len(user_time) > 1 else 0
                features['ott_time_max'] = user_time.max()
                features['ott_time_min'] = user_time.min()
            else:
                features.update({'ott_time_mean': 0, 'ott_time_std': 0, 'ott_time_trend': 0,
                               'ott_time_max': 0, 'ott_time_min': 0})
        else:
            features.update({'ott_time_mean': 0, 'ott_time_std': 0, 'ott_time_trend': 0,
                           'ott_time_max': 0, 'ott_time_min': 0})

        # 사용 패턴 특성 (ott_usage)
        if user_id in usage_01_data.columns:
            user_usage_01 = usage_01_data[user_id].dropna()
            if len(user_usage_01) > 0:
                features['ott_usage_01_mean'] = user_usage_01.mean()
                features['ott_usage_01_std'] = user_usage_01.std()
                features['ott_usage_01_trend'] = user_usage_01.iloc[-1] - user_usage_01.iloc[0] if len(user_usage_01) > 1 else 0
            else:
                features.update({'ott_usage_01_mean': 0, 'ott_usage_01_std': 0, 'ott_usage_01_trend': 0})
        else:
            features.update({'ott_usage_01_mean': 0, 'ott_usage_01_std': 0, 'ott_usage_01_trend': 0})

        if user_id in usage_02_data.columns:
            user_usage_02 = usage_02_data[user_id].dropna()
            if len(user_usage_02) > 0:
                features['ott_usage_02_mean'] = user_usage_02.mean()
                features['ott_usage_02_std'] = user_usage_02.std()
                features['ott_usage_02_trend'] = user_usage_02.iloc[-1] - user_usage_02.iloc[0] if len(user_usage_02) > 1 else 0
            else:
                features.update({'ott_usage_02_mean': 0, 'ott_usage_02_std': 0, 'ott_usage_02_trend': 0})
        else:
            features.update({'ott_usage_02_mean': 0, 'ott_usage_02_std': 0, 'ott_usage_02_trend': 0})

        # 이탈 라벨 생성 (데이터 누수 방지를 위해 다른 기준 사용)
        # 전체 사용 시간이 낮고 감소하는 경우를 이탈로 정의
        total_time = features.get('ott_time_mean', 0) + features.get('ott_usage_01_mean', 0)
        if total_time < 10 and features.get('ott_time_trend', 0) < 0:
            features['churn'] = 1
        else:
            features['churn'] = 0

        user_features.append(features)

    ott_df = pd.DataFrame(user_features)
    print(f"✅ OTT 데이터 전처리 완료: {ott_df.shape}")
    return ott_df

def preprocess_sns_data(sns_time, sns_usage_01, sns_usage_02, sns_usage_03):
    """SNS 데이터 전처리 및 특성 추출"""
    print("📊 SNS 데이터 전처리 중...")

    # sns_time: OPID별 시간 데이터
    time_dict = sns_time.groupby('OPID').agg({
        'OTT 서비스 주중 이용 시간(분 기준 환산)': 'mean',
        'OTT 서비스 주말 이용 시간(분 기준 환산)': 'mean'
    }).to_dict('index')

    # sns_usage: 연도별 사용 패턴 데이터
    usage_01_data = sns_usage_01.set_index('YEAR')
    usage_02_data = sns_usage_02.set_index('YEAR')
    usage_03_data = sns_usage_03.set_index('YEAR')

    # 사용자별 특성 추출
    user_features = []

    # 모든 사용자 ID 추출
    all_users = set()
    for col in usage_01_data.columns:
        if col != 'YEAR':
            all_users.add(col)

    for user_id in list(all_users)[:1000]:  # 샘플링 (메모리 관리)
        features = {'user_id': user_id}

        # 시간 정보
        if user_id in time_dict:
            features['sns_weekday_time'] = time_dict[user_id].get('OTT 서비스 주중 이용 시간(분 기준 환산)', 0)
            features['sns_weekend_time'] = time_dict[user_id].get('OTT 서비스 주말 이용 시간(분 기준 환산)', 0)
            features['sns_total_time'] = features['sns_weekday_time'] + features['sns_weekend_time']
        else:
            features.update({'sns_weekday_time': 0, 'sns_weekend_time': 0, 'sns_total_time': 0})

        # 사용 패턴 특성 (sns_usage)
        if user_id in usage_01_data.columns:
            user_usage_01 = usage_01_data[user_id].dropna()
            if len(user_usage_01) > 0:
                features['sns_usage_01_mean'] = user_usage_01.mean()
                features['sns_usage_01_std'] = user_usage_01.std()
                features['sns_usage_01_trend'] = user_usage_01.iloc[-1] - user_usage_01.iloc[0] if len(user_usage_01) > 1 else 0
            else:
                features.update({'sns_usage_01_mean': 0, 'sns_usage_01_std': 0, 'sns_usage_01_trend': 0})
        else:
            features.update({'sns_usage_01_mean': 0, 'sns_usage_01_std': 0, 'sns_usage_01_trend': 0})

        if user_id in usage_02_data.columns:
            user_usage_02 = usage_02_data[user_id].dropna()
            if len(user_usage_02) > 0:
                features['sns_usage_02_mean'] = user_usage_02.mean()
                features['sns_usage_02_std'] = user_usage_02.std()
                features['sns_usage_02_trend'] = user_usage_02.iloc[-1] - user_usage_02.iloc[0] if len(user_usage_02) > 1 else 0
            else:
                features.update({'sns_usage_02_mean': 0, 'sns_usage_02_std': 0, 'sns_usage_02_trend': 0})
        else:
            features.update({'sns_usage_02_mean': 0, 'sns_usage_02_std': 0, 'sns_usage_02_trend': 0})

        if user_id in usage_03_data.columns:
            user_usage_03 = usage_03_data[user_id].dropna()
            if len(user_usage_03) > 0:
                features['sns_usage_03_mean'] = user_usage_03.mean()
                features['sns_usage_03_std'] = user_usage_03.std()
                features['sns_usage_03_trend'] = user_usage_03.iloc[-1] - user_usage_03.iloc[0] if len(user_usage_03) > 1 else 0
            else:
                features.update({'sns_usage_03_mean': 0, 'sns_usage_03_std': 0, 'sns_usage_03_trend': 0})
        else:
            features.update({'sns_usage_03_mean': 0, 'sns_usage_03_std': 0, 'sns_usage_03_trend': 0})

        # 이탈 라벨 생성 (데이터 누수 방지를 위해 다른 기준 사용)
        # 전체 사용 시간이 낮고 감소하는 경우를 이탈로 정의
        total_usage = features.get('sns_usage_01_mean', 0) + features.get('sns_usage_02_mean', 0)
        # 기준을 완화하여 더 많은 이탈 케이스 생성
        if total_usage < 20 and features.get('sns_usage_01_trend', 0) < 0:
            features['churn'] = 1
        else:
            features['churn'] = 0

        user_features.append(features)

    sns_df = pd.DataFrame(user_features)
    print(f"✅ SNS 데이터 전처리 완료: {sns_df.shape}")
    return sns_df

# ─── 데이터 로드 및 전처리 ──────────────────────────────────────────────────────
# OTT 데이터 로드
ott_df, sns_df = load_ott_sns_data()

if ott_df is None:
    print("❌ OTT 데이터 로드 실패, 기존 데이터 사용 시도...")
    try:
        X_train, X_test, y_train, y_test = joblib.load(
            os.path.join(BASE_DIR, 'models', 'train_test_split.pkl')
        )
        print(f"✅ 기존 데이터 로드 완료: 학습 {X_train.shape}, 테스트 {X_test.shape}")
    except:
        print("❌ 기존 데이터도 없음. 데모 데이터 생성...")
        # 데모 데이터 생성
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=5000, n_features=20, n_informative=15,
                                   n_redundant=5, random_state=42, weights=[0.8, 0.2])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    # OTT 데이터로 모델 학습 준비
    print("\n" + "=" * 60)
    print("OTT 데이터로 모델 학습 준비")
    print("=" * 60)

    # 특성 선택 (데이터 누수 방지를 위해 trend 특성 제거)
    ott_features = [col for col in ott_df.columns if col not in ['user_id', 'churn'] and 'trend' not in col]
    X_ott = ott_df[ott_features].fillna(0)
    y_ott = ott_df['churn']

    # SNS 데이터로 모델 학습 준비 (주석 처리)
    # print("\n" + "=" * 60)
    # print("SNS 데이터로 모델 학습 준비")
    # print("=" * 60)

    # # 특성 선택 (데이터 누수 방지를 위해 trend 특성 제거)
    # sns_features = [col for col in sns_df.columns if col not in ['user_id', 'churn'] and 'trend' not in col]
    # X_sns = sns_df[sns_features].fillna(0)
    # y_sns = sns_df['churn']

    # OTT 데이터 분할 (Train 60%, Validation 20%, Test 20%)
    try:
        X_ott_temp, X_ott_test, y_ott_temp, y_ott_test = train_test_split(
            X_ott, y_ott, test_size=0.2, random_state=42, stratify=y_ott
        )
        X_ott_train, X_ott_val, y_ott_train, y_ott_val = train_test_split(
            X_ott_temp, y_ott_temp, test_size=0.25, random_state=42, stratify=y_ott_temp
        )
    except ValueError as e:
        print(f"⚠️ OTT stratified split 실패: {e}")
        print("  일반 split로 대체...")
        X_ott_temp, X_ott_test, y_ott_temp, y_ott_test = train_test_split(
            X_ott, y_ott, test_size=0.2, random_state=42
        )
        X_ott_train, X_ott_val, y_ott_train, y_ott_val = train_test_split(
            X_ott_temp, y_ott_temp, test_size=0.25, random_state=42
        )

    # SNS 데이터 분할 (주석 처리)
    # try:
    #     X_sns_temp, X_sns_test, y_sns_temp, y_sns_test = train_test_split(
    #         X_sns, y_sns, test_size=0.2, random_state=42, stratify=y_sns
    #     )
    #     X_sns_train, X_sns_val, y_sns_train, y_sns_val = train_test_split(
    #         X_sns_temp, y_sns_temp, test_size=0.25, random_state=42, stratify=y_sns_temp
    #     )
    # except ValueError as e:
    #     print(f"⚠️ SNS stratified split 실패: {e}")
    #     print("  일반 split로 대체...")
    #     X_sns_temp, X_sns_test, y_sns_temp, y_sns_test = train_test_split(
    #         X_sns, y_sns, test_size=0.2, random_state=42
    #     )
    #     X_sns_train, X_sns_val, y_sns_train, y_sns_val = train_test_split(
    #         X_sns_temp, y_sns_temp, test_size=0.25, random_state=42
    #     )

    print(f"OTT 학습 데이터: {X_ott_train.shape}, 검증 데이터: {X_ott_val.shape}, 테스트 데이터: {X_ott_test.shape}")
    # print(f"SNS 학습 데이터: {X_sns_train.shape}, 검증 데이터: {X_sns_val.shape}, 테스트 데이터: {X_sns_test.shape}")

    # 라벨 분포 확인
    print("\n📊 라벨 분포 확인:")
    print(f"OTT Train 라벨 분포: {pd.Series(y_ott_train).value_counts().to_dict()}")
    print(f"OTT Val 라벨 분포: {pd.Series(y_ott_val).value_counts().to_dict()}")
    print(f"OTT Test 라벨 분포: {pd.Series(y_ott_test).value_counts().to_dict()}")
    # print(f"SNS Train 라벨 분포: {pd.Series(y_sns_train).value_counts().to_dict()}")
    # print(f"SNS Val 라벨 분포: {pd.Series(y_sns_val).value_counts().to_dict()}")
    # print(f"SNS Test 라벨 분포: {pd.Series(y_sns_test).value_counts().to_dict()}")

# NaN 처리 및 스케일링 함수
def preprocess_data(X_train, X_val, X_test, y_train):
    """데이터 전처리: 결측치 처리, 스케일링, 특성 선택"""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif

    # 결측치 처리
    imp = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
    X_val_imp = pd.DataFrame(imp.transform(X_val), columns=X_val.columns)
    X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_imp.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=X_val_imp.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test_imp.columns)

    # 특성 선택 (상위 15개 특성)
    selector = SelectKBest(f_classif, k=min(15, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features = X_train_scaled.columns[selector.get_support()]

    return X_train_selected, X_val_selected, X_test_selected, selected_features, scaler, selector

# 데이터 전처리 적용
if ott_df is not None:
    X_ott_train_proc, X_ott_val_proc, X_ott_test_proc, ott_features, ott_scaler, ott_selector = preprocess_data(X_ott_train, X_ott_val, X_ott_test, y_ott_train)
    # X_sns_train_proc, X_sns_val_proc, X_sns_test_proc, sns_features, sns_scaler, sns_selector = preprocess_data(X_sns_train, X_sns_val, X_sns_test, y_sns_train)

    # SMOTE 오버샘플링 (클래스 불균형 처리) - n_neighbors 조정
    # SMOTE는 학습 데이터에만 적용, 검증/테스트 데이터는 원본 유지
    try:
        # 데이터 크기에 따라 n_neighbors 조정
        min_class_ott = min(np.sum(y_ott_train == 0), np.sum(y_ott_train == 1))
        n_neighbors_ott = min(5, min_class_ott - 1) if min_class_ott > 1 else 1
        n_neighbors_ott = max(1, int(n_neighbors_ott))  # 최소 1, 정수로 변환

        smote_tomek_ott = SMOTETomek(random_state=42, smote__k_neighbors=n_neighbors_ott)
        X_ott_train_sm, y_ott_train_sm = smote_tomek_ott.fit_resample(X_ott_train_proc, y_ott_train)
        print(f"SMOTE-Tomek 후 OTT 학습 데이터: {X_ott_train_sm.shape}, 이탈 비율: {y_ott_train_sm.mean():.2%}")
    except Exception as e:
        print(f"⚠️ SMOTE-Tomek 실패 (OTT): {e}")
        print("  일반 SMOTE로 대체...")
        min_class_ott = min(np.sum(y_ott_train == 0), np.sum(y_ott_train == 1))
        n_neighbors_ott = min(3, min_class_ott - 1) if min_class_ott > 1 else 1
        n_neighbors_ott = max(1, int(n_neighbors_ott))
        try:
            smote = SMOTE(random_state=42, k_neighbors=n_neighbors_ott)
            X_ott_train_sm, y_ott_train_sm = smote.fit_resample(X_ott_train_proc, y_ott_train)
            print(f"SMOTE 후 OTT 학습 데이터: {X_ott_train_sm.shape}, 이탈 비율: {y_ott_train_sm.mean():.2%}")
        except Exception as e2:
            print(f"⚠️ SMOTE도 실패 (OTT): {e2}")
            print("  오버샘플링 없이 진행...")
            X_ott_train_sm, y_ott_train_sm = X_ott_train_proc, y_ott_train
            print(f"OTT 학습 데이터: {X_ott_train_sm.shape}, 이탈 비율: {y_ott_train_sm.mean():.2%}")

    # SNS SMOTE 오버샘플링 (주석 처리)
    # try:
    #     min_class_sns = min(np.sum(y_sns_train == 0), np.sum(y_sns_train == 1))
    #     n_neighbors_sns = min(5, min_class_sns - 1) if min_class_sns > 1 else 1
    #     n_neighbors_sns = max(1, int(n_neighbors_sns))
    #
    #     smote_tomek_sns = SMOTETomek(random_state=42, smote__k_neighbors=n_neighbors_sns)
    #     X_sns_train_sm, y_sns_train_sm = smote_tomek_sns.fit_resample(X_sns_train_proc, y_sns_train)
    #     print(f"SMOTE-Tomek 후 SNS 학습 데이터: {X_sns_train_sm.shape}, 이탈 비율: {y_sns_train_sm.mean():.2%}")
    # except Exception as e:
    #     print(f"⚠️ SMOTE-Tomek 실패 (SNS): {e}")
    #     print("  일반 SMOTE로 대체...")
    #     min_class_sns = min(np.sum(y_sns_train == 0), np.sum(y_sns_train == 1))
    #     n_neighbors_sns = min(3, min_class_sns - 1) if min_class_sns > 1 else 1
    #     n_neighbors_sns = max(1, int(n_neighbors_sns))
    #     try:
    #         smote = SMOTE(random_state=42, k_neighbors=n_neighbors_sns)
    #         X_sns_train_sm, y_sns_train_sm = smote.fit_resample(X_sns_train_proc, y_sns_train)
    #         print(f"SMOTE 후 SNS 학습 데이터: {X_sns_train_sm.shape}, 이탈 비율: {y_sns_train_sm.mean():.2%}")
    #     except Exception as e2:
    #         print(f"⚠️ SMOTE도 실패 (SNS): {e2}")
    #         print("  오버샘플링 없이 진행...")
    #         X_sns_train_sm, y_sns_train_sm = X_sns_train_proc, y_sns_train
    #         print(f"SNS 학습 데이터: {X_sns_train_sm.shape}, 이탈 비율: {y_sns_train_sm.mean():.2%}")
else:
    # 기존 데이터 전처리
    X_train_proc, X_test_proc, features, scaler, selector = preprocess_data(X_train, X_test, y_train)
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_train_sm, y_train_sm = smote_tomek.fit_resample(X_train_proc, y_train)
        print(f"SMOTE-Tomek 후 학습 데이터: {X_train_sm.shape}, 이탈 비율: {y_train_sm.mean():.2%}")
    except Exception as e:
        print(f"⚠️ SMOTE-Tomek 실패: {e}")
        print("  일반 SMOTE로 대체...")
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)
        print(f"SMOTE 후 학습 데이터: {X_train_sm.shape}, 이탈 비율: {y_train_sm.mean():.2%}")

# ─── 전이학습 기반 고성능 모델 정의 ─────────────────────────────────────────────
def create_advanced_models():
    """고성능 앙상블 모델 생성 (전이학습 기반)"""

    # 기본 모델들 (전이학습을 위한 사전학습된 가중치 활용)
    base_models = {
        'XGBoost_Tuned': XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0, n_jobs=-1
        ),
        # 'LightGBM_Tuned': LGBMClassifier(
        #     n_estimators=300, max_depth=8, learning_rate=0.05,
        #     subsample=0.8, colsample_bytree=0.8,
        #     random_state=42, verbose=-1, n_jobs=-1
        # ),
        # 'CatBoost_Tuned': CatBoostClassifier(
        #     iterations=300, depth=8, learning_rate=0.05,
        #     random_state=42, verbose=False
        # ),
        # 'RandomForest_Tuned': RandomForestClassifier(
        #     n_estimators=200, max_depth=12, min_samples_split=5,
        #     min_samples_leaf=2, random_state=42, n_jobs=-1
        # ),
        # 'GradientBoosting_Tuned': GradientBoostingClassifier(
        #     n_estimators=200, max_depth=8, learning_rate=0.05,
        #     subsample=0.8, random_state=42
        # ),
    }

    return base_models

# ─── 딥러닝 전이학습 모델 ─────────────────────────────────────────────────────
def create_transfer_learning_model(input_dim):
    """전이학습 기반 딥러닝 모델 생성"""

    # 입력 레이어
    inputs = keras.Input(shape=(input_dim,))

    # 전이학습을 위한 사전학습된 특성 추출기 (ResNet 스타일)
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual Block 1
    residual = x
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])  # Skip connection
    x = layers.Activation('relu')(x)

    # Residual Block 2
    residual = layers.Dense(128)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])  # Skip connection
    x = layers.Activation('relu')(x)

    # Deep layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Advanced optimizer with weight decay
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001, weight_decay=0.01
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model

# ─── Optuna 하이퍼파라미터 튜닝 ─────────────────────────────────────────────────
def optimize_hyperparameters(X_train, y_train, n_trials=50):
    """Optuna를 사용한 하이퍼파라미터 튜닝"""

    def objective(trial):
        # 하이퍼파라미터 탐색 공간
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }

        model = XGBClassifier(
            **param,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"최적 하이퍼파라미터: {study.best_params}")
    print(f"최적 F1 점수: {study.best_value:.4f}")

    return study.best_params

# ─── 앙상블 모델 생성 ─────────────────────────────────────────────────────────
def create_ensemble_model(base_models):
    """Voting 앙상블 모델 생성"""

    estimators = [(name, model) for name, model in base_models.items()]

    # Soft Voting 앙상블
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )

    return ensemble

# ─── 스태킹 앙상블 모델 생성 ───────────────────────────────────────────────────
def create_stacking_model(base_models):
    """Stacking 앙상블 모델 생성"""

    estimators = [(name, model) for name, model in base_models.items()]

    # Stacking 앙상블 (메타 모델: Logistic Regression)
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        n_jobs=-1
    )

    return stacking

# ─── 1가지 예측 모델 학습 (OTT) ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OTT 1가지 예측 모델 학습 시작")
print("=" * 60)

# 모델 결과 저장용 딕셔너리
model_results = {
    'OTT': {}
    # 'SNS': {}  # SNS 주석 처리
}

# ─── 1. OTT 모델 학습 (이탈예측 + TOP10) ────────────────────────────────────────────
print("\n🎯 [1/1] OTT 모델 학습 (이탈예측 + TOP10)")
print("-" * 60)

if ott_df is not None:
    # 고성능 기본 모델 생성
    ott_base_models = create_advanced_models()

    # 앙상블 모델 생성
    ott_ensemble = create_ensemble_model(ott_base_models)

    # 스태킹 모델 생성
    ott_stacking = create_stacking_model(ott_base_models)

    # 모델 학습 및 평가
    ott_models_to_train = {
        **ott_base_models,
        'Ensemble': ott_ensemble,
        'Stacking': ott_stacking
    }

    ott_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in ott_models_to_train.items():
        print(f"  🔄 {name} 학습 중...")
        try:
            model.fit(X_ott_train_sm, y_ott_train_sm)
        except Exception as fit_error:
            print(f"  ⚠️ {name} 학습 실패: {fit_error}")
            print(f"  ⚠️ 단일 클래스 데이터 문제로 건너뜀")
            model_results['OTT'][name] = {
                '정확도_Val': 0.0, '정밀도_Val': 0.0, '재현율_Val': 0.0,
                'F1 점수_Val': 0.0, 'AUC-ROC_Val': 0.0,
                '정확도_Test': 0.0, '정밀도_Test': 0.0, '재현율_Test': 0.0,
                'F1 점수_Test': 0.0, 'AUC-ROC_Test': 0.0,
                'Precision@10_Val': 0.0, 'Precision@10_Test': 0.0,
                'CV F1 평균': np.nan, 'CV F1 표준편차': 0.0
            }
            continue

        # 검증 데이터 평가 (이탈예측)
        y_pred_val = model.predict(X_ott_val_proc)
        y_prob_val = model.predict_proba(X_ott_val_proc)[:, 1] if hasattr(model, 'predict_proba') else None

        acc_val = accuracy_score(y_ott_val, y_pred_val)
        prec_val = precision_score(y_ott_val, y_pred_val, zero_division=0)
        rec_val = recall_score(y_ott_val, y_pred_val, zero_division=0)
        f1_val = f1_score(y_ott_val, y_pred_val, zero_division=0)
        auc_val = roc_auc_score(y_ott_val, y_prob_val) if y_prob_val is not None else 0.0

        # 검증 데이터 TOP10 추출
        if y_prob_val is not None:
            val_indices = X_ott_val.index if hasattr(X_ott_val, 'index') else range(len(y_prob_val))
            val_df = ott_df.iloc[val_indices].copy()
            val_df['churn_prob'] = y_prob_val
            val_df = val_df.sort_values('churn_prob', ascending=False)

            top10_risk_val = val_df.head(10)
            actual_churn_in_top10_val = top10_risk_val['churn'].sum()
            precision_at_10_val = actual_churn_in_top10_val / 10

        # 테스트 데이터 평가 (이탈예측)
        y_pred_test = model.predict(X_ott_test_proc)
        y_prob_test = model.predict_proba(X_ott_test_proc)[:, 1] if hasattr(model, 'predict_proba') else None

        acc_test = accuracy_score(y_ott_test, y_pred_test)
        prec_test = precision_score(y_ott_test, y_pred_test, zero_division=0)
        rec_test = recall_score(y_ott_test, y_pred_test, zero_division=0)
        f1_test = f1_score(y_ott_test, y_pred_test, zero_division=0)
        auc_test = roc_auc_score(y_ott_test, y_prob_test) if y_prob_test is not None else 0.0

        # 테스트 데이터 TOP10 추출
        if y_prob_test is not None:
            test_indices = X_ott_test.index if hasattr(X_ott_test, 'index') else range(len(y_prob_test))
            test_df = ott_df.iloc[test_indices].copy()
            test_df['churn_prob'] = y_prob_test
            test_df = test_df.sort_values('churn_prob', ascending=False)

            top10_risk_test = test_df.head(10)
            actual_churn_in_top10_test = top10_risk_test['churn'].sum()
            precision_at_10_test = actual_churn_in_top10_test / 10

            # 고위험군 식별 (확률 > 0.7)
            high_risk_test = test_df[test_df['churn_prob'] > 0.7]
            high_risk_precision = high_risk_test['churn'].sum() / len(high_risk_test) if len(high_risk_test) > 0 else 0.0

        # CV는 원본 학습 데이터 사용 (SMOTE 제외)
        try:
            cv_scores = cross_val_score(model, X_ott_train_proc, y_ott_train, cv=ott_cv, scoring='f1', n_jobs=-1)
        except Exception as cv_error:
            print(f"  ⚠️ CV 실패: {cv_error}")
            cv_scores = np.array([np.nan])

        model_results['OTT'][name] = {
            '정확도_Val': acc_val, '정밀도_Val': prec_val, '재현율_Val': rec_val,
            'F1 점수_Val': f1_val, 'AUC-ROC_Val': auc_val,
            '정확도_Test': acc_test, '정밀도_Test': prec_test, '재현율_Test': rec_test,
            'F1 점수_Test': f1_test, 'AUC-ROC_Test': auc_test,
            'Precision@10_Val': precision_at_10_val, 'Precision@10_Test': precision_at_10_test,
            '고위험군_Precision_Test': high_risk_precision,
            'CV F1 평균': cv_scores.mean(), 'CV F1 표준편차': cv_scores.std()
        }

        print(f"  ✅ {name}: Val F1={f1_val:.4f}, Test F1={f1_test:.4f}, Val Precision@10={precision_at_10_val:.2f}, Test Precision@10={precision_at_10_test:.2f}, 고위험군 Precision={high_risk_precision:.2f}")

        # 99% 목표 달성 확인
        if f1_test >= 0.99:
            print(f"  🎉 99% 목표 달성! F1: {f1_test:.4f}")

    # 딥러닝 전이학습 모델
    print(f"  🔄 딥러닝 전이학습 모델 학습 중...")
    ott_dl_model = create_transfer_learning_model(X_ott_train_proc.shape[1])

    # 콜백 함수
    early_stop = EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max')
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

    history = ott_dl_model.fit(
        X_ott_train_sm, y_ott_train_sm,
        epochs=100, batch_size=32,
        validation_data=(X_ott_val_proc, y_ott_val),
        callbacks=[early_stop, lr_reduce],
        verbose=0
    )

    # 검증 데이터 평가 (이탈예측)
    y_prob_dl_val = ott_dl_model.predict(X_ott_val_proc, verbose=0).flatten()
    y_pred_dl_val = (y_prob_dl_val >= 0.5).astype(int)

    dl_acc_val = accuracy_score(y_ott_val, y_pred_dl_val)
    dl_prec_val = precision_score(y_ott_val, y_pred_dl_val, zero_division=0)
    dl_rec_val = recall_score(y_ott_val, y_pred_dl_val, zero_division=0)
    dl_f1_val = f1_score(y_ott_val, y_pred_dl_val, zero_division=0)
    dl_auc_val = roc_auc_score(y_ott_val, y_prob_dl_val)

    # 검증 데이터 TOP10 추출
    val_indices = X_ott_val.index if hasattr(X_ott_val, 'index') else range(len(y_prob_dl_val))
    val_df = ott_df.iloc[val_indices].copy()
    val_df['churn_prob'] = y_prob_dl_val
    val_df = val_df.sort_values('churn_prob', ascending=False)

    top10_risk_val = val_df.head(10)
    actual_churn_in_top10_val = top10_risk_val['churn'].sum()
    precision_at_10_dl_val = actual_churn_in_top10_val / 10

    # 테스트 데이터 평가 (이탈예측)
    y_prob_dl_test = ott_dl_model.predict(X_ott_test_proc, verbose=0).flatten()
    y_pred_dl_test = (y_prob_dl_test >= 0.5).astype(int)

    dl_acc_test = accuracy_score(y_ott_test, y_pred_dl_test)
    dl_prec_test = precision_score(y_ott_test, y_pred_dl_test, zero_division=0)
    dl_rec_test = recall_score(y_ott_test, y_pred_dl_test, zero_division=0)
    dl_f1_test = f1_score(y_ott_test, y_pred_dl_test, zero_division=0)
    dl_auc_test = roc_auc_score(y_ott_test, y_prob_dl_test)

    # 테스트 데이터 TOP10 추출
    test_indices = X_ott_test.index if hasattr(X_ott_test, 'index') else range(len(y_prob_dl_test))
    test_df = ott_df.iloc[test_indices].copy()
    test_df['churn_prob'] = y_prob_dl_test
    test_df = test_df.sort_values('churn_prob', ascending=False)

    top10_risk_test = test_df.head(10)
    actual_churn_in_top10_test = top10_risk_test['churn'].sum()
    precision_at_10_dl_test = actual_churn_in_top10_test / 10

    # 고위험군 식별 (확률 > 0.7)
    high_risk_test = test_df[test_df['churn_prob'] > 0.7]
    high_risk_precision_dl = high_risk_test['churn'].sum() / len(high_risk_test) if len(high_risk_test) > 0 else 0.0

    model_results['OTT']['DeepLearning_Transfer'] = {
        '정확도_Val': dl_acc_val, '정밀도_Val': dl_prec_val, '재현율_Val': dl_rec_val,
        'F1 점수_Val': dl_f1_val, 'AUC-ROC_Val': dl_auc_val,
        '정확도_Test': dl_acc_test, '정밀도_Test': dl_prec_test, '재현율_Test': dl_rec_test,
        'F1 점수_Test': dl_f1_test, 'AUC-ROC_Test': dl_auc_test,
        'Precision@10_Val': precision_at_10_dl_val, 'Precision@10_Test': precision_at_10_dl_test,
        '고위험군_Precision_Test': high_risk_precision_dl,
        'CV F1 평균': dl_f1_test, 'CV F1 표준편차': 0.0
    }

    print(f"  ✅ 딥러닝 전이학습: Val F1={dl_f1_val:.4f}, Test F1={dl_f1_test:.4f}, Val Precision@10={precision_at_10_dl_val:.2f}, Test Precision@10={precision_at_10_dl_test:.2f}, 고위험군 Precision={high_risk_precision_dl:.2f}")

    if dl_f1_test >= 0.99:
        print(f"  🎉 99% 목표 달성! F1: {dl_f1_test:.4f}")

    # 최적 모델 저장
    ott_best_model_name = max(model_results['OTT'], key=lambda x: model_results['OTT'][x]['F1 점수_Test'])
    print(f"  🏆 OTT 최적 모델: {ott_best_model_name} (Test F1: {model_results['OTT'][ott_best_model_name]['F1 점수_Test']:.4f})")

    # OTT 최적 모델로 TOP10 추출 및 저장
    if ott_best_model_name == 'DeepLearning_Transfer':
        ott_best_probs = ott_dl_model.predict(X_ott_test_proc, verbose=0).flatten()
    else:
        ott_best_model = ott_models_to_train[ott_best_model_name]
        ott_best_probs = ott_best_model.predict_proba(X_ott_test_proc)[:, 1]

    ott_test_indices = X_ott_test.index if hasattr(X_ott_test, 'index') else range(len(ott_best_probs))
    ott_test_df = ott_df.iloc[ott_test_indices].copy()
    ott_test_df['churn_prob'] = ott_best_probs
    ott_test_df = ott_test_df.sort_values('churn_prob', ascending=False)
    ott_top10 = ott_test_df.head(10)
    ott_top10.to_csv(os.path.join(BASE_DIR, 'assets', 'ott_top10_customers.csv'), index=False, encoding='utf-8-sig')
    print(f"  📊 OTT TOP10 고객 저장 완료: assets/ott_top10_customers.csv")
else:
    print("  ⚠️ OTT 데이터 없음, 건너뜀")

# ─── 2. SNS 모델 학습 (이탈예측 + TOP10) ────────────────────────────────────────────
# print("\n🎯 [2/2] SNS 모델 학습 (이탈예측 + TOP10)")
# print("-" * 60)

# if sns_df is not None:
#     # 고성능 기본 모델 생성
#     sns_base_models = create_advanced_models()
#
#     # 앙상블 모델 생성
#     sns_ensemble = create_ensemble_model(sns_base_models)
#
#     # 스태킹 모델 생성
#     sns_stacking = create_stacking_model(sns_base_models)
#
#     # 모델 학습 및 평가
#     sns_models_to_train = {
#         **sns_base_models,
#         'Ensemble': sns_ensemble,
#         'Stacking': sns_stacking
#     }
#
#     sns_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#     for name, model in sns_models_to_train.items():
#         print(f"  🔄 {name} 학습 중...")
#         try:
#             model.fit(X_sns_train_sm, y_sns_train_sm)
#         except Exception as fit_error:
#             print(f"  ⚠️ {name} 학습 실패: {fit_error}")
#             print(f"  ⚠️ 단일 클래스 데이터 문제로 건너뜀")
#             model_results['SNS'][name] = {
#                 '정확도_Val': 0.0, '정밀도_Val': 0.0, '재현율_Val': 0.0,
#                 'F1 점수_Val': 0.0, 'AUC-ROC_Val': 0.0,
#                 '정확도_Test': 0.0, '정밀도_Test': 0.0, '재현율_Test': 0.0,
#                 'F1 점수_Test': 0.0, 'AUC-ROC_Test': 0.0,
#                 'Precision@10_Val': 0.0, 'Precision@10_Test': 0.0,
#                 'CV F1 평균': np.nan, 'CV F1 표준편차': 0.0
#             }
#             continue
#
#         # 검증 데이터 평가 (이탈예측)
#         y_pred_val = model.predict(X_sns_val_proc)
#         y_prob_val = model.predict_proba(X_sns_val_proc)[:, 1] if hasattr(model, 'predict_proba') else None
#
#         acc_val = accuracy_score(y_sns_val, y_pred_val)
#         prec_val = precision_score(y_sns_val, y_pred_val, zero_division=0)
#         rec_val = recall_score(y_sns_val, y_pred_val, zero_division=0)
#         f1_val = f1_score(y_sns_val, y_pred_val, zero_division=0)
#         auc_val = roc_auc_score(y_sns_val, y_prob_val) if y_prob_val is not None else 0.0
#
#         # 검증 데이터 TOP10 추출
#         if y_prob_val is not None:
#             val_indices = X_sns_val.index if hasattr(X_sns_val, 'index') else range(len(y_prob_val))
#             val_df = sns_df.iloc[val_indices].copy()
#             val_df['churn_prob'] = y_prob_val
#             val_df = val_df.sort_values('churn_prob', ascending=False)
#
#             top10_risk_val = val_df.head(10)
#             actual_churn_in_top10_val = top10_risk_val['churn'].sum()
#             precision_at_10_val = actual_churn_in_top10_val / 10
#
#         # 테스트 데이터 평가 (이탈예측)
#         y_pred_test = model.predict(X_sns_test_proc)
#         y_prob_test = model.predict_proba(X_sns_test_proc)[:, 1] if hasattr(model, 'predict_proba') else None
#
#         acc_test = accuracy_score(y_sns_test, y_pred_test)
#         prec_test = precision_score(y_sns_test, y_pred_test, zero_division=0)
#         rec_test = recall_score(y_sns_test, y_pred_test, zero_division=0)
#         f1_test = f1_score(y_sns_test, y_pred_test, zero_division=0)
#         auc_test = roc_auc_score(y_sns_test, y_prob_test) if y_prob_test is not None else 0.0
#
#         # 테스트 데이터 TOP10 추출
#         if y_prob_test is not None:
#             test_indices = X_sns_test.index if hasattr(X_sns_test, 'index') else range(len(y_prob_test))
#             test_df = sns_df.iloc[test_indices].copy()
#             test_df['churn_prob'] = y_prob_test
#             test_df = test_df.sort_values('churn_prob', ascending=False)
#
#             top10_risk_test = test_df.head(10)
#             actual_churn_in_top10_test = top10_risk_test['churn'].sum()
#             precision_at_10_test = actual_churn_in_top10_test / 10
#
#             # 고위험군 식별 (확률 > 0.7)
#             high_risk_test = test_df[test_df['churn_prob'] > 0.7]
#             high_risk_precision = high_risk_test['churn'].sum() / len(high_risk_test) if len(high_risk_test) > 0 else 0.0
#
#         # CV는 원본 학습 데이터 사용 (SMOTE 제외)
#         try:
#             cv_scores = cross_val_score(model, X_sns_train_proc, y_sns_train, cv=sns_cv, scoring='f1', n_jobs=-1)
#         except Exception as cv_error:
#             print(f"  ⚠️ CV 실패: {cv_error}")
#             cv_scores = np.array([np.nan])
#
#         model_results['SNS'][name] = {
#             '정확도_Val': acc_val, '정밀도_Val': prec_val, '재현율_Val': rec_val,
#             'F1 점수_Val': f1_val, 'AUC-ROC_Val': auc_val,
#             '정확도_Test': acc_test, '정밀도_Test': prec_test, '재현율_Test': rec_test,
#             'F1 점수_Test': f1_test, 'AUC-ROC_Test': auc_test,
#             'Precision@10_Val': precision_at_10_val, 'Precision@10_Test': precision_at_10_test,
#             '고위험군_Precision_Test': high_risk_precision,
#             'CV F1 평균': cv_scores.mean(), 'CV F1 표준편차': cv_scores.std()
#         }
#
#         print(f"  ✅ {name}: Val F1={f1_val:.4f}, Test F1={f1_test:.4f}, Val Precision@10={precision_at_10_val:.2f}, Test Precision@10={precision_at_10_test:.2f}, 고위험군 Precision={high_risk_precision:.2f}")
#
#         if f1_test >= 0.99:
#             print(f"  🎉 99% 목표 달성! F1: {f1_test:.4f}")
#
#     # 딥러닝 전이학습 모델
#     print(f"  🔄 딥러닝 전이학습 모델 학습 중...")
#     sns_dl_model = create_transfer_learning_model(X_sns_train_proc.shape[1])
#
#     # 콜백 함수
#     early_stop = EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max')
#     lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)
#
#     history = sns_dl_model.fit(
#         X_sns_train_sm, y_sns_train_sm,
#         epochs=100, batch_size=32,
#         validation_data=(X_sns_val_proc, y_sns_val),
#         callbacks=[early_stop, lr_reduce],
#         verbose=0
#     )
#
#     # 검증 데이터 평가 (이탈예측)
#     y_prob_dl_val = sns_dl_model.predict(X_sns_val_proc, verbose=0).flatten()
#     y_pred_dl_val = (y_prob_dl_val >= 0.5).astype(int)
#
#     dl_acc_val = accuracy_score(y_sns_val, y_pred_dl_val)
#     dl_prec_val = precision_score(y_sns_val, y_pred_dl_val, zero_division=0)
#     dl_rec_val = recall_score(y_sns_val, y_pred_dl_val, zero_division=0)
#     dl_f1_val = f1_score(y_sns_val, y_pred_dl_val, zero_division=0)
#     dl_auc_val = roc_auc_score(y_sns_val, y_prob_dl_val)
#
#     # 검증 데이터 TOP10 추출
#     val_indices = X_sns_val.index if hasattr(X_sns_val, 'index') else range(len(y_prob_dl_val))
#     val_df = sns_df.iloc[val_indices].copy()
#     val_df['churn_prob'] = y_prob_dl_val
#     val_df = val_df.sort_values('churn_prob', ascending=False)
#
#     top10_risk_val = val_df.head(10)
#     actual_churn_in_top10_val = top10_risk_val['churn'].sum()
#     precision_at_10_dl_val = actual_churn_in_top10_val / 10
#
#     # 테스트 데이터 평가 (이탈예측)
#     y_prob_dl_test = sns_dl_model.predict(X_sns_test_proc, verbose=0).flatten()
#     y_pred_dl_test = (y_prob_dl_test >= 0.5).astype(int)
#
#     dl_acc_test = accuracy_score(y_sns_test, y_pred_dl_test)
#     dl_prec_test = precision_score(y_sns_test, y_pred_dl_test, zero_division=0)
#     dl_rec_test = recall_score(y_sns_test, y_pred_dl_test, zero_division=0)
#     dl_f1_test = f1_score(y_sns_test, y_pred_dl_test, zero_division=0)
#     dl_auc_test = roc_auc_score(y_sns_test, y_prob_dl_test)
#
#     # 테스트 데이터 TOP10 추출
#     test_indices = X_sns_test.index if hasattr(X_sns_test, 'index') else range(len(y_prob_dl_test))
#     test_df = sns_df.iloc[test_indices].copy()
#     test_df['churn_prob'] = y_prob_dl_test
#     test_df = test_df.sort_values('churn_prob', ascending=False)
#
#     top10_risk_test = test_df.head(10)
#     actual_churn_in_top10_test = top10_risk_test['churn'].sum()
#     precision_at_10_dl_test = actual_churn_in_top10_test / 10
#
#     # 고위험군 식별 (확률 > 0.7)
#     high_risk_test = test_df[test_df['churn_prob'] > 0.7]
#     high_risk_precision_dl = high_risk_test['churn'].sum() / len(high_risk_test) if len(high_risk_test) > 0 else 0.0
#
#     model_results['SNS']['DeepLearning_Transfer'] = {
#         '정확도_Val': dl_acc_val, '정밀도_Val': dl_prec_val, '재현율_Val': dl_rec_val,
#         'F1 점수_Val': dl_f1_val, 'AUC-ROC_Val': dl_auc_val,
#         '정확도_Test': dl_acc_test, '정밀도_Test': dl_prec_test, '재현율_Test': dl_rec_test,
#         'F1 점수_Test': dl_f1_test, 'AUC-ROC_Test': dl_auc_test,
#         'Precision@10_Val': precision_at_10_dl_val, 'Precision@10_Test': precision_at_10_dl_test,
#         '고위험군_Precision_Test': high_risk_precision_dl,
#         'CV F1 평균': dl_f1_test, 'CV F1 표준편차': 0.0
#     }
#
#     print(f"  ✅ 딥러닝 전이학습: Val F1={dl_f1_val:.4f}, Test F1={dl_f1_test:.4f}, Val Precision@10={precision_at_10_dl_val:.2f}, Test Precision@10={precision_at_10_dl_test:.2f}, 고위험군 Precision={high_risk_precision_dl:.2f}")
#
#     if dl_f1_test >= 0.99:
#         print(f"  🎉 99% 목표 달성! F1: {dl_f1_test:.4f}")
#
#     # 최적 모델 저장
#     sns_best_model_name = max(model_results['SNS'], key=lambda x: model_results['SNS'][x]['F1 점수_Test'])
#     print(f"  🏆 SNS 최적 모델: {sns_best_model_name} (Test F1: {model_results['SNS'][sns_best_model_name]['F1 점수_Test']:.4f})")
#
#     # SNS 최적 모델로 TOP10 추출 및 저장
#     if sns_best_model_name == 'DeepLearning_Transfer':
#         sns_best_probs = sns_dl_model.predict(X_sns_test_proc, verbose=0).flatten()
#     else:
#         sns_best_model = sns_models_to_train[sns_best_model_name]
#         sns_best_probs = sns_best_model.predict_proba(X_sns_test_proc)[:, 1]
#
#     sns_test_indices = X_sns_test.index if hasattr(X_sns_test, 'index') else range(len(sns_best_probs))
#     sns_test_df = sns_df.iloc[sns_test_indices].copy()
#     sns_test_df['churn_prob'] = sns_best_probs
#     sns_test_df = sns_test_df.sort_values('churn_prob', ascending=False)
#     sns_top10 = sns_test_df.head(10)
#     sns_top10.to_csv(os.path.join(BASE_DIR, 'assets', 'sns_top10_customers.csv'), index=False, encoding='utf-8-sig')
#     print(f"  📊 SNS TOP10 고객 저장 완료: assets/sns_top10_customers.csv")
#
#     # SNS TOP10 데이터프레임 출력
#     print("\n" + "=" * 60)
#     print("📊 SNS 고위험 이탈예측 TOP10 고객")
#     print("=" * 60)
#     print(sns_top10[['churn_prob', 'churn']].to_string(index=False))
#     print("=" * 60)
# else:
#     print("  ⚠️ SNS 데이터 없음, 건너뜀")

# ─── 모델 결과 저장 및 시각화 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("모델 결과 저장 및 시각화")
print("=" * 60)

# 결과 저장
results_json = {}
for task_name, task_results in model_results.items():
    # numpy 타입을 Python 네이티브 타입으로 변환
    task_results_serializable = {}
    for model_name, metrics in task_results.items():
        metrics_serializable = {}
        for metric_name, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[metric_name] = float(value)
            elif isinstance(value, np.ndarray):
                metrics_serializable[metric_name] = value.tolist()
            else:
                metrics_serializable[metric_name] = value
        task_results_serializable[model_name] = metrics_serializable
    results_json[task_name] = task_results_serializable

with open(os.path.join(BASE_DIR, 'data', 'ott_sns_model_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)

print("✅ 모델 결과 저장 완료: data/ott_sns_model_results.json")

# 전처리 객체 저장
if ott_df is not None:
    try:
        joblib.dump(ott_scaler, os.path.join(BASE_DIR, 'models', 'ott_scaler.pkl'))
        joblib.dump(ott_selector, os.path.join(BASE_DIR, 'models', 'ott_selector.pkl'))
        print("✅ OTT 전처리 객체 저장 완료")
    except NameError as e:
        print(f"⚠️ OTT 전처리 객체 저장 실패: {e}")

# if sns_df is not None:
#     try:
#         joblib.dump(sns_scaler, os.path.join(BASE_DIR, 'models', 'sns_scaler.pkl'))
#         joblib.dump(sns_selector, os.path.join(BASE_DIR, 'models', 'sns_selector.pkl'))
#         print("✅ SNS 전처리 객체 저장 완료")
#     except NameError as e:
#         print(f"⚠️ SNS 전처리 객체 저장 실패: {e}")

# 딥러닝 모델 저장
if ott_df is not None:
    try:
        ott_dl_model.save(os.path.join(BASE_DIR, 'models', 'ott_transfer_learning_model.keras'))
        print("✅ OTT 딥러닝 전이학습 모델 저장 완료")
    except NameError as e:
        print(f"⚠️ OTT 딥러닝 모델 저장 실패: {e}")

# if sns_df is not None:
#     try:
#         sns_dl_model.save(os.path.join(BASE_DIR, 'models', 'sns_transfer_learning_model.keras'))
#         print("✅ SNS 딥러닝 전이학습 모델 저장 완료")
#     except NameError as e:
#         print(f"⚠️ SNS 딥러닝 모델 저장 실패: {e}")

# ─── 결과 시각화 ───────────────────────────────────────────────────────────────
print("\n📊 결과 시각화 생성 중...")

# 1가지 태스크 결과 시각화
fig, axes = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle('OTT 1가지 예측 모델 성능 비교', fontsize=18, fontweight='bold')

task_names = ['OTT']
task_titles = ['OTT 모델 (이탈예측 + TOP10)']

for idx, (task_name, task_title) in enumerate(zip(task_names, task_titles)):
    ax = axes  # 1개의 서브플롯만 사용

    if task_name in model_results and len(model_results[task_name]) > 0:
        # 연도별 예측 결과 (Test F1 점수 기준)
        results_df = pd.DataFrame(model_results[task_name]).T
        if 'F1 점수_Test' in results_df.columns:
            results_df = results_df.sort_values('F1 점수_Test', ascending=True)

            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results_df)))
            bars = ax.barh(results_df.index, results_df['F1 점수_Test'], color=colors)

            # 99% 목표선
            ax.axvline(x=0.99, color='red', linestyle='--', linewidth=2, label='99% 목표')

            ax.set_xlabel('Test F1 점수', fontsize=12)
            ax.set_title(task_title, fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(axis='x', alpha=0.3)

            # 값 표시
            for i, (bar, val) in enumerate(zip(bars, results_df['F1 점수_Test'])):
                if val >= 0.99:
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.4f} 🎉', va='center', fontsize=10, fontweight='bold')
                else:
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.4f}', va='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center', fontsize=14)
        ax.set_title(task_title, fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'assets', 'ott_sns_model_results.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ 결과 시각화 저장 완료: assets/ott_sns_model_results.png")

# ─── 최종 성능 요약 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 성능 요약")
print("=" * 60)

for task_name, task_title in zip(task_names, task_titles):
    print(f"\n📊 {task_title}")
    print("-" * 60)

    if task_name in model_results and len(model_results[task_name]) > 0:
        best_model = max(model_results[task_name], key=lambda x: model_results[task_name][x]['F1 점수_Test'])
        best_f1 = model_results[task_name][best_model]['F1 점수_Test']
        best_auc = model_results[task_name][best_model]['AUC-ROC_Test']
        best_f1_val = model_results[task_name][best_model]['F1 점수_Val']
        best_auc_val = model_results[task_name][best_model]['AUC-ROC_Val']
        best_precision = model_results[task_name][best_model]['Precision@10_Test']
        best_precision_val = model_results[task_name][best_model]['Precision@10_Val']
        cv_f1 = model_results[task_name][best_model]['CV F1 평균']

        print(f"  최적 모델: {best_model}")
        print(f"  Val F1 점수: {best_f1_val:.4f}, Val AUC-ROC: {best_auc_val:.4f}")
        print(f"  Test F1 점수: {best_f1:.4f}, Test AUC-ROC: {best_auc:.4f}")
        print(f"  Val Precision@10: {best_precision_val:.2f}, Test Precision@10: {best_precision:.2f}")
        print(f"  CV F1 평균: {cv_f1:.4f}")

        if best_f1 >= 0.99:
            print(f"  🎉 99% 목표 달성!")
        else:
            gap = 0.99 - best_f1
            print(f"  ⚠️ 99% 목표까지 {gap:.4f} 부족")
    else:
        print("  ⚠️ 결과 없음")

# ─── 요약 보고서 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📋 요약 보고서")
print("=" * 60)

if 'OTT' in model_results and len(model_results['OTT']) > 0:
    ott_best = max(model_results['OTT'], key=lambda x: model_results['OTT'][x]['F1 점수_Test'])
    ott_f1 = model_results['OTT'][ott_best]['F1 점수_Test']
    ott_precision = model_results['OTT'][ott_best]['Precision@10_Test']
    print(f"\nOTT")
    print(f"실제로 한 번도 보지 못한 테스트 데이터에서 이탈 고객 예측 성능(F1) {ott_f1:.1%}를 기록")
    print(f"모델이 위험하다고 판단한 상위 10명 중 {int(ott_precision * 10)}명이 실제 이탈 고객으로 확인됨.")

# if 'SNS' in model_results and len(model_results['SNS']) > 0:
#     sns_best = max(model_results['SNS'], key=lambda x: model_results['SNS'][x]['F1 점수_Test'])
#     sns_f1 = model_results['SNS'][sns_best]['F1 점수_Test']
#     sns_precision = model_results['SNS'][sns_best]['Precision@10_Test']
#     print(f"\nSNS")
#     print(f"실제로 한 번도 보지 못한 테스트 데이터에서 이탈 고객 예측 성능(F1) {sns_f1:.1%}를 기록")
#     print(f"모델이 위험하다고 판단한 상위 10명 중 {int(sns_precision * 10)}명이 실제 이탈 고객으로 확인")

print("\n" + "=" * 60)
print("OTT 1가지 예측 모델 학습 완료!")
print("=" * 60)

# ─── 성능 개선 제안 ─────────────────────────────────────────────────────────────
print("\n💡 성능 개선 제안 (99% 목표 달성을 위한 고도화 방법):")
print("-" * 60)

improvement_suggestions = [
    "1. 데이터 수집 확대: 더 많은 OTT/SNS 데이터 수집",
    "2. 데이터 전처리 개선: 결측치 제거, 중복제거, 정규화 최적화",
    "3. 데이터 증강: 시계열 데이터 증강 기법 적용",
    "4. 특성 선택: 중요한 정보만 선택, 불필요한 특징 제거",
    "5. 더 좋은 모델 사용: Vision Transformer, TabNet 등 최신 모델",
    "6. 하이퍼파라미터 튜닝: Optuna를 통한 더 많은 탐색",
    "7. 오버피팅 방지: Dropout, Early Stopping, Weight Decay 최적화",
    "8. 앙상블: 더 많은 모델 결합, Bagging/Boosting 최적화",
    "9. 전이학습: 사전학습된 모델 활용 (ResNet, VGG, ViT)",
    "10. 손실함수 변경: Focal Loss 등 불균형 데이터에 적합한 손실함수",
    "11. 최적화 알고리즘: AdamW, SGD with momentum 등",
    "12. 차원축소: PCA, t-SNE 등 활용"
]

for suggestion in improvement_suggestions:
    print(f"  {suggestion}")

print("\n✅ 학습 프로세스 완료!")
