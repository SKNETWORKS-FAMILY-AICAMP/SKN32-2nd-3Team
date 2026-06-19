"""
머신러닝 / 딥러닝 모델 학습 및 평가
국내 통신사 고객 이탈 예측 프로젝트
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
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 한글 폰트
for f in fm.findSystemFonts():
    if 'Nanum' in f or 'nanum' in f:
        plt.rcParams['font.family'] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = joblib.load(
    '/home/ubuntu/churn_project/churn_project/models/train_test_split.pkl'
)
print(f"학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# NaN 처리 (SimpleImputer)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

# SMOTE 오버샘플링 (클래스 불균형 처리)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"SMOTE 후 학습 데이터: {X_train_sm.shape}, 이탈 비율: {y_train_sm.mean():.2%}")

# ─── 머신러닝 모델 정의 ────────────────────────────────────────────────────────
models = {
    '로지스틱 회귀': LogisticRegression(max_iter=1000, random_state=42),
    '의사결정나무': DecisionTreeClassifier(max_depth=8, random_state=42),
    '랜덤포레스트': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              use_label_encoder=False, eval_metric='logloss',
                              random_state=42, verbosity=0),
    'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                random_state=42, verbose=-1),
    '그래디언트 부스팅': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                 learning_rate=0.1, random_state=42),
}

# ─── 모델 학습 및 평가 ────────────────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print("머신러닝 모델 학습 및 평가")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring='f1', n_jobs=-1)

    results[name] = {
        '정확도': acc, '정밀도': prec, '재현율': rec,
        'F1 점수': f1, 'AUC-ROC': auc,
        'CV F1 평균': cv_scores.mean(), 'CV F1 표준편차': cv_scores.std()
    }
    print(f"\n[{name}]")
    print(f"  정확도: {acc:.4f} | 정밀도: {prec:.4f} | 재현율: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── 딥러닝 모델 (MLP) ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("딥러닝 모델 (MLP) 학습")
print("=" * 60)

input_dim = X_train_sm.shape[1]
dl_model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

dl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

early_stop = keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max')
lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

history = dl_model.fit(
    X_train_sm, y_train_sm,
    epochs=50, batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop, lr_reduce],
    verbose=0
)

y_prob_dl = dl_model.predict(X_test, verbose=0).flatten()
y_pred_dl = (y_prob_dl >= 0.5).astype(int)

dl_acc = accuracy_score(y_test, y_pred_dl)
dl_prec = precision_score(y_test, y_pred_dl, zero_division=0)
dl_rec = recall_score(y_test, y_pred_dl, zero_division=0)
dl_f1 = f1_score(y_test, y_pred_dl, zero_division=0)
dl_auc = roc_auc_score(y_test, y_prob_dl)

results['딥러닝 (MLP)'] = {
    '정확도': dl_acc, '정밀도': dl_prec, '재현율': dl_rec,
    'F1 점수': dl_f1, 'AUC-ROC': dl_auc,
    'CV F1 평균': dl_f1, 'CV F1 표준편차': 0.0
}
print(f"[딥러닝 MLP]")
print(f"  정확도: {dl_acc:.4f} | 정밀도: {dl_prec:.4f} | 재현율: {dl_rec:.4f} | F1: {dl_f1:.4f} | AUC: {dl_auc:.4f}")

# ─── 최적 모델 선정 ───────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).T
best_model_name = results_df['AUC-ROC'].idxmax()
print(f"\n최적 모델: {best_model_name} (AUC-ROC: {results_df.loc[best_model_name, 'AUC-ROC']:.4f})")

# 최적 모델 저장
if best_model_name == '딥러닝 (MLP)':
    dl_model.save('/home/ubuntu/churn_project/churn_project/models/best_model_dl.keras')
    joblib.dump({'type': 'dl', 'name': best_model_name}, '/home/ubuntu/churn_project/churn_project/models/best_model_info.pkl')
else:
    best_model = models[best_model_name]
    joblib.dump(best_model, '/home/ubuntu/churn_project/churn_project/models/best_model.pkl')
    joblib.dump({'type': 'ml', 'name': best_model_name}, '/home/ubuntu/churn_project/churn_project/models/best_model_info.pkl')

# 모든 ML 모델 저장
for name, model in models.items():
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(model, f'/home/ubuntu/churn_project/churn_project/models/model_{safe_name}.pkl')

# 딥러닝 모델 저장
dl_model.save('/home/ubuntu/churn_project/churn_project/models/model_DL_MLP.keras')

# 결과 저장
results_df.to_csv('/home/ubuntu/churn_project/churn_project/data/model_results.csv', encoding='utf-8-sig')
with open('/home/ubuntu/churn_project/churn_project/data/model_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ─── 시각화 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('모델 학습 결과 비교', fontsize=16, fontweight='bold')

# 1) 모델별 성능 비교
ax = axes[0, 0]
metrics = ['정확도', '정밀도', '재현율', 'F1 점수', 'AUC-ROC']
x = np.arange(len(results_df))
width = 0.15
colors_m = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=colors_m[i], alpha=0.85)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df.index, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('점수')
ax.set_title('모델별 성능 비교', fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

# 2) AUC-ROC 곡선
ax = axes[0, 1]
# ML 모델들
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=1.5, label=f'{name} ({auc:.3f})')
# DL 모델
fpr_dl, tpr_dl, _ = roc_curve(y_test, y_prob_dl)
ax.plot(fpr_dl, tpr_dl, linewidth=2.5, linestyle='--', color='black', label=f'딥러닝 MLP ({dl_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC 곡선 비교', fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)

# 3) 최적 모델 혼동행렬
ax = axes[1, 0]
if best_model_name == '딥러닝 (MLP)':
    y_pred_best = y_pred_dl
else:
    y_pred_best = models[best_model_name].predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['유지', '이탈'], yticklabels=['유지', '이탈'],
            annot_kws={'size': 14})
ax.set_title(f'최적 모델 혼동행렬\n({best_model_name})', fontweight='bold')
ax.set_ylabel('실제값')
ax.set_xlabel('예측값')

# 4) 딥러닝 학습 곡선
ax = axes[1, 1]
ax.plot(history.history['loss'], label='학습 손실', color='#2196F3', linewidth=2)
ax.plot(history.history['val_loss'], label='검증 손실', color='#F44336', linewidth=2)
ax2 = ax.twinx()
ax2.plot(history.history['auc'], label='학습 AUC', color='#4CAF50', linewidth=2, linestyle='--')
ax2.plot(history.history['val_auc'], label='검증 AUC', color='#FF9800', linewidth=2, linestyle='--')
ax.set_xlabel('에포크')
ax.set_ylabel('손실')
ax2.set_ylabel('AUC')
ax.set_title('딥러닝 학습 곡선', fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/churn_project/churn_project/assets/model_results.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("\n모델 결과 시각화 저장: assets/model_results.png")

# ─── 특성 중요도 (최적 ML 모델) ───────────────────────────────────────────────
feature_cols = joblib.load('/home/ubuntu/churn_project/churn_project/models/feature_cols.pkl')
best_ml_name = results_df[results_df.index != '딥러닝 (MLP)']['AUC-ROC'].idxmax()
best_ml_model = models[best_ml_name]

if hasattr(best_ml_model, 'feature_importances_'):
    importances = best_ml_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_fi = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feat_imp)))
    bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors_fi[::-1])
    ax.set_title(f'특성 중요도 TOP 15\n({best_ml_name})', fontsize=14, fontweight='bold')
    ax.set_xlabel('중요도')
    for bar, val in zip(bars, feat_imp.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/churn_project/churn_project/assets/feature_importance.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("특성 중요도 저장: assets/feature_importance.png")

print("\n" + "=" * 60)
print("모든 모델 학습 완료!")
print(f"최적 모델: {best_model_name}")
print(results_df[['정확도', 'F1 점수', 'AUC-ROC']].to_string())
print("=" * 60)
