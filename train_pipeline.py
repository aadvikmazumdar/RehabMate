import pandas as pd
import numpy as np
import pickle
import time
import warnings
import json
import os
import csv
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except ImportError:
    HAS_SGK = False
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
import shap

DATA_PATH     = "pushup_data_v2.csv"
PASTED_CSV    = "pasted_data.csv"
OUTPUT_DIR    = "pipeline_outputs"
N_TRIALS      = 100
RANDOM_STATE  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Frame-level features + cumulative rep stats (no future leakage)
FEATURES = [
    # Raw angles
    'elbow', 'elbow_flare', 'hip', 'shoulder_elevation', 'body_line',
    # Dynamics
    'elbow_velocity', 'hip_velocity', 'time_in_phase',
    'elbow_drift', 'hip_drift', 'velocity_trend',
    # Relative (person-normalized)
    'elbow_rel', 'hip_rel',
    # Interactions
    'hip_elbow_ratio', 'body_line_hip_diff',
    # Cumulative rep stats (computed up to current frame, no future leak)
    'elbow_cum_min', 'elbow_cum_max', 'elbow_cum_range',
    'hip_cum_min', 'hip_cum_max', 'hip_cum_range',
]


def load_csv_robust(path):
    header_18 = [
        'elbow', 'elbow_flare', 'hip', 'wrist', 'shoulder_elevation', 'body_line',
        'elbow_velocity', 'hip_velocity', 'time_in_phase', 'rep_number',
        'elbow_drift', 'hip_drift', 'velocity_trend',
        'rolling_elbow_5', 'rolling_hip_5', 'phase', 'form_quality', 'subject_id'
    ]
    old_rows = []
    new_rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 17:
                old_rows.append(row)
            elif len(row) == 18:
                new_rows.append(row)

    if old_rows:
        prev_rep = -1
        sessions = []
        current = []
        for row in old_rows:
            rep = int(float(row[9]))
            if rep == 0 and prev_rep > 0 and current:
                sessions.append(current)
                current = []
            current.append(row)
            prev_rep = rep
        if current:
            sessions.append(current)
        for si, sess in enumerate(sessions):
            for row in sess:
                row.append(str(100 + si))
                new_rows.append(row)

    df = pd.DataFrame(new_rows, columns=header_18)
    for col in header_18:
        if col not in ('phase', 'form_quality', 'subject_id'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['subject_id'] = df['subject_id'].astype(str)
    df['rep_number'] = df['rep_number'].astype(int)
    return df


def add_features(df):
    # Relative features
    df['elbow_rel'] = df['elbow'] - df['rolling_elbow_5']
    df['hip_rel'] = df['hip'] - df['rolling_hip_5']

    # Interaction features
    df['hip_elbow_ratio'] = df['hip'] / (df['elbow'] + 1e-6)
    df['body_line_hip_diff'] = df['body_line'] - (180 - df['hip'])

    # Cumulative rep stats — expanding min/max within each rep
    # This only looks at frames up to the current one (no future leakage)
    for col in ['elbow', 'hip']:
        grp = df.groupby(['subject_id', 'rep_number'])[col]
        df[f'{col}_cum_min'] = grp.cummin()
        df[f'{col}_cum_max'] = grp.cummax()
        df[f'{col}_cum_range'] = df[f'{col}_cum_max'] - df[f'{col}_cum_min']

    return df


def get_class_weights(y_series):
    counts = y_series.value_counts()
    total = len(y_series)
    n = len(counts)
    raw = {label: total / (n * count) for label, count in counts.items()}
    # Normalize so minimum weight = 1.0, then cap max
    min_w = min(raw.values())
    weights = {}
    for label, w in raw.items():
        normalized = w / min_w
        weights[label] = round(min(normalized, 8.0), 2)
    return weights


print("=" * 65)
print("RehabMate Training Pipeline v5")
print("Frame-Level + Cumulative Rep Stats | GroupKFold | No Leakage")
print("=" * 65)

dfs = [load_csv_robust(DATA_PATH)]
if os.path.exists(PASTED_CSV):
    dfs.append(load_csv_robust(PASTED_CSV))
    print("  Merged pasted_data.csv")
df = pd.concat(dfs, ignore_index=True)

# Drop tiny subjects
min_samples = 20
subj_counts = df['subject_id'].value_counts()
valid = subj_counts[subj_counts >= min_samples].index.tolist()
df = df[df['subject_id'].isin(valid)].copy()

df = add_features(df)
df = df.dropna(subset=FEATURES)

le = LabelEncoder()
X_df = df[FEATURES].copy()
y = le.fit_transform(df['form_quality'].values)
groups = df['subject_id'].values

raw_weights = get_class_weights(df['form_quality'])
encoded_weights = {le.transform([l])[0]: w for l, w in raw_weights.items()}

n_subjects = len(np.unique(groups))
print(f"\n[Data]    {len(df)} frames  |  {len(FEATURES)} features  |  {len(le.classes_)} classes  |  {n_subjects} subjects")
print(f"[Classes] {list(le.classes_)}")
print(f"\n[Distribution]")
for label, cnt in df['form_quality'].value_counts().sort_values().items():
    w = raw_weights.get(label, 1.0)
    print(f"  {label:<20}: {cnt:>5}  weight={w:.2f}")

# Split — use 3 folds since some subjects are small
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_df, y, groups))

X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
X_test_df  = X_df.iloc[test_idx].reset_index(drop=True)
y_train = y[train_idx]
y_test  = y[test_idx]
groups_train = groups[train_idx]

n_train_groups = len(np.unique(groups_train))
cv_folds = min(3, n_train_groups)
if HAS_SGK:
    gkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n[Split] Train={len(y_train)} Test={len(y_test)} | StratifiedGroupKFold({cv_folds})")
else:
    gkf = GroupKFold(n_splits=cv_folds)
    print(f"\n[Split] Train={len(y_train)} Test={len(y_test)} | GroupKFold({cv_folds})")

# Debug: check fold class distribution
print("\n[Fold Check]")
for fold_i, (tr_i, val_i) in enumerate(gkf.split(X_train_df, y_train, groups_train)):
    tr_classes = len(np.unique(y_train[tr_i]))
    val_classes = len(np.unique(y_train[val_i]))
    tr_subj = len(np.unique(groups_train[tr_i]))
    val_subj = len(np.unique(groups_train[val_i]))
    print(f"  Fold {fold_i}: train={len(tr_i)} ({tr_classes} cls, {tr_subj} subj) | val={len(val_i)} ({val_classes} cls, {val_subj} subj)")

pipeline_start = time.time()

# STEP 1: Optuna — LightGBM
print(f"\n{'─'*65}")
print(f"STEP 1: Optuna — LightGBM ({N_TRIALS} trials)")
print(f"{'─'*65}")

def lgbm_objective(trial):
    params = {
        'n_estimators':       trial.suggest_int('n_estimators', 50, 400),
        'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':          trial.suggest_int('max_depth', 3, 8),
        'num_leaves':         trial.suggest_int('num_leaves', 8, 50),
        'min_child_samples':  trial.suggest_int('min_child_samples', 3, 20),
        'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':          trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':         trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': 2,
    }
    try:
        scores = cross_val_score(lgb.LGBMClassifier(**params),
                                 X_train_df, y_train, cv=gkf, groups=groups_train,
                                 scoring='accuracy', n_jobs=1)
        result = scores.mean()
        if np.isnan(result):
            return 0.0
        return result
    except Exception as e:
        return 0.0

t0 = time.time()
lgbm_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
lgbm_study.optimize(lgbm_objective, n_trials=N_TRIALS, show_progress_bar=True)
lgbm_time = time.time() - t0

lgbm_best = lgbm_study.best_params.copy()
lgbm_best.update({'class_weight': 'balanced', 'random_state': RANDOM_STATE,
                   'verbose': -1, 'n_jobs': 2})

print(f"\n[LightGBM] Best CV: {lgbm_study.best_value:.4f} ({lgbm_time:.0f}s)")
for k, v in lgbm_study.best_params.items():
    print(f"  {k:<25}: {v}")

with open(f"{OUTPUT_DIR}/lgbm_best_params.json", 'w') as f:
    json.dump(lgbm_study.best_params, f, indent=2)

# STEP 2: Optuna — RandomForest
print(f"\n{'─'*65}")
print(f"STEP 2: Optuna — RandomForest ({N_TRIALS} trials)")
print(f"{'─'*65}")

def rf_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 50, 400),
        'max_depth':        trial.suggest_int('max_depth', 3, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'min_samples_split':trial.suggest_int('min_samples_split', 2, 15),
        'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
        'bootstrap':        trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE, 'n_jobs': 2,
    }
    try:
        scores = cross_val_score(RandomForestClassifier(**params),
                                 X_train_df.values, y_train, cv=gkf, groups=groups_train,
                                 scoring='accuracy', n_jobs=1)
        result = scores.mean()
        if np.isnan(result):
            return 0.0
        return result
    except Exception:
        return 0.0

t0 = time.time()
rf_study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
rf_study.optimize(rf_objective, n_trials=N_TRIALS, show_progress_bar=True)
rf_time = time.time() - t0

rf_best = rf_study.best_params.copy()
rf_best.update({'class_weight': encoded_weights, 'random_state': RANDOM_STATE, 'n_jobs': 2})

print(f"\n[RF] Best CV: {rf_study.best_value:.4f} ({rf_time:.0f}s)")
for k, v in rf_study.best_params.items():
    print(f"  {k:<25}: {v}")

with open(f"{OUTPUT_DIR}/rf_best_params.json", 'w') as f:
    json.dump(rf_study.best_params, f, indent=2)

# STEP 3: Train + evaluate
print(f"\n{'─'*65}")
print("STEP 3: Train + evaluate")
print(f"{'─'*65}")

lgbm_model = lgb.LGBMClassifier(**lgbm_best)
lgbm_model.fit(X_train_df, y_train)
lgbm_acc = accuracy_score(y_test, lgbm_model.predict(X_test_df))
print(f"  [LightGBM]     test={lgbm_acc:.4f}")

rf_model = RandomForestClassifier(**rf_best)
rf_model.fit(X_train_df.values, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_df.values))
print(f"  [RandomForest] test={rf_acc:.4f}")

# Manual ensemble
lgbm_p = lgbm_model.predict_proba(X_test_df)
rf_p   = rf_model.predict_proba(X_test_df.values)
ens_p  = (3 * lgbm_p + 2 * rf_p) / 5.0
ens_pred = np.argmax(ens_p, axis=1)
ens_acc = accuracy_score(y_test, ens_pred)
print(f"  [Ensemble 3:2] test={ens_acc:.4f}")

print(f"\n--- Classification Report ---")
print(classification_report(y_test, ens_pred, target_names=le.classes_, zero_division=0))

report = classification_report(y_test, ens_pred, target_names=le.classes_,
                               output_dict=True, zero_division=0)
with open(f"{OUTPUT_DIR}/classification_report.json", 'w') as f:
    json.dump(report, f, indent=2)

# STEP 4: GroupKFold CV full data
print(f"\n{'─'*65}")
print("STEP 4: GroupKFold CV (full data)")
print(f"{'─'*65}")

cv_all = min(3, n_subjects)
gkf_all = GroupKFold(n_splits=cv_all)

cv_results = {}
for name, mdl in [('LightGBM', lgb.LGBMClassifier(**lgbm_best)),
                   ('RandomForest', RandomForestClassifier(**rf_best))]:
    Xc = X_df if name == 'LightGBM' else X_df.values
    scores = cross_val_score(mdl, Xc, y, cv=gkf_all, groups=groups,
                             scoring='accuracy', n_jobs=-1)
    cv_results[name] = {'mean': float(scores.mean()), 'std': float(scores.std())}
    print(f"  {name:<14}: {scores.mean():.4f} +/- {scores.std():.4f}")

with open(f"{OUTPUT_DIR}/cv_results.json", 'w') as f:
    json.dump(cv_results, f, indent=2)

# STEP 5: SHAP
print(f"\n{'─'*65}")
print("STEP 5: SHAP")
print(f"{'─'*65}")

explainer = shap.TreeExplainer(lgbm_model)
sv = explainer.shap_values(X_test_df.iloc[:min(500, len(X_test_df))])
if isinstance(sv, list):
    ms = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
else:
    ms = np.abs(sv).mean(axis=0)
    if ms.ndim > 1: ms = ms.mean(axis=1)

shap_df = pd.DataFrame({'feature': FEATURES, 'shap': ms}).sort_values('shap', ascending=False)
shap_df.to_csv(f"{OUTPUT_DIR}/shap_importance.csv", index=False)
mx = shap_df['shap'].max()
for _, r in shap_df.iterrows():
    bar = '#' * int(r['shap'] / mx * 20)
    print(f"  {r['feature']:<25} {r['shap']:>8.4f}  {bar}")

# STEP 6: Final retrain
print(f"\n{'─'*65}")
print("STEP 6: Final retrain on 100% data")
print(f"{'─'*65}")

final_lgbm = lgb.LGBMClassifier(**lgbm_best)
final_lgbm.fit(X_df, y)
final_rf = RandomForestClassifier(**rf_best)
final_rf.fit(X_df.values, y)

MODEL_FILE = 'rehabmate_v3.pkl'
with open(f"{OUTPUT_DIR}/{MODEL_FILE}", 'wb') as f:
    pickle.dump({
        'lgbm': final_lgbm, 'rf': final_rf,
        'encoder': le, 'features': FEATURES,
        'classes': list(le.classes_), 'weights': [3, 2],
    }, f)
print(f"  Saved: {MODEL_FILE} ({len(X_df)} frames)")

total = time.time() - pipeline_start
print(f"\n{'='*65}")
print(f"DONE in {total/60:.1f} min")
print(f"  LGBM CV={lgbm_study.best_value:.4f} Test={lgbm_acc:.4f}")
print(f"  RF   CV={rf_study.best_value:.4f} Test={rf_acc:.4f}")
print(f"  Ens  Test={ens_acc:.4f}")
print(f"  GKF  LGBM={cv_results['LightGBM']['mean']:.4f} RF={cv_results['RandomForest']['mean']:.4f}")
print(f"{'='*65}")