import pandas as pd
import numpy as np
import json
import warnings
import time
import csv
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb

RAW_DATA_PATH  = "pushup_data_v2.csv"
PASTED_CSV     = "pasted_data.csv"
PARAMS_PATH    = "pipeline_outputs/lgbm_best_params.json"
OUTPUT_DIR     = "pipeline_outputs"
RANDOM_STATE   = 42

FEATURES = [
    'elbow', 'elbow_flare', 'hip', 'shoulder_elevation', 'body_line',
    'elbow_velocity', 'hip_velocity', 'time_in_phase',
    'elbow_drift', 'hip_drift', 'velocity_trend',
    'elbow_rel', 'hip_rel',
    'hip_elbow_ratio', 'body_line_hip_diff',
    'elbow_cum_min', 'elbow_cum_max', 'elbow_cum_range',
    'hip_cum_min', 'hip_cum_max', 'hip_cum_range',
    'elbow_znorm', 'hip_znorm', 'body_line_znorm',
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
            if len(row) == 17: old_rows.append(row)
            elif len(row) == 18: new_rows.append(row)

    if old_rows:
        prev_rep = -1
        sessions, current = [], []
        for row in old_rows:
            rep = int(float(row[9]))
            if rep == 0 and prev_rep > 0 and current:
                sessions.append(current); current = []
            current.append(row); prev_rep = rep
        if current: sessions.append(current)
        for si, sess in enumerate(sessions):
            for row in sess:
                row.append(str(100 + si)); new_rows.append(row)

    df = pd.DataFrame(new_rows, columns=header_18)
    for col in header_18:
        if col not in ('phase', 'form_quality', 'subject_id'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['subject_id'] = df['subject_id'].astype(str)
    df['rep_number'] = df['rep_number'].astype(int)
    return df


def add_features(df):
    df['elbow_rel'] = df['elbow'] - df['rolling_elbow_5']
    df['hip_rel'] = df['hip'] - df['rolling_hip_5']
    df['hip_elbow_ratio'] = df['hip'] / (df['elbow'] + 1e-6)
    df['body_line_hip_diff'] = df['body_line'] - (180 - df['hip'])
    for col in ['elbow', 'hip']:
        grp = df.groupby(['subject_id', 'rep_number'])[col]
        df[f'{col}_cum_min'] = grp.cummin()
        df[f'{col}_cum_max'] = grp.cummax()
        df[f'{col}_cum_range'] = df[f'{col}_cum_max'] - df[f'{col}_cum_min']
    for col in ['elbow', 'hip', 'body_line']:
        grp = df.groupby('subject_id')[col]
        df[f'{col}_znorm'] = grp.transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    return df


def get_class_weights_enc(y_series, le):
    counts = y_series.value_counts()
    total = len(y_series)
    n = len(counts)
    raw = {l: total / (n * c) for l, c in counts.items()}
    min_w = min(raw.values())
    return {le.transform([l])[0]: round(min(w / min_w, 8.0), 2) for l, w in raw.items()}


print("=" * 60)
print("RehabMate — LOSO CV (Frame + Cumulative Stats)")
print("=" * 60)

dfs = [load_csv_robust(RAW_DATA_PATH)]
if os.path.exists(PASTED_CSV):
    dfs.append(load_csv_robust(PASTED_CSV))
df = pd.concat(dfs, ignore_index=True)

min_samples = 20
sc = df['subject_id'].value_counts()
valid = sc[sc >= min_samples].index.tolist()
df = df[df['subject_id'].isin(valid)].copy()
df = add_features(df)
df = df.dropna(subset=FEATURES)

le = LabelEncoder()
le.fit(df['form_quality'].values)

print(f"\n[Data] {len(df)} frames, {len(valid)} subjects")
for sid in sorted(valid, key=lambda x: int(x)):
    sub = df[df['subject_id'] == sid]
    print(f"  {sid:>6}: {len(sub):>4} rows, {sub['form_quality'].nunique()} cls")

with open(PARAMS_PATH) as f:
    best_params = json.load(f)

print(f"\n{'─'*60}")
print("LOSO CV...")
print(f"{'─'*60}")

all_true, all_pred = [], []
fold_results = []
t0 = time.time()

for i, ts in enumerate(sorted(valid, key=lambda x: int(x))):
    tr = df[df['subject_id'] != ts]
    te = df[df['subject_id'] == ts]

    X_tr = pd.DataFrame(tr[FEATURES].values, columns=FEATURES)
    y_tr = le.transform(tr['form_quality'].values)
    X_te = pd.DataFrame(te[FEATURES].values, columns=FEATURES)
    y_te = le.transform(te['form_quality'].values)

    fw = get_class_weights_enc(tr['form_quality'], le)
    p = dict(best_params)
    p.update({'class_weight': fw, 'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1})

    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, y_tr)
    yp = m.predict(X_te)
    acc = accuracy_score(y_te, yp)

    all_true.extend(y_te.tolist())
    all_pred.extend(yp.tolist())
    fold_results.append({'subject': ts, 'n': len(te), 'acc': round(acc, 4),
                         'cls': sorted(te['form_quality'].unique().tolist())})
    print(f"  {i+1:>2} | {ts:>6} | n={len(te):>4} | acc={acc:.4f}")

accs = [r['acc'] for r in fold_results]
print(f"\n{'─'*60}")
print(f"  LOSO: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (min={np.min(accs):.4f} max={np.max(accs):.4f})")
print(f"{'─'*60}")
print(classification_report(all_true, all_pred, target_names=le.classes_, zero_division=0))

cm = pd.DataFrame(confusion_matrix(all_true, all_pred), index=le.classes_, columns=le.classes_)
print(cm.to_string())

with open(f"{OUTPUT_DIR}/loso_cv_results.json", 'w') as f:
    json.dump({'mean': round(np.mean(accs), 4), 'std': round(np.std(accs), 4),
               'folds': fold_results}, f, indent=2)
cm.to_csv(f"{OUTPUT_DIR}/loso_confusion_matrix.csv")
print(f"\n  Saved results to {OUTPUT_DIR}/")