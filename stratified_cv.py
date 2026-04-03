"""
stratified_cv.py
Run this ONCE to get the StratifiedKFold number for the paper comparison table.
This is NOT the honest metric — it's for showing what happens without subject-aware splitting.
"""
import pandas as pd
import numpy as np
import json
import warnings
import csv
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

DATA_PATH = "pushup_data_v2.csv"
PARAMS_PATH = "pipeline_outputs/lgbm_best_params.json"

FEATURES = [
    'elbow', 'elbow_flare', 'hip', 'shoulder_elevation', 'body_line',
    'elbow_velocity', 'hip_velocity', 'time_in_phase',
    'elbow_drift', 'hip_drift', 'velocity_trend',
    'elbow_rel', 'hip_rel',
    'hip_elbow_ratio', 'body_line_hip_diff',
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
    old_rows, new_rows = [], []
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
    return df


print("=" * 50)
print("StratifiedKFold CV (for paper comparison)")
print("NOT subject-aware — same person in train+test")
print("=" * 50)

df = load_csv_robust(DATA_PATH)
df = add_features(df)
df = df.dropna(subset=FEATURES)

le = LabelEncoder()
X = df[FEATURES].values
y = le.fit_transform(df['form_quality'].values)

print(f"\n[Data] {len(df)} frames, {len(FEATURES)} features, {len(le.classes_)} classes")

with open(PARAMS_PATH) as f:
    params = json.load(f)
params.update({'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1})

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning 5-fold StratifiedKFold CV...")
scores = cross_val_score(lgb.LGBMClassifier(**params), X, y, cv=skf, scoring='accuracy', n_jobs=-1)

print(f"\nResults:")
print(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")
print(f"  Mean: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"\n  This is the number for the RehabMate* row in the paper.")
print(f"  It is inflated because the same subject appears in train+test.")