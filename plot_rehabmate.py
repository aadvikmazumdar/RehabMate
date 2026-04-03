"""
plot_rehabmate.py
Generates 4 publication-ready figures for the RehabMate paper.

Outputs (saved to paper_figures/):
  fig1_confusion_matrix.pdf / .png
  fig2_shap_importance.pdf / .png
  fig3_cv_comparison.pdf / .png
  fig4_per_class_metrics.pdf / .png

Usage:
  python plot_rehabmate.py

Requirements:
  pip install matplotlib seaborn scikit-learn lightgbm shap pandas numpy
"""

import os, pickle, csv, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH   = "rehabmate_v3.pkl"
DATA_PATH    = "pushup_data_v2.csv"
PASTED_CSV   = "pasted_data.csv"
OUT_DIR      = "paper_figures"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ── Publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "axes.linewidth":     0.8,
})

# Grayscale-safe palette
GRAY_DARK   = "#1a1a1a"
GRAY_MED    = "#555555"
GRAY_LIGHT  = "#aaaaaa"
GRAY_PALE   = "#dddddd"
ACCENT      = "#2c2c2c"
BAR_COLORS  = ["#1a1a1a", "#555555", "#888888", "#bbbbbb"]

def save(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.pdf")
    fig.savefig(f"{OUT_DIR}/{name}.png")
    print(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


# ── Data loading ───────────────────────────────────────────────────────────
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

HEADER_18 = [
    'elbow', 'elbow_flare', 'hip', 'wrist', 'shoulder_elevation', 'body_line',
    'elbow_velocity', 'hip_velocity', 'time_in_phase', 'rep_number',
    'elbow_drift', 'hip_drift', 'velocity_trend',
    'rolling_elbow_5', 'rolling_hip_5', 'phase', 'form_quality', 'subject_id'
]

def load_csv_robust(path):
    old_rows, new_rows = [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if   len(row) == 17: old_rows.append(row)
            elif len(row) == 18: new_rows.append(row)
    if old_rows:
        prev_rep = -1; sessions, current = [], []
        for row in old_rows:
            rep = int(float(row[9]))
            if rep == 0 and prev_rep > 0 and current:
                sessions.append(current); current = []
            current.append(row); prev_rep = rep
        if current: sessions.append(current)
        for si, sess in enumerate(sessions):
            for row in sess: row.append(str(100 + si)); new_rows.append(row)
    df = pd.DataFrame(new_rows, columns=HEADER_18)
    for col in HEADER_18:
        if col not in ('phase', 'form_quality', 'subject_id'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['subject_id'] = df['subject_id'].astype(str)
    df['rep_number']  = df['rep_number'].astype(int)
    return df

def add_features(df):
    df['elbow_rel']          = df['elbow'] - df['rolling_elbow_5']
    df['hip_rel']            = df['hip']   - df['rolling_hip_5']
    df['hip_elbow_ratio']    = df['hip']   / (df['elbow'] + 1e-6)
    df['body_line_hip_diff'] = df['body_line'] - (180 - df['hip'])
    for col in ['elbow', 'hip']:
        grp = df.groupby(['subject_id', 'rep_number'])[col]
        df[f'{col}_cum_min']   = grp.cummin()
        df[f'{col}_cum_max']   = grp.cummax()
        df[f'{col}_cum_range'] = df[f'{col}_cum_max'] - df[f'{col}_cum_min']
    for col in ['elbow', 'hip', 'body_line']:
        grp = df.groupby('subject_id')[col]
        df[f'{col}_znorm'] = grp.transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    return df

# ── Load ───────────────────────────────────────────────────────────────────
print("Loading model and data...")
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

lgbm_model = bundle['lgbm']
rf_model   = bundle['rf']
le         = bundle['encoder']
classes    = bundle['classes']
weights    = bundle.get('weights', [3, 2])

dfs = [load_csv_robust(DATA_PATH)]
if os.path.exists(PASTED_CSV):
    dfs.append(load_csv_robust(PASTED_CSV))
df = pd.concat(dfs, ignore_index=True)

min_samples = 20
sc = df['subject_id'].value_counts()
valid = sc[sc >= min_samples].index.tolist()
df = df[df['subject_id'].isin(valid)].copy()
df = add_features(df)
df = df.dropna(subset=FEATURES)

X_df     = df[FEATURES].copy()
y        = le.transform(df['form_quality'].values)
groups   = df['subject_id'].values
n_subj   = len(np.unique(groups))
print(f"  {len(df)} frames | {len(classes)} classes | {n_subj} subjects")

# ── LOSO predictions for CM + per-class metrics ────────────────────────────
print("Running LOSO inference for confusion matrix...")
all_true, all_pred = [], []
for ts in sorted(valid, key=lambda x: int(x)):
    tr = df[df['subject_id'] != ts]
    te = df[df['subject_id'] == ts]
    X_tr = pd.DataFrame(tr[FEATURES].values, columns=FEATURES)
    y_tr = le.transform(tr['form_quality'].values)
    X_te = pd.DataFrame(te[FEATURES].values, columns=FEATURES)
    y_te = le.transform(te['form_quality'].values)

    fw = {}
    vc = tr['form_quality'].value_counts()
    tot = len(tr); n = len(vc)
    min_w = min(tot / (n * c) for c in vc.values)
    for lbl, cnt in vc.items():
        enc = le.transform([lbl])[0]
        raw = (tot / (n * cnt)) / min_w
        fw[enc] = round(min(raw, 8.0), 2)

    m = lgb.LGBMClassifier(**{**lgbm_model.get_params(),
                               'class_weight': fw, 'random_state': RANDOM_STATE,
                               'verbose': -1, 'n_jobs': -1})
    m.fit(X_tr, y_tr)
    yp = m.predict(X_te)
    all_true.extend(y_te.tolist())
    all_pred.extend(yp.tolist())

y_true = np.array(all_true)
y_pred = np.array(all_pred)


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Confusion matrix")

cm_raw = confusion_matrix(y_true, y_pred)
cm_pct = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True) * 100

n_cls   = len(classes)
fig, ax = plt.subplots(figsize=(4.5, 3.8))

# Grayscale heatmap (reversed so dark = high)
im = ax.imshow(cm_pct, cmap='Greys', vmin=0, vmax=100, aspect='auto')

# Colour-bar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row %", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Annotations
for i in range(n_cls):
    for j in range(n_cls):
        val  = cm_pct[i, j]
        raw  = cm_raw[i, j]
        col  = "white" if val > 55 else GRAY_DARK
        ax.text(j, i, f"{val:.1f}%\n({raw})",
                ha='center', va='center', fontsize=8, color=col)

tick_labels = [c.replace("_", "\n") for c in classes]
ax.set_xticks(range(n_cls)); ax.set_xticklabels(tick_labels, fontsize=8)
ax.set_yticks(range(n_cls)); ax.set_yticklabels(tick_labels, fontsize=8)
ax.set_xlabel("Predicted Label", fontsize=10)
ax.set_ylabel("True Label", fontsize=10)
ax.set_title("Confusion Matrix (LOSO-CV)", fontsize=11, fontweight='bold', pad=10)
ax.grid(False)
fig.tight_layout()
save(fig, "fig1_confusion_matrix")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — SHAP Feature Importance
# ══════════════════════════════════════════════════════════════════════════
print("Figure 2: SHAP feature importance")

shap_csv = "pipeline_outputs/shap_importance.csv"
if os.path.exists(shap_csv):
    shap_df = pd.read_csv(shap_csv)
else:
    print("  Computing SHAP (no cached CSV found)...")
    explainer = shap.TreeExplainer(lgbm_model)
    sv = explainer.shap_values(X_df.iloc[:min(600, len(X_df))])
    if isinstance(sv, list):
        ms = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        ms = np.abs(sv).mean(axis=0)
        if ms.ndim > 1: ms = ms.mean(axis=1)
    shap_df = pd.DataFrame({'feature': FEATURES, 'shap': ms})

shap_df = shap_df.sort_values('shap', ascending=True).tail(15)  # top-15

fig, ax = plt.subplots(figsize=(5.5, 4.8))

# Assign shade based on rank
n   = len(shap_df)
grays = [plt.cm.Greys(0.35 + 0.55 * i / (n - 1)) for i in range(n)]

bars = ax.barh(range(n), shap_df['shap'].values,
               color=grays, edgecolor='none', height=0.65)

# Value labels
for i, (bar, val) in enumerate(zip(bars, shap_df['shap'].values)):
    ax.text(val + shap_df['shap'].max() * 0.01, i, f"{val:.4f}",
            va='center', ha='left', fontsize=7.5, color=GRAY_DARK)

feat_labels = [f.replace('_', ' ') for f in shap_df['feature'].values]
ax.set_yticks(range(n))
ax.set_yticklabels(feat_labels, fontsize=8.5)
ax.set_xlabel("Mean |SHAP| Value", fontsize=10)
ax.set_title("Feature Importance (SHAP, LightGBM)\nTop 15 Features", fontsize=11,
             fontweight='bold', pad=10)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)
ax.set_xlim(right=shap_df['shap'].max() * 1.18)
fig.tight_layout()
save(fig, "fig2_shap_importance")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — CV Accuracy Comparison
# ══════════════════════════════════════════════════════════════════════════
print("Figure 3: CV accuracy comparison")

# LOSO (already computed above)
loso_acc = accuracy_score(y_true, y_pred)

# GroupKFold (subject-aware)
n_folds_gkf = min(3, n_subj)
gkf = GroupKFold(n_splits=n_folds_gkf)
gkf_scores = cross_val_score(
    lgb.LGBMClassifier(**{**lgbm_model.get_params(),
                          'class_weight': 'balanced',
                          'random_state': RANDOM_STATE,
                          'verbose': -1, 'n_jobs': -1}),
    X_df, y, cv=gkf, groups=groups, scoring='accuracy', n_jobs=-1)

# StratifiedKFold (inflated baseline)
SKF_FEATURES = [f for f in FEATURES
                if f not in ('elbow_znorm', 'hip_znorm', 'body_line_znorm')]
X_skf = df[[f for f in SKF_FEATURES if f in df.columns]].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
skf_params = {**lgbm_model.get_params(),
              'class_weight': 'balanced',
              'random_state': RANDOM_STATE,
              'verbose': -1, 'n_jobs': -1}
skf_scores = cross_val_score(
    lgb.LGBMClassifier(**skf_params),
    X_skf, y, cv=skf, scoring='accuracy', n_jobs=-1)

labels = [
    "StratifiedKFold\n(5-fold, inflated)",
    f"GroupKFold\n({n_folds_gkf}-fold, subject-aware)",
    "LOSO-CV\n(held-out subject)",
]
means  = [skf_scores.mean(), gkf_scores.mean(), loso_acc]
stds   = [skf_scores.std(),  gkf_scores.std(),  0.0]

fig, ax = plt.subplots(figsize=(5.2, 3.6))

x = np.arange(len(labels))
bar_colors_cv = [GRAY_PALE, GRAY_MED, GRAY_DARK]
bars = ax.bar(x, means, yerr=stds, capsize=4,
              color=bar_colors_cv, edgecolor=GRAY_DARK,
              linewidth=0.7, width=0.52, error_kw={'linewidth': 1.0})

# Annotate
for i, (m, s) in enumerate(zip(means, stds)):
    label = f"{m:.4f}" if s == 0 else f"{m:.4f}±{s:.4f}"
    ax.text(i, m + (s or 0) + 0.004, label,
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Annotation arrow: "data leakage"
ax.annotate("← data leakage", xy=(0, means[0]),
            xytext=(0.48, means[0] + 0.02),
            fontsize=8, color='#666666',
            arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Accuracy", fontsize=10)
ax.set_ylim(0, min(1.0, max(means) + max(stds) * 2 + 0.08))
ax.set_title("Cross-Validation Strategy Comparison", fontsize=11,
             fontweight='bold', pad=10)

# Legend patch
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor=GRAY_PALE,   edgecolor=GRAY_DARK, label='Naive (inflated)'),
    Patch(facecolor=GRAY_MED,    edgecolor=GRAY_DARK, label='Subject-aware'),
    Patch(facecolor=GRAY_DARK,   edgecolor=GRAY_DARK, label='LOSO (honest)'),
]
ax.legend(handles=legend_els, fontsize=8, loc='lower right',
          framealpha=0.6, edgecolor=GRAY_PALE)
fig.tight_layout()
save(fig, "fig3_cv_comparison")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Per-class Precision / Recall / F1
# ══════════════════════════════════════════════════════════════════════════
print("Figure 4: Per-class P/R/F1")

report = classification_report(y_true, y_pred, target_names=classes,
                               output_dict=True, zero_division=0)
metrics_data = {cls: {
    'Precision': report[cls]['precision'],
    'Recall':    report[cls]['recall'],
    'F1-Score':  report[cls]['f1-score'],
} for cls in classes}

n_cls   = len(classes)
metrics = ['Precision', 'Recall', 'F1-Score']
n_met   = len(metrics)
x       = np.arange(n_cls)
width   = 0.24
offsets = [-width, 0, width]
m_colors = [GRAY_DARK, GRAY_MED, GRAY_PALE]
hatches  = ['', '///', '...']

fig, ax = plt.subplots(figsize=(max(5, n_cls * 1.4), 3.8))

for mi, (metric, color, hatch, offset) in enumerate(
        zip(metrics, m_colors, hatches, offsets)):
    vals = [metrics_data[cls][metric] for cls in classes]
    ax.bar(x + offset, vals, width, label=metric,
           color=color, hatch=hatch,
           edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Per-Class Precision, Recall & F1-Score (LOSO-CV)",
             fontsize=11, fontweight='bold', pad=10)
ax.legend(fontsize=9, loc='lower right', framealpha=0.7,
          edgecolor=GRAY_PALE)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# Support annotation
for i, cls in enumerate(classes):
    sup = int(report[cls]['support'])
    ax.text(i, 1.06, f"n={sup}", ha='center', fontsize=7.5, color=GRAY_MED)

fig.tight_layout()
save(fig, "fig4_per_class_metrics")


# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("All figures saved to paper_figures/")
print(f"  LOSO accuracy : {loso_acc:.4f}")
print(f"  GKF  accuracy : {gkf_scores.mean():.4f} ± {gkf_scores.std():.4f}")
print(f"  SKF  accuracy : {skf_scores.mean():.4f} ± {skf_scores.std():.4f}")
print("=" * 55)