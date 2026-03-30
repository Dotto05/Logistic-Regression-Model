# ================================================
# HEART DISEASE PREDICTION — FUZZY LOGISTIC REGRESSION
# Single file: fuzzy preprocessing + model +
# baseline comparison + membership function plots
# ================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ================================================
# STEP 1: LOAD DATA
# ================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))

print(f"Dataset loaded: {len(df)} total samples")

# ================================================
# STEP 2: FUZZY MEMBERSHIP FUNCTIONS
# Why fuzzy? Medical data is inherently uncertain.
# A cholesterol of 201 isn't cleanly "high" — it
# sits right at the boundary. Fuzzy logic preserves
# that uncertainty by assigning graded membership
# scores instead of hard 0/1 labels, which better
# reflects how clinicians actually reason about risk.
# ================================================
def fuzzy_age(age):
    if age <= 40:
        return 0.3        # Young — lower risk
    elif age <= 60:
        return 0.6        # Middle-aged — moderate risk
    else:
        return 0.9        # Elderly — higher risk

def fuzzy_chol(chol):
    if chol <= 180:
        return 0.3        # Low cholesterol
    elif chol <= 240:
        return 0.6        # Borderline high
    else:
        return 0.9        # High cholesterol

def fuzzy_bp(bp):
    if bp <= 120:
        return 0.3        # Normal
    elif bp <= 140:
        return 0.6        # Elevated / pre-hypertension
    else:
        return 0.9        # High blood pressure

# ================================================
# STEP 3: PLOT FUZZY MEMBERSHIP FUNCTIONS
# ================================================
plt.rcParams.update({
    "figure.facecolor": "#f9f9f9",
    "axes.facecolor":   "#ffffff",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

fig = plt.figure(figsize=(15, 4.5))
fig.suptitle(
    "Fuzzy Membership Functions — Heart Disease Prediction",
    fontsize=14, fontweight="bold", y=1.01
)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# --- Age ---
ax1 = fig.add_subplot(gs[0])
age_x  = np.linspace(0, 80, 300)
young  = np.maximum(0, np.minimum((40 - age_x) / 40, 1))
middle = np.maximum(0, np.minimum((age_x - 30) / 20, (60 - age_x) / 20))
old    = np.maximum(0, np.minimum((age_x - 50) / 30, 1))

ax1.plot(age_x, young,  label="Young",       color="#2176ae", linewidth=2)
ax1.plot(age_x, middle, label="Middle-aged", color="#28a745", linewidth=2)
ax1.plot(age_x, old,    label="Elderly",     color="#d9534f", linewidth=2)
ax1.axvline(40, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.axvline(60, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.text(40, 1.02, "40", ha="center", fontsize=8, color="gray")
ax1.text(60, 1.02, "60", ha="center", fontsize=8, color="gray")
ax1.annotate("→ score: 0.3", xy=(20, 0.5), fontsize=8, color="#2176ae", alpha=0.8)
ax1.annotate("→ score: 0.6", xy=(44, 0.5), fontsize=8, color="#28a745", alpha=0.8)
ax1.annotate("→ score: 0.9", xy=(65, 0.5), fontsize=8, color="#d9534f", alpha=0.8)
ax1.set_title("Age", fontweight="bold")
ax1.set_xlabel("Age (years)")
ax1.set_ylabel("Membership degree")
ax1.set_ylim(-0.05, 1.15)
ax1.legend(fontsize=8)

# --- Cholesterol ---
ax2 = fig.add_subplot(gs[1])
chol_x = np.linspace(100, 400, 300)
c_low  = np.maximum(0, np.minimum((200 - chol_x) / 100, 1))
c_med  = np.maximum(0, np.minimum((chol_x - 150) / 50, (250 - chol_x) / 50))
c_high = np.maximum(0, np.minimum((chol_x - 200) / 100, 1))

ax2.plot(chol_x, c_low,  label="Low",    color="#2176ae", linewidth=2)
ax2.plot(chol_x, c_med,  label="Medium", color="#28a745", linewidth=2)
ax2.plot(chol_x, c_high, label="High",   color="#d9534f", linewidth=2)
ax2.axvline(180, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax2.axvline(240, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax2.text(180, 1.02, "180", ha="center", fontsize=8, color="gray")
ax2.text(240, 1.02, "240", ha="center", fontsize=8, color="gray")
ax2.annotate("→ 0.3", xy=(115, 0.5), fontsize=8, color="#2176ae", alpha=0.8)
ax2.annotate("→ 0.6", xy=(195, 0.5), fontsize=8, color="#28a745", alpha=0.8)
ax2.annotate("→ 0.9", xy=(305, 0.5), fontsize=8, color="#d9534f", alpha=0.8)
ax2.set_title("Cholesterol", fontweight="bold")
ax2.set_xlabel("Cholesterol (mg/dL)")
ax2.set_ylabel("Membership degree")
ax2.set_ylim(-0.05, 1.15)
ax2.legend(fontsize=8)

# --- Blood Pressure ---
ax3 = fig.add_subplot(gs[2])
bp_x     = np.linspace(80, 200, 300)
bp_norm  = np.maximum(0, np.minimum((120 - bp_x) / 40, 1))
bp_elev  = np.maximum(0, np.minimum((bp_x - 110) / 20, (140 - bp_x) / 20))
bp_high  = np.maximum(0, np.minimum((bp_x - 130) / 50, 1))

ax3.plot(bp_x, bp_norm, label="Normal",   color="#2176ae", linewidth=2)
ax3.plot(bp_x, bp_elev, label="Elevated", color="#28a745", linewidth=2)
ax3.plot(bp_x, bp_high, label="High",     color="#d9534f", linewidth=2)
ax3.axvline(120, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax3.axvline(140, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax3.text(120, 1.02, "120", ha="center", fontsize=8, color="gray")
ax3.text(140, 1.02, "140", ha="center", fontsize=8, color="gray")
ax3.annotate("→ 0.3", xy=(83,  0.5), fontsize=8, color="#2176ae", alpha=0.8)
ax3.annotate("→ 0.6", xy=(121, 0.5), fontsize=8, color="#28a745", alpha=0.8)
ax3.annotate("→ 0.9", xy=(160, 0.5), fontsize=8, color="#d9534f", alpha=0.8)
ax3.set_title("Blood Pressure", fontweight="bold")
ax3.set_xlabel("BP (mmHg)")
ax3.set_ylabel("Membership degree")
ax3.set_ylim(-0.05, 1.15)
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fuzzy_membership_functions.png"), dpi=150, bbox_inches="tight")
print("Saved: fuzzy_membership_functions.png")
plt.show()

# ================================================
# STEP 4: APPLY FUZZY TRANSFORMS TO DATAFRAME
# ================================================
df["age"]      = df["age"].apply(fuzzy_age)
df["chol"]     = df["chol"].apply(fuzzy_chol)
df["trestbps"] = df["trestbps"].apply(fuzzy_bp)

# ================================================
# STEP 5: TRAIN/TEST SPLIT
# 70% training, 30% testing
# All reported metrics are on the TEST SET only
# ================================================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ================================================
# STEP 6: NORMALIZATION
# fit_transform on train only — no data leakage
# ================================================
scaler = MinMaxScaler()
X_train_fuzzy = scaler.fit_transform(X_train)
X_test_fuzzy  = scaler.transform(X_test)

# ================================================
# STEP 7: TRAIN FUZZY LOGISTIC REGRESSION MODEL
# ================================================
fuzzy_model = LogisticRegression(max_iter=1000)
fuzzy_model.fit(X_train_fuzzy, y_train)
y_pred_fuzzy = fuzzy_model.predict(X_test_fuzzy)

# ================================================
# STEP 8: BASELINE — plain logistic regression
# Same split, NO fuzzy preprocessing applied
# Used to prove the fuzzy step adds value
# ================================================
df_raw = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))
X_raw  = df_raw.drop("target", axis=1)
y_raw  = df_raw["target"]

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42
)

scaler_raw  = MinMaxScaler()
X_train_raw = scaler_raw.fit_transform(X_train_raw)
X_test_raw  = scaler_raw.transform(X_test_raw)

base_model  = LogisticRegression(max_iter=1000)
base_model.fit(X_train_raw, y_train_raw)
y_pred_base = base_model.predict(X_test_raw)

# ================================================
# STEP 9: METRICS
# ================================================
def get_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
    }

fuzzy_metrics = get_metrics(y_test, y_pred_fuzzy)
base_metrics  = get_metrics(y_test_raw, y_pred_base)
cm            = confusion_matrix(y_test, y_pred_fuzzy)

# ================================================
# STEP 10: PRINT RESULTS
# ================================================
print("\n" + "="*55)
print(f"  FUZZY LOGISTIC REGRESSION MODEL ACCURACY: "
      f"{fuzzy_metrics['accuracy']*100:.1f}%  (test set, 30%)")
print("="*55)

print("\n-------- FULL METRICS (Test Set) --------")
print(f"{'Metric':<12} {'Fuzzy Log. Reg.':>15}   {'Baseline':>10}")
print("-" * 42)
print(f"{'Accuracy':<12} {fuzzy_metrics['accuracy']*100:>14.1f}%"
      f"   {base_metrics['accuracy']*100:>9.1f}%")
print(f"{'Precision':<12} {fuzzy_metrics['precision']:>15.4f}"
      f"   {base_metrics['precision']:>10.4f}")
print(f"{'Recall':<12} {fuzzy_metrics['recall']:>15.4f}"
      f"   {base_metrics['recall']:>10.4f}")
print(f"{'F1 Score':<12} {fuzzy_metrics['f1']:>15.4f}"
      f"   {base_metrics['f1']:>10.4f}")
print("-" * 42)

diff = fuzzy_metrics['accuracy'] - base_metrics['accuracy']
if diff > 0:
    print(f"\nFuzzy preprocessing improved accuracy by "
          f"+{diff*100:.1f}pp over the baseline.")
elif diff < 0:
    print(f"\nBaseline outperformed fuzzy logistic regression by "
          f"{abs(diff)*100:.1f}pp — consider tuning thresholds.")
else:
    print("\nBoth models achieved the same accuracy.")

print("\n-------- CONFUSION MATRIX (Fuzzy Logistic Regression) --------")
print(f"                   Predicted No   Predicted Yes")
print(f"Actual No          {cm[0][0]:<15} {cm[0][1]}")
print(f"Actual Yes         {cm[1][0]:<15} {cm[1][1]}")
print(f"\nTrue Negatives  (TN): {cm[0][0]}")
print(f"False Positives (FP): {cm[0][1]}  ← healthy flagged as sick")
print(f"False Negatives (FN): {cm[1][0]}  ← disease MISSED (more costly)")
print(f"True Positives  (TP): {cm[1][1]}")

# ================================================
# STEP 11: PLOT COMPARISON BAR CHART
# ================================================
metrics_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
fuzzy_vals = [fuzzy_metrics["accuracy"], fuzzy_metrics["precision"],
              fuzzy_metrics["recall"],   fuzzy_metrics["f1"]]
base_vals  = [base_metrics["accuracy"],  base_metrics["precision"],
              base_metrics["recall"],    base_metrics["f1"]]

x     = np.arange(len(metrics_labels))
width = 0.35

fig2, ax = plt.subplots(figsize=(9, 5))
fig2.patch.set_facecolor("#f9f9f9")
ax.set_facecolor("#ffffff")

bars1 = ax.bar(x - width/2, fuzzy_vals, width,
               label="Fuzzy Logistic Regression", color="#2176ae", alpha=0.85)
bars2 = ax.bar(x + width/2, base_vals,  width,
               label="Baseline (no fuzzy)", color="#adb5bd", alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=9, color="#2176ae", fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=9, color="#555", fontweight="bold")

ax.set_title("Fuzzy Logistic Regression vs Baseline — Performance Comparison",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.12)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "comparison_chart.png"), dpi=150, bbox_inches="tight")
print("\nSaved: comparison_chart.png")
plt.show()