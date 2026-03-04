"""
Error analysis for binary PCL classification.
Compares predicted labels (dev.txt) against true labels (dev_semeval_parids-labels.csv).
"""

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
)

CATEGORY_NAMES = [
    "Unbalanced Power",
    "Shallow Solution",
    "Presupposition",
    "Authority Voice",
    "Metaphor",
    "Compassion",
    "Poorer the Merrier",
]

# ── Load data ──────────────────────────────────────────────────────────────────

dev_labels_df = pd.read_csv("dev_semeval_parids-labels.csv")
dev_labels_df["label_vec"] = dev_labels_df["label"].apply(ast.literal_eval)

# True binary label: 1 if ANY subcategory is 1, else 0
dev_labels_df["true_label"] = dev_labels_df["label_vec"].apply(
    lambda v: 1 if any(v) else 0
)

predictions = np.loadtxt("dev.txt", dtype=int)

y_true = dev_labels_df["true_label"].values
y_pred = predictions

assert len(y_true) == len(y_pred), (
    f"Length mismatch: {len(y_true)} true labels vs {len(y_pred)} predictions"
)

# ── Confusion Matrix ──────────────────────────────────────────────────────────

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=["No PCL (0)", "PCL (1)"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Binary PCL Classification — Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/error_cm.png", dpi=150)
plt.close()
print("Saved: outputs/error_cm.png")

# ── Classification statistics ─────────────────────────────────────────────────

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(
    classification_report(
        y_true, y_pred, target_names=["No PCL (0)", "PCL (1)"], digits=4
    )
)

print("=" * 60)
print("ADDITIONAL METRICS")
print("=" * 60)
print(f"  Accuracy          : {accuracy_score(y_true, y_pred):.4f}")
print(f"  F1 (PCL class)    : {f1_score(y_true, y_pred):.4f}")
print(f"  Precision (PCL)   : {precision_score(y_true, y_pred):.4f}")
print(f"  Recall (PCL)      : {recall_score(y_true, y_pred):.4f}")
print(f"  MCC               : {matthews_corrcoef(y_true, y_pred):.4f}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print()

# ── Error breakdown by PCL subcategory ────────────────────────────────────────
# For each of the 7 subcategories, show how often the model correctly predicted
# PCL when that subcategory was active — i.e. recall per subcategory.

label_matrix = np.array(dev_labels_df["label_vec"].tolist())

print("=" * 60)
print("RECALL BREAKDOWN BY PCL SUBCATEGORY")
print("(Of paragraphs with this subcategory, how many predicted as PCL?)")
print("=" * 60)

category_stats = []
for i, name in enumerate(CATEGORY_NAMES):
    mask = label_matrix[:, i] == 1
    n_total = mask.sum()
    if n_total == 0:
        continue
    n_correct = y_pred[mask].sum()
    recall = n_correct / n_total
    category_stats.append((name, n_total, n_correct, recall))
    print(f"  {name:<22s}  {n_correct:>4d}/{n_total:<4d}  recall = {recall:.4f}")

print()

# ── Plot recall by subcategory ────────────────────────────────────────────────

cat_names = [s[0] for s in category_stats]
cat_recalls = [s[3] for s in category_stats]
cat_counts = [s[1] for s in category_stats]

fig, ax1 = plt.subplots(figsize=(9, 5))
x = np.arange(len(cat_names))

bars = ax1.bar(x, cat_recalls, color="#4C72B0", alpha=0.85, label="Recall")
ax1.set_ylabel("Recall")
ax1.set_ylim(0, 1.05)
ax1.set_xticks(x)
ax1.set_xticklabels(cat_names, rotation=30, ha="right")

for bar, r in zip(bars, cat_recalls):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{r:.2f}", ha="center", va="bottom", fontsize=9)

ax2 = ax1.twinx()
ax2.plot(x, cat_counts, "o--", color="#C44E52", label="Count")
ax2.set_ylabel("# paragraphs with subcategory")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

ax1.set_title("Model Recall per PCL Subcategory")
plt.tight_layout()
plt.savefig("outputs/error_recall_by_category.png", dpi=150)
plt.close()
print("Saved: outputs/error_recall_by_category.png")

# ── False-positive & false-negative analysis ─────────────────────────────────

fp_mask = (y_true == 0) & (y_pred == 1)
fn_mask = (y_true == 1) & (y_pred == 0)

print("=" * 60)
print("FALSE NEGATIVES — PCL subcategory distribution")
print("(What types of PCL does the model miss?)")
print("=" * 60)

fn_labels = label_matrix[fn_mask]
if len(fn_labels) > 0:
    fn_category_counts = fn_labels.sum(axis=0)
    for name, count in zip(CATEGORY_NAMES, fn_category_counts):
        pct = count / fn_labels.shape[0] * 100
        print(f"  {name:<22s}  {int(count):>4d}  ({pct:.1f}%)")

print()
print("=" * 60)
print(f"SUMMARY: {fp_mask.sum()} false positives, {fn_mask.sum()} false negatives")
print("=" * 60)

# ── Number of active subcategories vs error rate ─────────────────────────────

pcl_mask = y_true == 1
n_active = label_matrix[pcl_mask].sum(axis=1)
pred_pcl = y_pred[pcl_mask]

print()
print("=" * 60)
print("RECALL BY NUMBER OF ACTIVE SUBCATEGORIES (PCL samples only)")
print("=" * 60)

for k in sorted(set(n_active)):
    sel = n_active == k
    n = sel.sum()
    correct = pred_pcl[sel].sum()
    print(f"  {k} subcategories: {correct}/{n} recalled  ({correct/n:.4f})")

# ── Sample false-negative and false-positive par_ids ─────────────────────────

dev_labels_df["pred"] = y_pred

fn_df = dev_labels_df[fn_mask].copy()
fn_df["n_subcats"] = fn_df["label_vec"].apply(sum)
fn_df["active_cats"] = fn_df["label_vec"].apply(
    lambda v: ", ".join(name for name, flag in zip(CATEGORY_NAMES, v) if flag)
)

fp_df = dev_labels_df[fp_mask].copy()

print()
print("=" * 60)
print("FALSE NEGATIVE EXAMPLES (par_ids to investigate)")
print("=" * 60)

fn_single = fn_df[fn_df["n_subcats"] == 1].head(5)
fn_multi = fn_df[fn_df["n_subcats"] >= 3].head(5)

print("\n-- FN with only 1 subcategory (subtle/borderline PCL):")
for _, row in fn_single.iterrows():
    print(f"  par_id={row['par_id']:<8d}  categories: {row['active_cats']}")

print("\n-- FN with 3+ subcategories (strong PCL the model still missed):")
for _, row in fn_multi.iterrows():
    print(f"  par_id={row['par_id']:<8d}  categories: {row['active_cats']}")

print()
print("=" * 60)
print("FALSE POSITIVE EXAMPLES (par_ids to investigate)")
print("=" * 60)
for _, row in fp_df.head(10).iterrows():
    print(f"  par_id={row['par_id']}")
