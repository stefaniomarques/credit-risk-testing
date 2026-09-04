"""
Credit Risk: PD Modelling, Expected Loss, and Macroeconomic Stress Testing
============================================================================
Data: UCI Statlog (German Credit Data), Hofmann (1994), CC BY 4.0.
1,000 loan applicants, classified as good/bad credit risk.

This script:
  1. Loads and decodes the raw data (coded categorical attributes -> labels)
  2. Fits a Probability of Default (PD) model (logistic regression)
  3. Calculates Expected Loss (EL = PD x LGD x EAD) under Basel/IFRS 9 conventions
  4. Applies the Basel Asymptotic Single Risk Factor (ASRF) formula to stress
     PD under macroeconomic scenarios (the same formula underlying Basel IRB
     capital requirements and industry stress-testing practice)
  5. Segments PD by loan purpose to check model discrimination across segments
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

pd.set_option("display.width", 120)

# -----------------------------------------------------------------------
# 1. LOAD AND DECODE
# -----------------------------------------------------------------------
COLUMNS = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate_pct",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age_years", "other_installment_plans", "housing", "existing_credits_count",
    "job", "num_dependents", "telephone", "foreign_worker", "target",
]

df = pd.read_csv("../data/german_credit_raw.txt", sep=" ", header=None, names=COLUMNS)

# target: 1 = good credit, 2 = bad credit in the source data -> convert to default flag
df["default"] = (df["target"] == 2).astype(int)
df = df.drop(columns=["target"])

PURPOSE_MAP = {
    "A40": "car_new", "A41": "car_used", "A42": "furniture_equipment",
    "A43": "radio_tv", "A44": "domestic_appliances", "A45": "repairs",
    "A46": "education", "A47": "vacation", "A48": "retraining",
    "A49": "business", "A410": "other",
}
df["purpose"] = df["purpose"].map(PURPOSE_MAP).fillna(df["purpose"])

EMPLOYMENT_MAP = {
    "A71": "unemployed", "A72": "lt_1yr", "A73": "1_to_4yr",
    "A74": "4_to_7yr", "A75": "gte_7yr",
}
df["employment_since"] = df["employment_since"].map(EMPLOYMENT_MAP).fillna(df["employment_since"])

print(f"Loaded {len(df)} loan applicants, {df['default'].mean():.1%} baseline default rate\n")

# -----------------------------------------------------------------------
# 2. PD MODEL (Probability of Default)
# -----------------------------------------------------------------------
categorical_cols = [
    "checking_account_status", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status_sex", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker",
]
numeric_cols = [
    "duration_months", "credit_amount", "installment_rate_pct", "residence_since",
    "age_years", "existing_credits_count", "num_dependents",
]

X = df[categorical_cols + numeric_cols]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

pd_model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000)),
])
# Note: class_weight="balanced" is deliberately NOT used here. It would
# improve raw classification accuracy on this artificially balanced dataset,
# but it distorts predicted probabilities away from true likelihoods --
# and for a PD model, calibration (does a 20% PD really mean ~20% of such
# borrowers default?) matters more than discrimination alone, since PD
# feeds directly into Expected Loss and capital calculations downstream.

pd_model.fit(X_train, y_train)

pred_proba_test = pd_model.predict_proba(X_test)[:, 1]
pred_class_test = pd_model.predict(X_test)

auc = roc_auc_score(y_test, pred_proba_test)
acc = accuracy_score(y_test, pred_class_test)
cm = confusion_matrix(y_test, pred_class_test)

print("=" * 60)
print("PD MODEL PERFORMANCE (held-out test set, n=%d)" % len(y_test))
print("=" * 60)
print(f"AUC-ROC:  {auc:.3f}")
print(f"Accuracy: {acc:.3f}")
print("Confusion matrix (rows=actual, cols=predicted) [0=Good, 1=Default]:")
print(cm)
print()
print(classification_report(y_test, pred_class_test, target_names=["Good", "Default"]))

# -----------------------------------------------------------------------
# 3. EXPECTED LOSS (IFRS 9 / Basel convention: EL = PD x LGD x EAD)
# -----------------------------------------------------------------------
# LGD and EAD are not present in this dataset (it predates IFRS 9 reporting).
# We apply standard Basel supervisory assumptions for illustration:
#   - LGD: 45% is the Basel Foundation-IRB supervisory LGD for senior
#     unsecured retail/corporate exposures with no eligible collateral.
#   - EAD: approximated here as the disbursed credit_amount (no undrawn
#     revolving facilities in this product type, so EAD = drawn balance).
LGD_ASSUMPTION = 0.45

df["pd_baseline"] = pd_model.predict_proba(X)[:, 1]
df["ead"] = df["credit_amount"]
df["expected_loss_baseline"] = df["pd_baseline"] * LGD_ASSUMPTION * df["ead"]

portfolio_ead = df["ead"].sum()
portfolio_el_baseline = df["expected_loss_baseline"].sum()

print("=" * 60)
print("PORTFOLIO EXPECTED LOSS (baseline, LGD assumption = %.0f%%)" % (LGD_ASSUMPTION * 100))
print("=" * 60)
print(f"Total EAD (portfolio exposure): DM {portfolio_ead:,.0f}")
print(f"Total Expected Loss (baseline): DM {portfolio_el_baseline:,.0f}")
print(f"Portfolio EL rate: {portfolio_el_baseline / portfolio_ead:.2%}")
print()

# -----------------------------------------------------------------------
# 4. MACROECONOMIC STRESS TESTING (Basel ASRF / Vasicek single-factor model)
# -----------------------------------------------------------------------
# Stressed PD = Phi[ (Phi^-1(PD_baseline) - sqrt(rho) * Z) / sqrt(1 - rho) ]
#   rho = asset correlation (Basel retail unsecured range: ~0.03-0.16;
#         0.06 used here as a representative illustrative value)
#   Z   = systematic macro factor (standard normal). Z < 0 = adverse shock.
# This is the same functional form underlying Basel IRB capital requirements
# and is a standard simplified technique for translating a macro scenario
# into a portfolio-wide PD shift for stress testing purposes.
RHO = 0.06

SCENARIOS = {
    "Baseline":        0.0,
    "Moderate Stress": -1.5,   # ~ a 1-in-15-year downturn
    "Severe Stress":   -3.0,   # ~ a 1-in-740-year downturn (severe recession)
}

def stressed_pd(pd_baseline, rho, z):
    pd_clipped = np.clip(pd_baseline, 1e-6, 1 - 1e-6)
    return norm.cdf((norm.ppf(pd_clipped) - np.sqrt(rho) * z) / np.sqrt(1 - rho))

print("=" * 60)
print("STRESS TEST RESULTS (Basel ASRF model, rho = %.2f)" % RHO)
print("=" * 60)

stress_results = []
for name, z in SCENARIOS.items():
    df[f"pd_{name.lower().replace(' ', '_')}"] = stressed_pd(df["pd_baseline"], RHO, z)
    el = (df[f"pd_{name.lower().replace(' ', '_')}"] * LGD_ASSUMPTION * df["ead"]).sum()
    avg_pd = df[f"pd_{name.lower().replace(' ', '_')}"].mean()
    stress_results.append({
        "scenario": name, "shock_z": z, "avg_portfolio_pd": avg_pd,
        "portfolio_el": el, "el_rate": el / portfolio_ead,
    })
    print(f"{name:18s} (Z={z:+.1f}): avg PD = {avg_pd:6.2%}   "
          f"Portfolio EL = DM {el:>12,.0f}   EL rate = {el/portfolio_ead:.2%}")

stress_df = pd.DataFrame(stress_results)
print()
print(f"EL increase, Baseline -> Severe Stress: "
      f"{(stress_df.iloc[2]['portfolio_el'] / stress_df.iloc[0]['portfolio_el'] - 1):.1%}")
print()

# -----------------------------------------------------------------------
# 5. SEGMENTATION: PD by loan purpose
# -----------------------------------------------------------------------
print("=" * 60)
print("SEGMENTATION: Average baseline PD by loan purpose")
print("=" * 60)
segment_pd = (
    df.groupby("purpose")
    .agg(n=("pd_baseline", "size"), avg_pd=("pd_baseline", "mean"), avg_ead=("ead", "mean"))
    .sort_values("avg_pd", ascending=False)
)
print(segment_pd.round(3))

# -----------------------------------------------------------------------
# SAVE OUTPUTS
# -----------------------------------------------------------------------
with open("../output/model_performance.md", "w") as f:
    f.write("# PD Model Performance\n\n")
    f.write(f"- **AUC-ROC:** {auc:.3f}\n")
    f.write(f"- **Accuracy:** {acc:.3f}\n")
    f.write(f"- Test set size: {len(y_test)}\n\n")
    f.write("## Confusion Matrix\n\n")
    f.write("| | Predicted Good | Predicted Default |\n|---|---|---|\n")
    f.write(f"| **Actual Good** | {cm[0][0]} | {cm[0][1]} |\n")
    f.write(f"| **Actual Default** | {cm[1][0]} | {cm[1][1]} |\n")

stress_df.to_csv("../output/stress_test_results.csv", index=False)
segment_pd.to_csv("../output/segment_pd_by_purpose.csv")

print("\nSaved: output/model_performance.md, stress_test_results.csv, segment_pd_by_purpose.csv")
