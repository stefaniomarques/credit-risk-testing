# Credit Risk: PD Modelling, Expected Loss & Stress Testing

A probability-of-default (PD) model, IFRS 9-style Expected Loss calculation,
and Basel-consistent macroeconomic stress test, built on the UCI Statlog
German Credit dataset (1,000 loan applicants).

## What This Demonstrates

- **PD modelling** — a logistic regression credit scoring model, evaluated
  on discrimination (AUC) *and* calibration (does a predicted 30% PD really
  correspond to a ~30% observed default rate?)
- **Expected Loss** — applying the standard IFRS 9 / Basel identity
  `EL = PD x LGD x EAD` at portfolio level
- **Stress testing** — translating a macroeconomic shock into a stressed PD
  using the Basel Asymptotic Single Risk Factor (ASRF) model, the same
  functional form underlying Basel IRB capital requirements
- **Segmentation** — checking whether the model's risk ranking holds up
  across loan-purpose segments, not just in aggregate

## Key Results

| Scenario | Shock (Z) | Avg. Portfolio PD | Expected Loss | EL Rate |
|---|---|---|---|---|
| Baseline | 0.0 | 29.9% | DM 525,647 | 16.1% |
| Moderate Stress | -1.5 | 39.6% | DM 672,426 | 20.6% |
| Severe Stress | -3.0 | 50.4% | DM 827,376 | 25.3% |

Portfolio Expected Loss rises **57%** moving from baseline to a severe stress
scenario — driven entirely by the systematic macro factor, holding every
borrower's individual characteristics fixed.

PD model: **AUC 0.804**, calibrated average predicted PD (29.9%) matching the
sample's actual default rate (30.0%) closely.

## Methodology Notes (read before drawing conclusions)

This is a portfolio/methodology demonstration, not a production credit model
— a few things are worth being explicit about:

- **The 30% default rate is not realistic.** Real retail loan portfolios
  typically see default rates in the low single digits. This dataset was
  deliberately constructed by researchers with a much higher (and roughly
  3:1) good:bad ratio for classroom/research use, so absolute Expected Loss
  figures here should be read as illustrative of the *methodology*, not as
  real-world loss estimates.
- **LGD (45%) and EAD (= disbursed amount) are Basel supervisory
  assumptions, not fitted values.** The dataset has no recovery or exposure
  data, so both are stated conventions rather than estimated parameters.
- **The asset correlation (ρ = 0.06) is an illustrative parameter**, in line
  with the Basel retail unsecured range, not calibrated to this specific
  portfolio.
- **Calibration was prioritised over raw accuracy.** An earlier version of
  this model used class-balanced weighting, which improved classification
  metrics but shifted predicted PDs away from true likelihoods. Since PD
  feeds directly into Expected Loss and capital calculations downstream,
  the final model uses natural class weights instead — a real trade-off any
  PD model has to make explicitly.

## Repository Structure

```
credit-risk-pd-stress-testing/
├── README.md
├── analysis/
│   └── credit_risk_model.py   # full pipeline: data prep -> PD model -> EL -> stress test -> segmentation
├── data/
│   ├── german_credit_raw.txt
│   └── README.md               # data dictionary and source
└── output/
    ├── model_performance.md
    ├── stress_test_results.csv
    └── segment_pd_by_purpose.csv
```

## Running It

```bash
pip install pandas numpy scipy scikit-learn
cd analysis
python credit_risk_model.py
```

## Tools

Python — pandas, numpy, scipy, scikit-learn
