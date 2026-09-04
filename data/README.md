# Data Source

**UCI Statlog (German Credit Data)**, donated by Prof. Hans Hofmann,
Universität Hamburg (1994).

- Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- License: CC BY 4.0
- 1,000 loan applicants, 20 attributes, classified as good/bad credit risk

## Attributes Used

| Column | Description |
|---|---|
| checking_account_status | Status of existing checking account |
| duration_months | Loan duration |
| credit_history | Credit history / repayment record |
| purpose | Loan purpose (car, education, business, etc.) |
| credit_amount | Loan amount (DM) — used here as EAD proxy |
| savings_account | Savings account / bonds balance |
| employment_since | Length of current employment |
| installment_rate_pct | Installment rate as % of disposable income |
| personal_status_sex | Personal status and sex |
| other_debtors | Other debtors / guarantors |
| residence_since | Years at present residence |
| property | Property ownership |
| age_years | Applicant age |
| other_installment_plans | Other installment plans (bank/store/none) |
| housing | Housing situation (rent/own/free) |
| existing_credits_count | Number of existing credits at this bank |
| job | Job / employment skill level |
| num_dependents | Number of dependents |
| telephone | Telephone registered (yes/no) |
| foreign_worker | Foreign worker status |
| **target → default** | 1 = good credit, 2 = bad credit (recoded to 0/1 default flag) |

## Note on Realism

This dataset predates IFRS 9 and does not include Loss Given Default (LGD)
or Exposure at Default (EAD) fields directly — both are approximated in this
project using Basel supervisory conventions (see root README). The default
rate (30%) is also far higher than a real retail portfolio would show,
reflecting the dataset's research/teaching origin rather than a live book of
loans.
