# InvoiceIQ — Week 3 Data Leakage Audit & Verification Note

## 1. Prediction Point Definition
All model predictions are executed at **t0 = `invoice_date`** (the exact moment an invoice is generated and issued). Every input feature is strictly restricted to information observable on or before `t0`.

## 2. Excluded Post-Invoice & Outcome Columns
The following columns depend on events occurring **after `invoice_date`** and were strictly excluded from the input feature matrix (`feature_cols`):
- `payment_date`: Actual settlement timestamp (future event; null for open invoices).
- `payment_status`: Post-issuance lifecycle state (`Paid`, `Pending`, `Disputed`).
- `is_overdue`: Post-due-date delinquency indicator.
- `open_paid_flag`: Post-issuance settlement indicator.
- `days_to_pay`: Derived directly from `payment_date - invoice_date`.
- `delay`: Derived directly from `payment_date - due_date`.
- `paid_late_flag`: Historical outcome indicator derived from `delay`.
- `paid_late` & `delay_days`: Prediction target variables.

## 3. Point-in-Time Historical Aggregate Verification
- **Daily Rollup Shift (`prev_invoice_count`, `prev_avg_invoice_amount`):** Aggregated by `(customer_id, invoice_date)` and subtracted the current day's total (`cumsum - day_total`), ensuring invoices issued on the same day never leak into one another.
- **Outcome-Date Merge (`prev_avg_delay`, `prev_late_rate`):** Merged via `pd.merge_asof(..., direction='backward', allow_exact_matches=False)` matching `invoice_date` against `payment_date` (for closed bills) and `_outcome_date` (`payment_date` if closed, `due_date` if unpaid). Outcomes realized on or after `t0` are never visible.

## 4. Train-Only Preprocessing & Chronological Split Verification
- **Train:** `2019-01-01` to `2022-12-31` (2,000,456 rows, 80.02%)
- **Validation:** `2023-01-01` to `2023-06-30` (247,598 rows, 9.90%)
- **Test:** `2023-07-01` to `2023-12-31` (251,946 rows, 10.08%)
- Cold-start medians, frequency encodings, `OneHotEncoder`, and `StandardScaler` were fitted **exclusively on `train_df`** and applied read-only to `val_df` and `test_df`.
- **Automated Assertion:** `len(set(feature_cols).intersection(forbidden_cols)) == 0` across all **85** input features — **PASSED** (0 nulls across Train, Validation, and Test).
