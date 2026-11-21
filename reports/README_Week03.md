# InvoiceIQ — Week 3: Feature Engineering & Dataset Preparation

## 1. Dataset & Prediction Setup
- **Source Dataset:** `data/processed/cleaned_invoices.parquet` (2,500,000 cleaned B2B invoices, `2019-01-01` to `2023-12-31`).
- **Prediction Point ($t_0$):** `invoice_date` (scoring performed at invoice issuance).
- **Classification Target (`paid_late`):** `1` if the invoice is unpaid/delinquent (`open_paid_flag == 1`) or settled after `due_date` (`delay > 0`), `0` otherwise (17.02% positive class).
- **Regression Target (`delay_days`):** Settlement offset in days (`payment_date - due_date`) for closed invoices (2,074,410 rows), `NaN` for open/unpaid invoices.

## 2. Feature Engineering Pipeline (`notebooks/Week03_Feature_Engineering.ipynb`)
1. **Time & Term Features (Task A):** `issue_month`, `issue_quarter`, `issue_weekday`, `is_month_end`, `is_quarter_end`, `payment_term_days`, `days_until_due`, and milestone flags (`is_net_15`, `is_net_30`, `is_net_45`, `is_net_60`).
2. **Leakage-Safe Customer History (Task B):** `prev_invoice_count`, `prev_avg_invoice_amount`, `prev_avg_delay`, `prev_late_rate`, and `is_first_time_customer` computed strictly from events occurring prior to `invoice_date`.
3. **Invoice Amount & Ratio Features (Task C):** `log_total_amount`, `tax_ratio`, `discount_ratio`, `shipping_ratio`, `amount_to_cust_avg_ratio`, and `amount_bucket`.
4. **Encoding & Scaling (Task D):** Frequency encoding for high-cardinality nominal columns (`customer_country`, `customer_city`, `vendor_id`, `sales_rep`), One-Hot Encoding for low-cardinality nominal attributes, and `StandardScaler` on continuous numerics—all fitted **exclusively on the training split**.
5. **Chronological Split (Task F):**
   - **Train (`2019-01-01` to `2022-12-31`):** 2,000,456 rows (80.02%)
   - **Validation (`2023-01-01` to `2023-06-30`):** 247,598 rows (9.90%)
   - **Test (`2023-07-01` to `2023-12-31`):** 251,946 rows (10.08%)

## 3. Generated Deliverables
- `data/processed/ml_train.parquet`, `ml_val.parquet`, `ml_test.parquet` (Full ML-ready splits with 85 features)
- `data/processed/ml_ready_invoices_sample.csv` (50,000-row structured CSV sample across splits)
- `reports/Week03_Feature_Catalogue.csv` & `reports/Week03_Feature_Catalogue.md`
- `reports/Week03_Leakage_Check_Note.md`
- `reports/README_Week03.md`
