"""
InvoiceIQ — Week 3: Production Feature Engineering & Chronological Split Pipeline.

Implements leakage-free temporal features, point-in-time customer behavioral
aggregates, domain amount ratios, train-only categorical/numeric preprocessing,
and automated leakage verification for 2.5M B2B invoices.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("InvoiceIQ.FeatureEngineering")


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration parameters for feature engineering and chronological splitting."""

    input_path: Path = Path("data/processed/cleaned_invoices.parquet")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports")
    artifacts_dir: Path = Path("models")

    # Chronological split boundaries (Task F)
    val_start_date: str = "2023-01-01"
    test_start_date: str = "2023-07-01"

    # Contractual milestone terms (days)
    milestone_terms: Tuple[int, ...] = (15, 30, 45, 60)

    # Amount tier boundaries
    amount_bins: Tuple[float, ...] = (-np.inf, 10_000.0, 25_000.0, 50_000.0, np.inf)
    amount_labels: Tuple[str, ...] = (
        "Small (<10k)",
        "Medium (10k-25k)",
        "Large (25k-50k)",
        "Enterprise (50k+)",
    )

    # Column groups (Task D)
    freq_encode_cols: Tuple[str, ...] = (
        "customer_country",
        "customer_city",
        "vendor_id",
        "sales_rep",
    )
    ohe_cols: Tuple[str, ...] = (
        "currency",
        "payment_method",
        "product_category",
        "order_channel",
        "amount_bucket",
    )
    numeric_scale_cols: Tuple[str, ...] = (
        "quantity",
        "unit_price",
        "discount_percent",
        "tax_percent",
        "subtotal",
        "discount_amount",
        "tax_amount",
        "shipping_cost",
        "total_amount",
        "log_total_amount",
        "tax_ratio",
        "discount_ratio",
        "shipping_ratio",
        "payment_term_days",
        "days_until_due",
        "prev_invoice_count",
        "prev_avg_invoice_amount",
        "prev_avg_delay",
        "prev_late_rate",
        "amount_to_cust_avg_ratio",
    )
    id_trace_cols: Tuple[str, ...] = (
        "invoice_id",
        "customer_id",
        "invoice_date",
        "due_date",
    )
    target_cols: Tuple[str, ...] = ("paid_late", "delay_days")

    # Strictly forbidden post-invoice columns (Task G)
    forbidden_cols: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "payment_date",
                "payment_status",
                "is_overdue",
                "open_paid_flag",
                "days_to_pay",
                "delay",
                "paid_late_flag",
            }
        )
    )


class InvoiceFeatureEngineer:
    """Leakage-safe feature engineering and train-only transformation pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.train_medians: Dict[str, float] = {}
        self.freq_maps: Dict[str, Dict[object, float]] = {}
        self.ohe = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.float32
        )
        self.scaler = StandardScaler()
        self.ohe_feature_names: List[str] = []
        self.freq_feature_names: List[str] = []
        self.scaled_feature_names: List[str] = []
        self.feature_cols: List[str] = []

    def load_data(self) -> pd.DataFrame:
        """Load cleaned parquet dataset and sort chronologically by t0 = invoice_date."""
        logger.info("Loading cleaned dataset from %s", self.config.input_path)
        df = pd.read_parquet(self.config.input_path)
        for col in ("invoice_date", "due_date", "payment_date"):
            df[col] = pd.to_datetime(df[col])

        df = df.sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True)
        logger.info(
            "Loaded %s rows (%s to %s)",
            f"{len(df):,}",
            df["invoice_date"].min().date(),
            df["invoice_date"].max().date(),
        )
        return df

    @staticmethod
    def define_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Define classification (paid_late) and regression (delay_days) targets (Task E)."""
        df["paid_late"] = np.where(
            (df["open_paid_flag"] == 1) | (df["delay"] > 0), 1, 0
        ).astype(np.int8)
        df["delay_days"] = np.where(
            df["open_paid_flag"] == 0, df["delay"], np.nan
        ).astype(np.float32)
        logger.info(
            "Targets defined | paid_late rate: %.2f%% | delay_days non-null: %s",
            df["paid_late"].mean() * 100,
            f"{df['delay_days'].notna().sum():,}",
        )
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer calendar and contractual credit window features at t0 (Task A)."""
        df["issue_month"] = df["invoice_date"].dt.month.astype(np.int8)
        df["issue_quarter"] = df["invoice_date"].dt.quarter.astype(np.int8)
        df["issue_weekday"] = df["invoice_date"].dt.dayofweek.astype(np.int8)
        df["is_month_end"] = df["invoice_date"].dt.is_month_end.astype(np.int8)
        df["is_quarter_end"] = df["invoice_date"].dt.is_quarter_end.astype(np.int8)

        term_days = (df["due_date"] - df["invoice_date"]).dt.days.astype(np.int16)
        df["payment_term_days"] = term_days
        df["days_until_due"] = term_days

        for milestone in self.config.milestone_terms:
            df[f"is_net_{milestone}"] = (df["payment_term_days"] == milestone).astype(
                np.int8
            )
        logger.info("Task A complete: Engineered time and payment term features.")
        return df

    @staticmethod
    def add_customer_history_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute strictly past-only customer aggregates prior to t0 (Task B)."""
        cust_col = "customer_id"

        # 1. Prior invoice count & spend (strictly prior_invoice_date < t0)
        daily_issued = (
            df.groupby([cust_col, "invoice_date"], as_index=False)
            .agg(
                day_inv_count=("invoice_id", "count"),
                day_inv_sum=("total_amount", "sum"),
            )
            .sort_values([cust_col, "invoice_date"])
        )
        daily_issued["prev_invoice_count"] = (
            daily_issued.groupby(cust_col)["day_inv_count"].cumsum()
            - daily_issued["day_inv_count"]
        ).astype(np.int32)
        daily_issued["prev_amount_sum"] = (
            daily_issued.groupby(cust_col)["day_inv_sum"].cumsum()
            - daily_issued["day_inv_sum"]
        )
        daily_issued["prev_avg_invoice_amount"] = np.where(
            daily_issued["prev_invoice_count"] > 0,
            daily_issued["prev_amount_sum"] / daily_issued["prev_invoice_count"],
            np.nan,
        ).astype(np.float32)

        df = df.merge(
            daily_issued[
                [
                    cust_col,
                    "invoice_date",
                    "prev_invoice_count",
                    "prev_avg_invoice_amount",
                ]
            ],
            on=[cust_col, "invoice_date"],
            how="left",
        )
        del daily_issued
        gc.collect()

        # 2. Prior average delay (strictly payment_date < t0)
        closed_df = df.loc[
            df["open_paid_flag"] == 0, [cust_col, "payment_date", "delay"]
        ]
        daily_settled = (
            closed_df.groupby([cust_col, "payment_date"], as_index=False)
            .agg(day_settled_cnt=("delay", "count"), day_delay_sum=("delay", "sum"))
            .sort_values([cust_col, "payment_date"])
        )
        daily_settled["cum_settled_cnt"] = daily_settled.groupby(cust_col)[
            "day_settled_cnt"
        ].cumsum()
        daily_settled["cum_delay_sum"] = daily_settled.groupby(cust_col)[
            "day_delay_sum"
        ].cumsum()

        df = pd.merge_asof(
            df.sort_values("invoice_date"),
            daily_settled[
                [cust_col, "payment_date", "cum_settled_cnt", "cum_delay_sum"]
            ].sort_values("payment_date"),
            left_on="invoice_date",
            right_on="payment_date",
            by=cust_col,
            direction="backward",
            allow_exact_matches=False,
        )
        df["prev_avg_delay"] = np.where(
            df["cum_settled_cnt"].fillna(0) > 0,
            df["cum_delay_sum"] / df["cum_settled_cnt"],
            np.nan,
        ).astype(np.float32)
        df.drop(
            columns=["payment_date_y", "cum_settled_cnt", "cum_delay_sum"],
            errors="ignore",
            inplace=True,
        )
        if "payment_date_x" in df.columns:
            df.rename(columns={"payment_date_x": "payment_date"}, inplace=True)
        del closed_df, daily_settled
        gc.collect()

        # 3. Prior late rate (strictly realized outcome_date < t0)
        df["_outcome_date"] = pd.to_datetime(
            np.where(df["open_paid_flag"] == 0, df["payment_date"], df["due_date"])
        )
        daily_outcome = (
            df.groupby([cust_col, "_outcome_date"], as_index=False)
            .agg(day_out_cnt=("paid_late", "count"), day_late_sum=("paid_late", "sum"))
            .sort_values("_outcome_date")
        )
        daily_outcome["cum_out_cnt"] = daily_outcome.groupby(cust_col)[
            "day_out_cnt"
        ].cumsum()
        daily_outcome["cum_late_sum"] = daily_outcome.groupby(cust_col)[
            "day_late_sum"
        ].cumsum()

        df = pd.merge_asof(
            df.sort_values("invoice_date"),
            daily_outcome[
                [cust_col, "_outcome_date", "cum_out_cnt", "cum_late_sum"]
            ].sort_values("_outcome_date"),
            left_on="invoice_date",
            right_on="_outcome_date",
            by=cust_col,
            direction="backward",
            allow_exact_matches=False,
        )
        df["prev_late_rate"] = np.where(
            df["cum_out_cnt"].fillna(0) > 0,
            df["cum_late_sum"] / df["cum_out_cnt"],
            np.nan,
        ).astype(np.float32)
        df["is_first_time_customer"] = (df["prev_invoice_count"] == 0).astype(np.int8)

        df.drop(
            columns=[
                "_outcome_date",
                "_outcome_date_x",
                "_outcome_date_y",
                "cum_out_cnt",
                "cum_late_sum",
            ],
            errors="ignore",
            inplace=True,
        )
        df = df.sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True)
        del daily_outcome
        gc.collect()
        logger.info("Task B complete: Engineered point-in-time customer history.")
        return df

    def add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer log transforms, component ratios, and amount tiers (Task C)."""
        df["log_total_amount"] = np.log1p(
            np.clip(df["total_amount"], a_min=0.0, a_max=None)
        ).astype(np.float32)
        df["tax_ratio"] = np.where(
            df["total_amount"] > 0, df["tax_amount"] / df["total_amount"], 0.0
        ).astype(np.float32)
        df["discount_ratio"] = np.where(
            df["subtotal"] > 0, df["discount_amount"] / df["subtotal"], 0.0
        ).astype(np.float32)
        df["shipping_ratio"] = np.where(
            df["total_amount"] > 0, df["shipping_cost"] / df["total_amount"], 0.0
        ).astype(np.float32)

        df["amount_to_cust_avg_ratio"] = np.where(
            df["prev_avg_invoice_amount"].notna()
            & (df["prev_avg_invoice_amount"] > 0),
            df["total_amount"] / df["prev_avg_invoice_amount"],
            1.0,
        ).astype(np.float32)

        df["amount_bucket"] = pd.cut(
            df["total_amount"],
            bins=list(self.config.amount_bins),
            labels=list(self.config.amount_labels),
        ).astype(str)
        logger.info("Task C complete: Engineered invoice amount and ratio features.")
        return df

    def split_chronologically(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset chronologically into Train, Validation, and Test (Task F)."""
        train_df = df.loc[df["invoice_date"] < self.config.val_start_date].copy()
        val_df = df.loc[
            (df["invoice_date"] >= self.config.val_start_date)
            & (df["invoice_date"] < self.config.test_start_date)
        ].copy()
        test_df = df.loc[df["invoice_date"] >= self.config.test_start_date].copy()

        for label, part in (("Train", train_df), ("Val", val_df), ("Test", test_df)):
            logger.info(
                "Split %-5s | %9s rows (%5.2f%%) | %s -> %s | Late Rate: %.2f%%",
                label,
                f"{len(part):,}",
                len(part) / len(df) * 100,
                part["invoice_date"].min().date(),
                part["invoice_date"].max().date(),
                part["paid_late"].mean() * 100,
            )
        return train_df, val_df, test_df

    def fit_transform_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit imputers, encoders, and scalers on Train only; transform all splits (Task D)."""
        hist_cols = ["prev_avg_invoice_amount", "prev_avg_delay", "prev_late_rate"]
        self.train_medians = {col: float(train_df[col].median()) for col in hist_cols}

        for part in (train_df, val_df, test_df):
            for col, med in self.train_medians.items():
                part[col] = part[col].fillna(med).astype(np.float32)

        # 1. Frequency encoding (Train-only)
        active_freq_cols = [
            c for c in self.config.freq_encode_cols if c in train_df.columns
        ]
        self.freq_feature_names = []
        for col in active_freq_cols:
            self.freq_maps[col] = train_df[col].value_counts(normalize=True).to_dict()
            new_col = f"{col}_freq_enc"
            self.freq_feature_names.append(new_col)
            for part in (train_df, val_df, test_df):
                part[new_col] = (
                    part[col].map(self.freq_maps[col]).fillna(0.0).astype(np.float32)
                )

        # 2. One-Hot Encoding (Train-only)
        active_ohe_cols = [c for c in self.config.ohe_cols if c in train_df.columns]
        self.ohe.fit(train_df[active_ohe_cols])
        self.ohe_feature_names = list(self.ohe.get_feature_names_out(active_ohe_cols))

        for part in (train_df, val_df, test_df):
            encoded = self.ohe.transform(part[active_ohe_cols])
            for idx, col_name in enumerate(self.ohe_feature_names):
                part[col_name] = encoded[:, idx]

        # 3. StandardScaler (Train-only)
        active_num_cols = [
            c for c in self.config.numeric_scale_cols if c in train_df.columns
        ]
        self.scaler.fit(train_df[active_num_cols])
        self.scaled_feature_names = [f"{c}_scaled" for c in active_num_cols]

        for part in (train_df, val_df, test_df):
            scaled = self.scaler.transform(part[active_num_cols]).astype(np.float32)
            for idx, col_name in enumerate(self.scaled_feature_names):
                part[col_name] = scaled[:, idx]

        time_binary_cols = [
            "issue_month",
            "issue_quarter",
            "issue_weekday",
            "is_month_end",
            "is_quarter_end",
            "is_first_time_customer",
            *[f"is_net_{m}" for m in self.config.milestone_terms],
        ]
        self.feature_cols = (
            time_binary_cols
            + active_num_cols
            + self.scaled_feature_names
            + self.freq_feature_names
            + self.ohe_feature_names
        )
        logger.info(
            "Task D complete: Fitted encoders/scalers on Train (%d input features).",
            len(self.feature_cols),
        )
        return train_df, val_df, test_df

    def validate_no_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Run strict automated leakage, chronological boundary, and null assertions (Task G)."""
        forbidden = self.config.forbidden_cols.union(self.config.target_cols)
        overlap = set(self.feature_cols).intersection(forbidden)
        if overlap:
            raise ValueError(f"DATA LEAKAGE DETECTED: {overlap}")

        if not (train_df["invoice_date"].max() < val_df["invoice_date"].min()):
            raise ValueError("Chronological boundary violated between Train and Val.")
        if not (val_df["invoice_date"].max() < test_df["invoice_date"].min()):
            raise ValueError("Chronological boundary violated between Val and Test.")

        for name, part in (("Train", train_df), ("Val", val_df), ("Test", test_df)):
            null_count = int(part[self.feature_cols].isna().sum().sum())
            if null_count > 0:
                raise ValueError(f"Null features detected in {name}: {null_count}")

        logger.info(
            "Task G Leakage & Integrity Audit PASSED (0 leaked columns, 0 nulls)."
        )

    def save_artifacts_and_deliverables(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Persist ML-ready splits, fitted preprocessor artifact, and documentation."""
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        final_cols = (
            list(self.config.id_trace_cols)
            + self.feature_cols
            + list(self.config.target_cols)
        )

        # 1. Parquet datasets & CSV sample
        train_df[final_cols].to_parquet(
            self.config.processed_dir / "ml_train.parquet", index=False
        )
        val_df[final_cols].to_parquet(
            self.config.processed_dir / "ml_val.parquet", index=False
        )
        test_df[final_cols].to_parquet(
            self.config.processed_dir / "ml_test.parquet", index=False
        )

        sample_df = pd.concat(
            [
                train_df[final_cols].tail(30_000),
                val_df[final_cols].head(10_000),
                test_df[final_cols].head(10_000),
            ],
            ignore_index=True,
        )
        sample_df.to_csv(
            self.config.processed_dir / "ml_ready_invoices_sample.csv", index=False
        )

        # 2. Fitted preprocessor artifact for Week 4/5 & FastAPI deployment
        joblib.dump(
            {
                "train_medians": self.train_medians,
                "freq_maps": self.freq_maps,
                "ohe": self.ohe,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
            },
            self.config.artifacts_dir / "feature_preprocessor.joblib",
        )
        logger.info("Saved ML datasets and fitted preprocessor artifact.")

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the full end-to-end Week 3 feature engineering pipeline."""
        df = self.load_data()
        df = self.define_targets(df)
        df = self.add_time_features(df)
        df = self.add_customer_history_features(df)
        df = self.add_amount_features(df)

        train_df, val_df, test_df = self.split_chronologically(df)
        del df
        gc.collect()

        train_df, val_df, test_df = self.fit_transform_splits(train_df, val_df, test_df)
        self.validate_no_leakage(train_df, val_df, test_df)
        self.save_artifacts_and_deliverables(train_df, val_df, test_df)
        return train_df, val_df, test_df


if __name__ == "__main__":
    InvoiceFeatureEngineer().run()