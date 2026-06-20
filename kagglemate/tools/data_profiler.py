"""Data profiler — reads CSV files and generates structured profiles.

Pure Python analysis (no LLM). Fast, reliable, deterministic.
Output is a DataProfile dict that downstream nodes consume.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from kagglemate.types import ColumnInfo, DataProfile


class DataProfiler:
    """Analyze Kaggle competition data files and produce a DataProfile."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def run(self) -> DataProfile:
        """Run all profiling steps. Returns a complete DataProfile dict."""
        train_path = self._find_file("train")
        test_path = self._find_file("test")
        sample_path = self._find_file("sample")
        if sample_path is None:
            sample_path = self._find_file("sample_submission")

        # Read CSVs
        train_df = self._read_csv(train_path) if train_path else None
        test_df = self._read_csv(test_path) if test_path else None
        sample_df = self._read_csv(sample_path) if sample_path else None

        # ---------- Build profile ----------
        profile: DataProfile = {
            "train_rows": len(train_df) if train_df is not None else 0,
            "test_rows": len(test_df) if test_df is not None else 0,
            "columns": [],
            "target_col": "",
            "id_col": "",
            "numerical_cols": [],
            "categorical_cols": [],
            "datetime_cols": [],
            "missing_values": {},
            "target_distribution": None,
            "submission_cols": [],
            "submission_rows": len(sample_df) if sample_df is not None else 0,
            "column_details": [],
        }

        if train_df is not None:
            profile["columns"] = [str(c) for c in train_df.columns]
            profile["missing_values"] = {
                str(col): float(round(train_df[col].isna().mean() * 100, 2))
                for col in train_df.columns
                if train_df[col].isna().any()
            }
            profile["numerical_cols"] = [
                str(col) for col in train_df.columns
                if pd.api.types.is_numeric_dtype(train_df[col])
            ]
            profile["datetime_cols"] = [
                str(col) for col in train_df.columns
                if pd.api.types.is_datetime64_any_dtype(train_df[col])
            ]
            profile["categorical_cols"] = [
                str(col) for col in train_df.columns
                if str(col) not in profile["numerical_cols"] + profile["datetime_cols"]
            ]

            # Column details
            for col in train_df.columns:
                col_type = str(train_df[col].dtype)
                n_missing = int(train_df[col].isna().sum())
                try:
                    n_unique = int(train_df[col].nunique())
                except Exception:
                    n_unique = -1
                profile["column_details"].append({
                    "name": str(col),
                    "dtype": col_type,
                    "n_missing": n_missing,
                    "missing_pct": float(round(n_missing / len(train_df) * 100, 2)),
                    "n_unique": n_unique,
                })

        # Heuristic: guess ID + target columns
        if train_df is not None:
            profile["id_col"] = self._guess_id_column(train_df, test_df, sample_df)
            profile["target_col"] = self._guess_target_column(train_df, sample_df)

            # Target distribution for classification
            target = profile["target_col"]
            if target and target in train_df.columns:
                value_counts = train_df[target].value_counts()
                if len(value_counts) <= 20:  # likely categorical target
                    items = [f"{k}: {v}" for k, v in value_counts.head(10).items()]
                    profile["target_distribution"] = ", ".join(items)

        # Submission format
        if sample_df is not None:
            profile["submission_cols"] = list(sample_df.columns)

        return profile

    def save_markdown(self, profile: DataProfile, output_path: str | Path) -> Path:
        """Render the profile as a markdown file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Data Profile\n",
            f"**Competition data directory:** `{self.data_dir}`\n",
            "\n## Files\n",
        ]

        for csv_file in sorted(self.data_dir.glob("*.csv")):
            df = self._read_csv(csv_file)
            if df is not None:
                lines.append(f"- `{csv_file.name}` — {len(df)} rows × {len(df.columns)} cols\n")

        lines += [
            "\n## Train Set\n",
            f"- Rows: **{profile['train_rows']}**\n",
            f"- Columns: **{len(profile['columns'])}**\n",
            "\n## Test Set\n",
            f"- Rows: **{profile['test_rows']}**\n",
            "\n## ID Column (guessed)\n",
            f"- `{profile['id_col']}`\n",
            "\n## Target Column (guessed)\n",
            f"- `{profile['target_col']}`\n",
        ]

        if profile["target_distribution"]:
            lines += ["\n## Target Distribution\n", f"{profile['target_distribution']}\n"]

        lines += ["\n## Column Details\n"]
        lines.append("| Column | Dtype | Missing | Missing % | Unique |")
        lines.append("|--------|-------|---------|-----------|--------|")
        for cd in profile.get("column_details", []):
            lines.append(
                f"| `{cd['name']}` | {cd['dtype']} | {cd['n_missing']} | "
                f"{cd['missing_pct']}% | {cd['n_unique']} |"
            )

        lines += ["\n## Submission Format\n"]
        if profile["submission_cols"]:
            lines.append(f"- Columns: `{', '.join(profile['submission_cols'])}`\n")
            lines.append(f"- Required rows: **{profile['submission_rows']}**\n")
        else:
            lines.append("- *No sample submission found*\n")

        if profile["missing_values"]:
            lines += ["\n## Missing Values (>0%)\n"]
            lines.append("| Column | Missing % |")
            lines.append("|--------|-----------|")
            for col, pct in sorted(profile["missing_values"].items(), key=lambda x: -x[1]):
                lines.append(f"| `{col}` | {pct}% |")

        out.write_text("".join(lines), encoding="utf-8")
        return out

    # ── Helpers ──

    def _find_file(self, pattern: str) -> Optional[Path]:
        """Find a CSV file whose name contains `pattern` (case-insensitive).

        Special handling for sample_submission:
        - "sample" matches "sample_submission.csv" and "gender_submission.csv"
        """
        matches = [
            p for p in self.data_dir.glob("*.csv")
            if pattern.lower() in p.name.lower()
        ]
        # For "sample", also try "gender_submission" (common Titanic-era naming)
        if not matches and pattern == "sample":
            matches = [
                p for p in self.data_dir.glob("*.csv")
                if "gender_submission" in p.name.lower() or "submission" in p.name.lower()
            ]
        return matches[0] if matches else None

    def _read_csv(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _guess_id_column(
        self,
        train: Optional[pd.DataFrame],
        test: Optional[pd.DataFrame],
        sample: Optional[pd.DataFrame],
    ) -> str:
        """Heuristic: find the ID column shared across train, test, and sample."""
        candidates = []
        for col, dt in train.dtypes.items() if train is not None else []:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["id", "passengerid", "index"]):
                candidates.append(col)
            elif col_lower == col_lower and pd.api.types.is_integer_dtype(dt):
                # Could be a plain integer ID — check if it's present in sample
                if sample is not None and col in sample.columns:
                    candidates.append(col)

        if not candidates and train is not None:
            # Fallback: first column
            candidates = [train.columns[0]]

        return candidates[0] if candidates else ""

    def _guess_target_column(
        self, train: Optional[pd.DataFrame], sample: Optional[pd.DataFrame]
    ) -> str:
        """Heuristic: target is the column in train that's NOT in sample submission.

        Usually sample_submission has (id_col, target_col), and train has
        (id_col, ..., target_col). The column in train that's also in sample
        (aside from id) is the target."""
        if train is None:
            return ""

        if sample is not None:
            sample_cols = set(sample.columns)
            # Target is typically the prediction column in sample_submission
            for col in sample.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ["survived", "target", "label",
                                                   "price", "fare", "score",
                                                   "prediction"]):
                    return col
            # Fallback: the non-ID column in sample
            non_id = [c for c in sample_cols if c != self._guess_id_column(train, None, sample)]
            if len(non_id) == 1:
                return non_id[0]

        # No sample — guess from train
        if train is not None:
            target_keywords = ["target", "label", "y", "class", "category",
                               "survived", "price", "fare", "score"]
            for col in train.columns:
                if col.lower() in target_keywords:
                    return col
            # Fallback: last column
            return train.columns[-1]

        return ""
