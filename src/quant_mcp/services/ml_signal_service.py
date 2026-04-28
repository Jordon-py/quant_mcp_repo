"""ML signal generation service for strategy enhancement.

Trains a lightweight GradientBoostedTree on lagged features to produce a
binary ml_signal column. The model is trained using a strict temporal split
to prevent lookahead bias: only bars before the split point inform the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from quant_mcp.adapters.persistence.parquet_store import ParquetStore
from quant_mcp.settings import AppSettings


class MLSignalService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.store = ParquetStore(settings.data_dir)

    def generate_ml_signal(
        self,
        dataset_id: str,
        train_fraction: float = 0.6,
        max_depth: int = 3,
        n_estimators: int = 50,
    ) -> dict:
        """Train a GBT model and write ml_signal into the feature table.

        The model predicts whether the NEXT bar's return will be positive.
        Training uses only the first `train_fraction` of the data.
        Out-of-sample predictions cover the remainder.

        Returns metadata about the model and signal quality.
        """
        path = f"features/{dataset_id}_features.parquet"
        frame = self.store.read_frame(path).copy()

        # --- Feature engineering: add momentum and structure features ---
        # These are all computed from already-lagged columns in the feature table,
        # so they don't introduce lookahead bias.
        frame["momentum_3"] = frame["close"].pct_change(3)
        frame["momentum_5"] = frame["close"].pct_change(5)
        frame["ma_spread"] = (frame["ma_fast"] - frame["ma_slow"]) / frame["ma_slow"]
        frame["price_vs_high"] = (frame["close"] - frame["rolling_high_20"]) / frame["rolling_high_20"]
        frame["vol_ratio"] = frame["volatility"] / (frame["rolling_vol_median"] + 1e-9)

        # Shift these new features by 1 bar to prevent same-bar leakage
        new_features = ["momentum_3", "momentum_5", "ma_spread", "price_vs_high", "vol_ratio"]
        frame[new_features] = frame[new_features].shift(1)

        # --- Target: next bar return > 0 ---
        # shift(-1) looks at the NEXT bar's return. This is the prediction target.
        frame["target"] = (frame["ret_1"].shift(-1) > 0).astype(int)

        # Drop rows with NaN
        frame_clean = frame.dropna(subset=new_features + ["target", "ret_1", "zscore_close", "volatility", "signal_trend_up"]).copy()

        # --- Define feature columns for the model ---
        feature_cols = [
            "ret_1",
            "zscore_close",
            "volatility",
            "signal_trend_up",
            "momentum_3",
            "momentum_5",
            "ma_spread",
            "price_vs_high",
            "vol_ratio",
        ]

        X = frame_clean[feature_cols].values
        y = frame_clean["target"].values

        # --- Temporal train/test split ---
        split_idx = int(len(X) * train_fraction)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # --- Train GradientBoosting classifier ---
        model = GradientBoostingClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # --- Generate predictions ---
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        # Predict on ALL rows (in-sample + out-of-sample)
        all_preds = model.predict(X)

        # --- Write ml_signal back into the feature table ---
        # Initialize with 0 for rows that were dropped (NaN)
        frame["ml_signal"] = 0
        frame.loc[frame_clean.index, "ml_signal"] = all_preds.astype(int)

        # Also write the new features so the engine can reference them
        # Drop the target column (it leaks future data)
        frame.drop(columns=["target"], inplace=True)

        # Save updated feature table
        self.store.write_frame(path, frame)

        # --- Feature importances ---
        importances = dict(zip(feature_cols, model.feature_importances_.tolist()))

        return {
            "dataset_id": dataset_id,
            "model_type": "GradientBoostingClassifier",
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "train_rows": split_idx,
            "test_rows": len(X) - split_idx,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "total_signals": int(all_preds.sum()),
            "signal_rate": round(float(all_preds.mean()), 4),
            "feature_importances": importances,
            "notes": (
                f"Trained on first {train_fraction*100:.0f}% of data. "
                f"OOS accuracy: {test_acc:.2%}. "
                f"ml_signal column written to feature table."
            ),
        }
