"""
Ensemble forecasting pipeline.

Compares SARIMA, Prophet, XGBoost/LightGBM, and LSTM for predicting
weekly waste tonnages by district.

Usage:
    python -m src.analysis.prediction
"""

import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.analysis.evaluation import build_comparison_table, compute_prediction_metrics
from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    set_seeds,
    setup_logging,
)

logger = setup_logging("prediction")

TEST_WEEKS = 52  # last year as holdout
LOOKBACK = 52    # input sequence length for LSTM


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame, group_col: str = "District_Code"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: everything before last TEST_WEEKS weeks is train."""
    max_date = df["week_start"].max()
    cutoff = max_date - pd.Timedelta(weeks=TEST_WEEKS)
    train = df[df["week_start"] <= cutoff].copy()
    test = df[df["week_start"] > cutoff].copy()
    logger.info(f"Train: {train['week_start'].min()} to {train['week_start'].max()} ({len(train):,} rows)")
    logger.info(f"Test:  {test['week_start'].min()} to {test['week_start'].max()} ({len(test):,} rows)")
    return train, test


# ---------------------------------------------------------------------------
# Model 1: SARIMA (per district)
# ---------------------------------------------------------------------------

def run_sarima(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Fit auto-ARIMA per district and forecast TEST_WEEKS ahead."""
    import pmdarima as pm

    logger.info("Running SARIMA forecasting...")
    predictions = []
    districts = train[group_col].unique()

    for district in tqdm(districts, desc="  SARIMA"):
        tr = train[train[group_col] == district].sort_values("week_start")
        te = test[test[group_col] == district].sort_values("week_start")

        if len(tr) < 104 or len(te) == 0:  # need at least 2 years
            continue

        y_train = tr["tons_refuse"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = pm.auto_arima(
                    y_train,
                    seasonal=True,
                    m=52,
                    max_p=3, max_q=3,
                    max_P=2, max_Q=2,
                    max_d=2, max_D=1,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    random_state=RANDOM_SEED,
                )
                fc = model.predict(n_periods=len(te))
            except Exception as e:
                logger.warning(f"  SARIMA failed for {district}: {e}")
                continue

        for i, (_, row) in enumerate(te.iterrows()):
            predictions.append({
                group_col: district,
                "week_start": row["week_start"],
                "y_true": row["tons_refuse"],
                "y_pred": fc[i] if i < len(fc) else np.nan,
                "model": "SARIMA",
            })

    return pd.DataFrame(predictions)


# ---------------------------------------------------------------------------
# Model 2: Prophet (per district)
# ---------------------------------------------------------------------------

def run_prophet(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Fit Prophet per district."""
    from prophet import Prophet

    logger.info("Running Prophet forecasting...")
    predictions = []
    districts = train[group_col].unique()

    for district in tqdm(districts, desc="  Prophet"):
        tr = train[train[group_col] == district].sort_values("week_start")
        te = test[test[group_col] == district].sort_values("week_start")

        if len(tr) < 104 or len(te) == 0:
            continue

        prophet_df = tr[["week_start", "tons_refuse"]].rename(
            columns={"week_start": "ds", "tons_refuse": "y"}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                model.fit(prophet_df)
                future = pd.DataFrame({"ds": te["week_start"]})
                fc = model.predict(future)
            except Exception as e:
                logger.warning(f"  Prophet failed for {district}: {e}")
                continue

        for i, (_, row) in enumerate(te.iterrows()):
            predictions.append({
                group_col: district,
                "week_start": row["week_start"],
                "y_true": row["tons_refuse"],
                "y_pred": fc.iloc[i]["yhat"] if i < len(fc) else np.nan,
                "model": "Prophet",
            })

    return pd.DataFrame(predictions)


# ---------------------------------------------------------------------------
# Model 3: Gradient Boosting (global model)
# ---------------------------------------------------------------------------

def _build_gb_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Create lag and temporal features for gradient boosting."""
    df = df.sort_values([group_col, "week_start"]).copy()

    # Lag features
    for lag in [1, 2, 4, 13, 26, 52]:
        df[f"lag_{lag}"] = df.groupby(group_col)["tons_refuse"].shift(lag)

    # Rolling features
    for window in [4, 13, 52]:
        df[f"roll_mean_{window}"] = df.groupby(group_col)["tons_refuse"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"roll_std_{window}"] = df.groupby(group_col)["tons_refuse"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # Temporal features
    df["month"] = df["week_start"].dt.month
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["year"] = df["week_start"].dt.year

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # District encoding
    df["district_encoded"] = pd.Categorical(df[group_col]).codes

    return df


def run_gradient_boosting(
    train: pd.DataFrame, test: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    """Global XGBoost + LightGBM models across all districts."""
    import lightgbm as lgb
    import xgboost as xgb

    logger.info("Running Gradient Boosting forecasting...")

    # Build features on combined data, then split
    combined = pd.concat([train, test], ignore_index=True)
    combined = _build_gb_features(combined, group_col)

    cutoff = train["week_start"].max()
    feat_train = combined[combined["week_start"] <= cutoff].dropna()
    feat_test = combined[combined["week_start"] > cutoff].dropna()

    feature_cols = [c for c in feat_train.columns if c.startswith(("lag_", "roll_", "district_"))]
    feature_cols += ["year", "month", "week_of_year", "month_sin", "month_cos", "week_sin", "week_cos"]
    feature_cols = [c for c in feature_cols if c in feat_train.columns]

    X_train = feat_train[feature_cols].values
    y_train = feat_train["tons_refuse"].values
    X_test = feat_test[feature_cols].values
    y_test = feat_test["tons_refuse"].values

    predictions = []

    # XGBoost
    logger.info("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)

    # Save feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "xgb_feature_importance.csv", index=False)

    for i, (_, row) in enumerate(feat_test.iterrows()):
        predictions.append({
            group_col: row[group_col],
            "week_start": row["week_start"],
            "y_true": row["tons_refuse"],
            "y_pred": xgb_pred[i],
            "model": "XGBoost",
        })

    # LightGBM
    logger.info("  Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    lgb_pred = lgb_model.predict(X_test)

    for i, (_, row) in enumerate(feat_test.iterrows()):
        predictions.append({
            group_col: row[group_col],
            "week_start": row["week_start"],
            "y_true": row["tons_refuse"],
            "y_pred": lgb_pred[i],
            "model": "LightGBM",
        })

    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(MODELS_DIR / "xgb_model.json"))
    lgb_model.booster_.save_model(str(MODELS_DIR / "lgb_model.txt"))

    return pd.DataFrame(predictions)


# ---------------------------------------------------------------------------
# Model 4: LSTM (PyTorch)
# ---------------------------------------------------------------------------

class WasteLSTM(nn.Module):
    """Simple LSTM for waste tonnage forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_lstm_sequences(
    df: pd.DataFrame, group_col: str, lookback: int = LOOKBACK
) -> tuple[np.ndarray, np.ndarray, list]:
    """Build (lookback, features) sequences for LSTM."""
    feature_cols = ["tons_refuse", "tons_paper", "tons_mgp"]

    sequences = []
    targets = []
    meta = []  # (district, week_start) for each target

    for district, grp in df.groupby(group_col):
        grp = grp.sort_values("week_start")
        vals = grp[feature_cols].values
        weeks = grp["week_start"].values

        for i in range(lookback, len(grp)):
            sequences.append(vals[i - lookback:i])
            targets.append(vals[i, 0])  # predict refuse
            meta.append((district, weeks[i]))

    return np.array(sequences), np.array(targets), meta


def run_lstm(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Train LSTM and forecast on test set."""
    logger.info("Running LSTM forecasting...")
    device = _get_device()
    logger.info(f"  Using device: {device}")

    # Build sequences
    combined = pd.concat([train, test], ignore_index=True)
    cutoff = train["week_start"].max()

    X_all, y_all, meta_all = _build_lstm_sequences(combined, group_col)

    # Split by cutoff date
    train_mask = np.array([m[1] <= cutoff for m in meta_all])
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[~train_mask], y_all[~train_mask]
    meta_test = [m for m, mask in zip(meta_all, ~train_mask) if mask]

    if len(X_train) == 0 or len(X_test) == 0:
        logger.warning("  Insufficient data for LSTM")
        return pd.DataFrame()

    # Normalize
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std

    # DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train_norm),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Model
    model = WasteLSTM(input_dim=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs}, loss={epoch_loss/len(train_loader):.4f}")

    # Predict
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_norm).to(device)
        preds_norm = model(X_test_t).cpu().numpy()

    preds = preds_norm * y_std + y_mean

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pt")

    predictions = []
    for i, (district, week) in enumerate(meta_test):
        predictions.append({
            group_col: district,
            "week_start": pd.Timestamp(week),
            "y_true": y_test[i],
            "y_pred": preds[i],
            "model": "LSTM",
        })

    return pd.DataFrame(predictions)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    set_seeds()
    logger.info("=" * 60)
    logger.info("Starting prediction pipeline")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    district_path = DATA_PROCESSED / "weekly_district.parquet"
    if not district_path.exists():
        logger.error(f"Data not found: {district_path}. Run aggregate.py first.")
        sys.exit(1)

    df = pd.read_parquet(district_path)
    logger.info(f"Loaded {len(df):,} rows, {df['District_Code'].nunique()} districts")

    # Split
    train, test = temporal_split(df)

    # Run all models
    all_predictions = []

    sarima_preds = run_sarima(train, test, "District_Code")
    all_predictions.append(sarima_preds)
    logger.info(f"SARIMA: {len(sarima_preds)} predictions")

    prophet_preds = run_prophet(train, test, "District_Code")
    all_predictions.append(prophet_preds)
    logger.info(f"Prophet: {len(prophet_preds)} predictions")

    gb_preds = run_gradient_boosting(train, test, "District_Code")
    all_predictions.append(gb_preds)
    logger.info(f"Gradient Boosting: {len(gb_preds)} predictions")

    lstm_preds = run_lstm(train, test, "District_Code")
    if len(lstm_preds) > 0:
        all_predictions.append(lstm_preds)
        logger.info(f"LSTM: {len(lstm_preds)} predictions")

    # Combine and evaluate
    results = pd.concat(all_predictions, ignore_index=True)
    results.to_parquet(RESULTS_DIR / "prediction_results.parquet", index=False)

    # Per-model metrics
    model_metrics = {}
    for model_name in results["model"].unique():
        subset = results[results["model"] == model_name]
        metrics = compute_prediction_metrics(
            subset["y_true"].values, subset["y_pred"].values
        )
        model_metrics[model_name] = metrics
        logger.info(f"\n{model_name}: {metrics}")

    comparison = build_comparison_table(model_metrics)
    comparison.to_csv(RESULTS_DIR / "prediction_comparison.csv")
    logger.info(f"\nModel comparison:\n{comparison.to_string()}")

    # Per-district metrics for best model
    best_model = comparison["rmse"].idxmin()
    logger.info(f"\nBest model by RMSE: {best_model}")

    best_preds = results[results["model"] == best_model]
    district_metrics = []
    for district in best_preds["District_Code"].unique():
        subset = best_preds[best_preds["District_Code"] == district]
        m = compute_prediction_metrics(subset["y_true"].values, subset["y_pred"].values)
        m["District_Code"] = district
        district_metrics.append(m)

    district_df = pd.DataFrame(district_metrics)
    district_df.to_csv(RESULTS_DIR / "prediction_by_district.csv", index=False)

    logger.info("Prediction pipeline complete")


if __name__ == "__main__":
    main()
