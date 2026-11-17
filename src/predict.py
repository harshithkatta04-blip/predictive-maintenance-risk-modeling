import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Reference wear limits used in training (engineering assumptions)
TYPE_WEAR_LIMITS = {"L": 180.0, "M": 200.0, "H": 220.0}
NOMINAL_POWER_W = 6250.0  # Midpoint of [3500 W, 9000 W] operating band


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Replicate Phase 1 domain feature engineering and column sanitization."""
    df = df_raw.copy()

    # Core physical interactions
    df["Power [W]"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60.0)
    df["Temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power_deviation [W]"] = np.abs(df["Power [W]"] - NOMINAL_POWER_W)
    df["Thermal_speed_risk [K/rpm]"] = df["Temp_diff [K]"] / df["Rotational speed [rpm]"]
    df["Wear_torque_load [Nm*min]"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["Torque_speed_ratio [Nm/rpm]"] = df["Torque [Nm]"] / df["Rotational speed [rpm]"]
    df["Type_wear_limit_ratio"] = df["Tool wear [min]"] / df["Type"].map(TYPE_WEAR_LIMITS)

    # Sanitize column names to match training pipeline
    df.columns = [re.sub(r"[\[\]<]", "", str(c)).strip().replace(" ", "_") for c in df.columns]
    return df


def predict_machine_failure(
    df_sensor: pd.DataFrame,
    artifact_path: str | Path = "models/gradient_boosting_champion.joblib",
) -> pd.DataFrame:
    """Score raw telemetry rows and return failure probabilities, alerts, and risk tiers."""
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    threshold = artifact["threshold"]
    expected_features = artifact["features"]

    X_feat = engineer_features(df_sensor)[expected_features]
    probs = model.predict_proba(X_feat)[:, 1]
    preds = (probs >= threshold).astype(int)

    results = df_sensor.copy()
    results["Failure_Probability"] = np.round(probs, 4)
    results["Failure_Alert"] = preds
    results["Risk_Tier"] = np.select(
        [probs >= threshold, probs >= 0.20],
        ["CRITICAL (Inspect Immediately)", "WATCHLIST (Elevated Thermal/Wear Risk)"],
        default="NORMAL",
    )
    return results


if __name__ == "__main__":
    # Quick smoke test on sample telemetry rows
    sample_telemetry = pd.DataFrame([
        {
            "Type": "M",
            "Air temperature [K]": 298.5,
            "Process temperature [K]": 309.0,
            "Rotational speed [rpm]": 1520,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 85,
        },
        {
            "Type": "H",
            "Air temperature [K]": 302.3,
            "Process temperature [K]": 310.9,
            "Rotational speed [rpm]": 1366,
            "Torque [Nm]": 48.4,
            "Tool wear [min]": 130,
        },
        {
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1250,
            "Torque [Nm]": 68.0,
            "Tool wear [min]": 215,
        },
    ])

    scored = predict_machine_failure(sample_telemetry)
    print(scored[["Type", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                  "Failure_Probability", "Failure_Alert", "Risk_Tier"]].to_string(index=False))