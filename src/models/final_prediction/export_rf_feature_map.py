# src/models/final_prediction/export_rf_feature_map.py
from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

from src.config import MODEL_DIR, SHAP_DIR  # ✅ 关键：用项目统一配置

MODEL_PATH = MODEL_DIR / "avms" / "rf_final_model" / "rf_final_model.joblib"
OUT_PATH = SHAP_DIR / "feature_name_map.csv" 


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RF model not found.\n"
            f"Expected path:\n  {MODEL_PATH}\n\n"
            f"Please check the exact filename inside rf_final_model directory."
        )

    model = joblib.load(MODEL_PATH)

    # =================================================
    # Recover feature names (critical)
    # =================================================
    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "model.feature_names_in_ not found.\n"
            "This usually means the model was trained on a numpy array\n"
            "instead of a pandas DataFrame.\n"
            "In that case, feature-name recovery is impossible."
        )

    feature_names = list(model.feature_names_in_)
    n_features = len(feature_names)

    df_map = pd.DataFrame(
        {
            "feature": [f"f_{i}" for i in range(n_features)],
            "pretty_name": feature_names,
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_map.to_csv(OUT_PATH, index=False)

    print("\n✅ Feature name mapping successfully generated")
    print(f"📄 Saved to: {OUT_PATH.resolve()}")
    print("\nPreview:")
    print(df_map.head(10))


if __name__ == "__main__":
    main()
