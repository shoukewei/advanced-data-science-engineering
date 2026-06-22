# run_persistence_pipeline.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from modules.data_io import load_dataset
from modules.pipeline import PreprocessingPipeline, summarise_pipeline
from modules.persistence import ArtifactStore


def main():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
    df = load_dataset(url, index_col=0)

    X = df[["TV", "radio", "newspaper"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simple preprocessing pipeline config: scale numeric features
    config = {
        "feature_engineering": {},
        "missing": {"strategies": {}},
        "outliers": {"columns": []},
        "scaling": {"columns": list(X.columns), "method": "standard"},
    }

    pipeline = PreprocessingPipeline(config)
    pipeline.fit(X_train)
    summarise_pipeline(pipeline)

    X_train_p = pipeline.transform(X_train)
    X_test_p  = pipeline.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_p, y_train)

    print(f"Trained model R²: {model.score(X_test_p, y_test):.4f}")

    artifact_dir = Path("artifacts/gradient_boosting_v1")
    artifact = (
        ArtifactStore("gradient_boosting_v1")
        .set_model(model)
        .set_pipeline(pipeline)
        .set_features(list(X.columns))
        .set_metadata({
            "model_class": type(model).__name__,
            "test_r2": round(model.score(X_test_p, y_test), 4),
            "training_rows": len(X_train),
            "target": "sales",
        })
    )

    artifact.save(str(artifact_dir))

    # Reload and verify predictions on raw data
    loaded = ArtifactStore.load(str(artifact_dir))
    preds_loaded = loaded.predict(X_test)
    preds_original = model.predict(X_test_p)

    print("Predictions identical:", np.allclose(preds_loaded, preds_original))


if __name__ == "__main__":
    main()
