# run_dskit_pipeline.py

"""Quick runnable examples for chapter-21 `dskit` modules.

Run from the chapter-21 folder:

    python run_dskit_examples.py

The script demonstrates loading data, running the preprocessing pipeline,
and training two simple models via `ModelRegistry`.
"""
from pathlib import Path
import sys

# Make the local `dskit` package importable
HERE = Path(__file__).resolve().parent
DSKIT = str(HERE / "dskit")
if DSKIT not in sys.path:
    sys.path.insert(0, DSKIT)

# Some dskit modules import from a top-level package named `modules`.
# Provide a lightweight shim so `import modules.*` resolves to the `dskit`
# folder during this test script.
import types
if "modules" not in sys.modules:
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = [DSKIT]
    sys.modules["modules"] = modules_pkg

import warnings
warnings.filterwarnings("ignore")

from data_io import load_dataset
from preprocessing import compute_fill_values, fill_missing
from pipeline import PreprocessingPipeline, summarise_pipeline
from modeling import ModelRegistry

import numpy as np
import pandas as pd


def main():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

    print("Loading Advertising dataset...")
    df = load_dataset(url, index_col=0)
    print("Loaded shape:", df.shape)

    # Define a simple pipeline config similar to the chapter examples
    config = {
        "feature_engineering": {
            "log_columns": ["TV"]
        },
        "missing": {
            "strategies": {"TV": "median", "radio": "mean"},
            "indicator_columns": []
        },
        "outliers": {"columns": ["TV"], "method": "iqr", "multiplier": 1.5},
        "scaling": {"columns": ["TV", "radio", "newspaper"], "method": "standard"},
    }

    pipeline = PreprocessingPipeline(config)
    df_clean = pipeline.fit_transform(df)
    print("Preprocessing complete; cleaned shape:", df_clean.shape)
    summarise_pipeline(pipeline)

    # Prepare training data
    feature_cols = config["scaling"]["columns"]
    X = df_clean[feature_cols]
    # the Advertising dataset uses lowercase 'sales'
    y = df_clean["sales"] if "sales" in df_clean.columns else df_clean["Sales"]

    # Train/validate split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    # Register two simple models and evaluate
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    registry = ModelRegistry(task="regression")
    registry.register("LinearRegression", LinearRegression())
    registry.register("DecisionTree", DecisionTreeRegressor(max_depth=5, random_state=42))

    results = registry.fit_all(X_train, y_train, X_val, y_val)
    print("\nModel comparison results:\n", results)


if __name__ == "__main__":
    main()
