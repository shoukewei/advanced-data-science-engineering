# run_polars_pipeline.py
import pandas as pd
import polars as pl

# Import the production functions directly from your modules folder
from modules.data_io import load_dataset
from modules.preprocessing import fill_missing_polars, standardize_polars

if __name__ == "__main__":
    # URL to the Advertising dataset used throughout the chapter
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
    
    print("--- Step 1: Loading Native Polars DataFrame ---")
    # By passing backend="polars" and extracting to_pandas manually,
    # we prevent read_csv from breaking on foreign arguments.
    df_polars = load_dataset(url, backend="polars", to_pandas=False)
    
    print(f"Loaded type: {type(df_polars)}")
    print(f"Initial shape: {df_polars.shape}\n")

    print("--- Step 2: Preprocessing with Polars Expressions ---")
    # Fill missing values using Polars strategies
    df_filled = fill_missing_polars(df_polars, {"TV": "median", "radio": "mean"})
    
    # Scale numerical features across all available threads simultaneously
    feature_cols = ["TV", "radio", "newspaper"]
    df_scaled = standardize_polars(df_filled, feature_cols)
    
    print("Polars standardized means (should be ~0):")
    print(df_scaled.select(feature_cols).mean())
    print("")

    print("--- Step 3: Converting at the Modeling Boundary ---")
    # Pragmatic Hybrid Approach: Convert to Pandas just before scikit-learn
    X_scaled = df_scaled.select(feature_cols).to_pandas()
    y = df_scaled.select("sales").to_pandas().squeeze() 

    print(f"Conversion complete!")
    print(f"X shape (Pandas DataFrame): {X_scaled.shape}")
    print(f"y shape (Pandas Series):    {y.shape}")
    print("\nData is ready for the scikit-learn modeling layer.")