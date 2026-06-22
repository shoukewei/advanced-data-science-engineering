# run_performance_pipeline.py

"""Quick runnable examples for chapter-20 `performance` utilities.

Run from the chapter-20 folder:

    python run_performance_examples.py

The script is intentionally lightweight so it runs quickly in CI/local.
"""
from pathlib import Path
import sys
import tempfile

# Ensure the local modules folder is importable
HERE = Path(__file__).resolve().parent
MODULES = str(HERE / "modules")
if MODULES not in sys.path:
    sys.path.insert(0, MODULES)

from performance import (
    timer,
    memory_profile,
    profile_function,
    memory_report,
    optimize_dtypes,
    process_in_chunks,
    parallel_standardize,
    make_cache_key,
    PipelineCache,
)

import numpy as np
import pandas as pd


def make_large_df(n=50000):
    np.random.seed(42)
    return pd.DataFrame({
        "TV":        np.random.uniform(0, 300, n),
        "Radio":     np.random.uniform(0, 50, n),
        "Newspaper": np.random.uniform(0, 120, n),
        "Sales":     np.random.uniform(1, 27, n),
        "Region":    np.random.choice(["North", "South", "East", "West"], n),
        "Channel":   np.random.choice(["TV", "Radio", "Digital", "Print"], n),
    })


@timer
def compute_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    return (numeric - numeric.mean()) / numeric.std()


def test_memory_and_profile(df):
    print("\n-- timing compute_scaled_features --")
    _ = compute_scaled_features(df)

    print("\n-- memory profile compute_scaled_features --")
    _, peak = memory_profile(compute_scaled_features, df)
    print(f"peak MB: {peak:.1f}")

    print("\n-- cProfile compute_groupby --")
    def compute_groupby(d):
        return d.groupby("Region")["Sales"].mean()
    _ = profile_function(compute_groupby, df, n_lines=8)


def test_dtype_optimization(df):
    print("\n-- memory report before optimize_dtypes --")
    print(memory_report(df).head(6).to_string(index=False))

    df_unopt = df.copy()
    df_unopt["TV"] = df_unopt["TV"].astype("float64")
    df_unopt["Radio"] = df_unopt["Radio"].astype("float64")
    df_unopt["Newspaper"] = df_unopt["Newspaper"].astype("float64")
    df_opt = optimize_dtypes(df_unopt)

    print("\n-- memory report after optimize_dtypes --")
    print(memory_report(df_opt).head(6).to_string(index=False))


def test_parallel_and_chunking(df):
    print("\n-- parallel_standardize (3 cols) --")
    cols = ["TV", "Radio", "Newspaper"]
    df2 = df.copy()
    df2 = parallel_standardize(df2, cols, max_workers=2)
    print("parallel_standardize done; sample means:", df2[cols].mean().round(6).to_dict())

    print("\n-- process_in_chunks --")
    # write small CSV and process in two chunks
    tmp = tempfile.NamedTemporaryFile(prefix="perf_test_", suffix=".csv", delete=False)
    tmp_name = tmp.name
    tmp.close()
    df.head(10000).to_csv(tmp_name, index=False)

    def proc(chunk: pd.DataFrame) -> pd.DataFrame:
        # return simple aggregation per chunk
        return chunk.groupby(chunk.columns[4]).size().reset_index(name="count")

    res = process_in_chunks(tmp_name, proc, chunksize=5000)
    print("process_in_chunks result shape:", res.shape)


def test_cache(df):
    print("\n-- PipelineCache --")
    key = make_cache_key(df.head(100))
    cache = PipelineCache()

    def compute(x):
        return x.sum(numeric_only=True)

    out1 = cache.get_or_compute(key, compute, df.head(100))
    out2 = cache.get_or_compute(key, compute, df.head(100))
    print("cache.stats:", cache.stats())


def main():
    df = make_large_df(n=50000)
    test_memory_and_profile(df)
    test_dtype_optimization(df)
    test_parallel_and_chunking(df)
    test_cache(df)


if __name__ == "__main__":
    main()
