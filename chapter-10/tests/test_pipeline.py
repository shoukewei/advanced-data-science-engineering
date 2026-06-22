import pandas as pd

from dskit.framework import FunctionStep, StepRegistry
from dskit.pipeline import PreprocessingPipeline


def test_pipeline_runs_custom_steps_before_scaling(sample_df):
    def add_total_spend(df: pd.DataFrame) -> pd.DataFrame:
        df["total_spend"] = df["TV"] + df["Radio"] + df["Newspaper"]
        return df

    pipeline = PreprocessingPipeline(
        {
            "scaling": {"columns": ["total_spend"], "method": "standard"},
        },
        steps=[FunctionStep(add_total_spend, {})],
    )

    result = pipeline.fit_transform(sample_df.drop(columns=["Sales"]))

    assert "total_spend" in result.columns
    assert abs(result["total_spend"].mean()) < 1e-9


def test_step_registry_builds_named_steps(sample_df):
    registry = StepRegistry()
    registry.register_function(
        "drop_newspaper", lambda df: df.drop(columns=["Newspaper"])
    )

    step = registry.create("drop_newspaper")
    result = step.fit(sample_df).transform(sample_df)

    assert "Newspaper" not in result.columns
