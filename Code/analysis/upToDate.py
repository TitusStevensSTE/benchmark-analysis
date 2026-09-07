import pandas as pd

url = "https://tabarena.s3.us-west-2.amazonaws.com/evaluation/tabarena_paper_full_51/data/df_results.parquet"

df = pd.read_parquet(url)

# Pick one dataset and one method
example = df[
    (df["dataset"] == "APSFailure") &
    (df["framework"] == "REALMLP (tuned + ensemble)")
]

print("=" * 70)
print("EXAMPLE: REALMLP (tuned + ensemble) on APSFailure")
print("=" * 70)

print("Rows:", len(example))

print("\nFold values:")
print(example["fold"].value_counts().sort_index())

print("\nSeed values:")
print(example["seed"].value_counts().sort_index())

print("\nConfig selected:")
print(example["config_selected"].value_counts().head(20))

print("\nMetric error:")
print(example["metric_error"].describe())

print("\nMetric error validation:")
print(example["metric_error_val"].describe())

print("\nFirst 20 rows:")
print(
    example[
        [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric_error_val",
            "seed",
            "config_selected"
        ]
    ].head(20).to_string(index=False)
)