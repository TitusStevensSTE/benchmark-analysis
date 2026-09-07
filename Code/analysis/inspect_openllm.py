import json
from huggingface_hub import list_repo_files, hf_hub_download


REPO_ID = "open-llm-leaderboard-old/results"

print("Finding files...")

files = list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

json_files = [
    f for f in files
    if f.endswith(".json")
]

print(f"Number of JSON files: {len(json_files)}")

# Take the first result file
file = json_files[0]

print("\nInspecting:")
print(file)

local_file = hf_hub_download(
    repo_id=REPO_ID,
    filename=file,
    repo_type="dataset"
)

with open(
    local_file,
    "r",
    encoding="utf-8"
) as f:

    result = json.load(f)

print("\nTop-level type:")
print(type(result))

print("\nTop-level keys:")

if isinstance(result, dict):
    print(result.keys())

print("\nFull structure:")
print(json.dumps(result, indent=2)[:10000])
print("\n" + "=" * 70)
print("SUMMARY TASKS")
print("=" * 70)

print(json.dumps(
    result["summary_tasks"],
    indent=2
)[:15000])