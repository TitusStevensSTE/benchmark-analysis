# Cross-Benchmark LODO Comparison

Canonical summary assembled from the current TabArena Score LODO and the Phase 1 BenchBench-loader analysis.

| benchmark | tasks | competitors | worst_kendall_tau | mean_kendall_tau | mean_lodo_instability | max_rank_change | max_num_rank_changes | winner_changed_removals | tau_break_frequency | lodo_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TabArena (Score) | 51 | 44 | 0.9704 | 0.9881 | 0.0119 | 5.0000 | 18 | 0 | 0.0000 | 51 |
| GLUE (score) | 9 | 87 | 0.9342 | 0.9723 | 0.0277 | 28.0000 | 70 | 3 | 0.0000 | 9 |
| OpenLLM (row_level) | 6 | 100 | 0.3030 | 0.6999 | 0.3001 | 90.0000 | 100 | 2 | 1.0000 | 6 |
| OpenLLM (unique_model_mean) | 6 | 87 | 0.3152 | 0.7044 | 0.2956 | 77.0000 | 87 | 2 | 1.0000 | 6 |
| ImageNet (score) | 1000 | 112 | 0.9977 | 0.9996 | 0.0004 | 1.0000 | 14 | 0 | 0.0000 | 1000 |
