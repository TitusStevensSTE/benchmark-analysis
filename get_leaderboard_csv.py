import os
from pathlib import Path
from tabrepo.tabarena.tabarena import TabArena
from autogluon.common.loaders import load_pd, load_pkl
from autogluon.common.savers import save_pd, save_pkl
import pandas as pd

def compute_normalized_error_dynamic(df_results: pd.DataFrame) -> pd.DataFrame:
    df_results = df_results.copy(deep=True)
    df_results_og = df_results.copy(deep=True)

    df_results = df_results.drop(columns=["normalized-error-dataset", "normalized-error-task"], errors="ignore")

    method_col = "method"

    df_results_per_dataset = df_results.groupby([method_col, "dataset"])["metric_error"].mean().reset_index(
        drop=False)

    from tabrepo.utils.normalized_scorer import NormalizedScorer

    # Alternative, this also incorporates Portfolios and HPO into the normalized scoring. This makes normalized-error dependent on what simulations we run.
    # This is unbiased against very strong simulation results because the best method defines what is `0.0` on a dataset.
    normalized_scorer_dataset = NormalizedScorer(
        df_results_per_dataset,
        tasks=list(df_results_per_dataset["dataset"].unique()),
        baseline=None,
        task_col="dataset",
        framework_col=method_col,
    )

    all_tasks = df_results[["dataset", "fold"]].drop_duplicates().values.tolist()
    all_tasks = [tuple(task) for task in all_tasks]

    normalized_scorer_task = NormalizedScorer(
        df_results,
        tasks=all_tasks,
        baseline=None,
        task_col=["dataset", "fold"],
        framework_col=method_col,
    )

    df_results["normalized-error-task"] = [normalized_scorer_task.rank(task=(dataset, fold), error=error) for
                                            (dataset, fold, error) in
                                            zip(df_results["dataset"], df_results["fold"],
                                                df_results["metric_error"])]

    df_results_per_dataset["normalized-error-dataset"] = [
        normalized_scorer_dataset.rank(task=dataset, error=error) for (dataset, error) in
        zip(df_results_per_dataset["dataset"], df_results_per_dataset["metric_error"])
    ]

    df_results_per_dataset = df_results_per_dataset.set_index(["dataset", method_col], drop=True)[
        "normalized-error-dataset"]
    df_results = df_results.merge(df_results_per_dataset, left_on=["dataset", method_col], right_index=True)

    df_results_og["normalized-error-dataset"] = df_results["normalized-error-dataset"]
    df_results_og["normalized-error-task"] = df_results["normalized-error-task"]
    return df_results_og


dataset_sizes = {'APSFailure': 76000.0,
    'Amazon_employee_access': 32769.0,
    'Another-Dataset-on-used-Fiat-500': 1538.0,
    'Bank_Customer_Churn': 10000.0,
    'Bioresponse': 3751.0,
    'Diabetes130US': 71518.0,
    'E-CommereShippingData': 10999.0,
    'Fitness_Club': 1500.0,
    'Food_Delivery_Time': 45451.0,
    'GiveMeSomeCredit': 150000.0,
    'HR_Analytics_Job_Change_of_Data_Scientists': 19158.0,
    'Is-this-a-good-customer': 1723.0,
    'MIC': 1699.0,
    'Marketing_Campaign': 2240.0,
    'NATICUSdroid': 7491.0,
    'QSAR-TID-11': 5742.0,
    'QSAR_fish_toxicity': 907.0,
    'SDSS17': 78053.0,
    'airfoil_self_noise': 1503.0,
    'anneal': 898.0,
    'bank-marketing': 45211.0,
    'blood-transfusion-service-center': 748.0,
    'churn': 5000.0,
    'coil2000_insurance_policies': 9822.0,
    'concrete_compressive_strength': 1030.0,
    'credit-g': 1000.0,
    'credit_card_clients_default': 30000.0,
    'customer_satisfaction_in_airline': 129880.0,
    'diabetes': 768.0,
    'diamonds': 53940.0,
    'hazelnut-spread-contaminant-detection': 2400.0,
    'healthcare_insurance_expenses': 1338.0,
    'heloc': 10459.0,
    'hiva_agnostic': 3845.0,
    'houses': 20640.0,
    'in_vehicle_coupon_recommendation': 12684.0,
    'jm1': 10885.0,
    'kddcup09_appetency': 50000.0,
    'maternal_health_risk': 1014.0,
    'miami_housing': 13776.0,
    'online_shoppers_intention': 12330.0,
    'physiochemical_protein': 45730.0,
    'polish_companies_bankruptcy': 5910.0,
    'qsar-biodeg': 1054.0,
    'seismic-bumps': 2584.0,
    'splice': 3190.0,
    'students_dropout_and_academic_success': 4424.0,
    'superconductivity': 21263.0,
    'taiwanese_bankruptcy_prediction': 6819.0,
    'website_phishing': 1353.0,
    'wine_quality': 6497.0
    }

if __name__ == '__main__':

    if os.path.exists("benchmark_results/df_results.parquet"):
        df_results = load_pd.load(path="benchmark_results/df_results.parquet")
    else:
        print(f"Loading results...")

        context_name = "tabarena_paper_full_51"
        s3_prefix_public = "https://tabarena.s3.us-west-2.amazonaws.com/evaluation"
        df_result_save_path = f"{context_name}/data/df_results.parquet"
        df_results = load_pd.load(path=f"{s3_prefix_public}/{df_result_save_path}")


        df_results.rename({"framework": "method"}, inplace=True,axis=1)

        df_results["method"] = df_results["method"].map({
            "AutoGluon_bq_4h8c": "AutoGluon 1.3 (4h)",
            "AutoGluon_bq_1h8c": "AutoGluon 1.3 (1h)",
            "AutoGluon_bq_5m8c": "AutoGluon 1.3 (5m)",
            "LightGBM_c1_BAG_L1": "GBM (default)",
            "XGBoost_c1_BAG_L1": "XGB (default)",
            "CatBoost_c1_BAG_L1": "CAT (default)",
            "NeuralNetTorch_c1_BAG_L1": "NN_TORCH (default)",
            "NeuralNetFastAI_c1_BAG_L1": "FASTAI (default)",
            "KNeighbors_c1_BAG_L1": "KNN (default)",
            "RandomForest_c1_BAG_L1": "RF (default)",
            "ExtraTrees_c1_BAG_L1": "XT (default)",
            "LinearModel_c1_BAG_L1": "LR (default)",
            "TabPFN_c1_BAG_L1": "TABPFN (default)",
            "RealMLP_c1_BAG_L1": "REALMLP (default)",
            "ExplainableBM_c1_BAG_L1": "EBM (default)",
            "FTTransformer_c1_BAG_L1": "FT_TRANSFORMER (default)",
            "TabPFNv2_c1_BAG_L1": "TABPFNV2 (default)",
            "TabICL_c1_BAG_L1": "TABICL (default)",
            'TabDPT_c1_BAG_L1': "TABDPT (default)",
            'TabM_c1_BAG_L1': "TABM (default)",
            'ModernNCA_c1_BAG_L1': "MNCA (default)",
        }).fillna(df_results["method"])

        df_results = df_results.loc[df_results["method"].apply(lambda x: "default" in x or "(tuned)" in x or "(tuned + ensemble)" in x or "AutoGluon 1.3 (4h)" in x)]
    
        df_results.loc[:, "seed"] = 0
        df_results.drop(columns=["config_selected", "metadata", "rank"], inplace=True, errors="ignore")

        save_pd.save(path="benchmark_results/df_results.parquet", df=df_results)
    
    
    elo_bootstrap_rounds = 100


    df_results = compute_normalized_error_dynamic(df_results)
    df_results["normalized-error"] = df_results["normalized-error-dataset"]

    df_results["num_instances"] = df_results["dataset"].map(dataset_sizes)

    df_results['time_train_s_per_1K'] = df_results['time_train_s'] * 1000 / (2 / 3 * df_results['num_instances'])
    df_results['time_infer_s_per_1K'] = df_results['time_infer_s'] * 1000 / (1 / 3 * df_results['num_instances'])


    tabarena = TabArena(
        method_col="method",
        task_col="dataset",
        seed_column="fold",
        error_col="metric_error",
        columns_to_agg_extra=[
            "time_train_s",
            "time_infer_s",
            "time_train_s_per_1K",
            "time_infer_s_per_1K",
            "normalized-error",
            "normalized-error-task",
        ],
        groupby_columns=[
            "metric",
            "problem_type",
        ],
    )

    calibration_framework = "RF (default)"

    # configs_all_success = ["TabPFNv2_c1_BAG_L1"]
    # datasets_tabpfn_valid = self.repo.datasets(configs=configs_all_success, union=False)
    # df_results_rank_compare3 = df_results_rank_compare[df_results_rank_compare["dataset"].isin(datasets_tabpfn_valid)]

    leaderboard = tabarena.leaderboard(
        data=df_results,
        # data=df_results_rank_compare3,
        include_winrate=True,
        include_mrr=True,
        # include_failure_counts=True,
        include_rank_counts=True,
        include_elo=True,
        elo_kwargs=dict(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=elo_bootstrap_rounds,
        )
    )
    elo_map = leaderboard["elo"]
    leaderboard = leaderboard.reset_index(drop=False)

    save_pd.save(path=f"benchmark_results/tabarena_leaderboard.csv", df=leaderboard)
