from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_FILE = (
     PROJECT_ROOT
    / "benchbench"
    / "benchbench"
    / "data"
    / "imagenet"
    / "leaderboard.tsv"
)
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots' / 'imagenet'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LODO_OUTPUT = RESULTS_DIR / 'imagenet_lodo_results.csv'
FULL_OUTPUT = RESULTS_DIR / 'imagenet_full_ranking.csv'
RANK_MATRIX_OUTPUT = RESULTS_DIR / 'imagenet_lodo_rank_matrix.csv'
SUMMARY_OUTPUT = RESULTS_DIR / 'imagenet_lodo_summary.txt'
UNIQUE_LODO_OUTPUT = RESULTS_DIR / 'imagenet_unique_architecture_lodo_results.csv'
UNIQUE_FULL_OUTPUT = RESULTS_DIR / 'imagenet_unique_architecture_full_ranking.csv'
UNIQUE_RANK_MATRIX_OUTPUT = RESULTS_DIR / 'imagenet_unique_architecture_lodo_rank_matrix.csv'
UNIQUE_SUMMARY_OUTPUT = RESULTS_DIR / 'imagenet_unique_architecture_lodo_summary.txt'
COMPARISON_OUTPUT = RESULTS_DIR / 'imagenet_weight_vs_architecture_lodo_comparison.csv'

COMPETITOR_COL = 'Weight'
ARCHITECTURE_COL = 'Model'


def rank_series(scores):
    return scores.rank(method='average', ascending=False)


def ranking_metrics(full_rank, lodo_rank):
    tau, tau_p = kendalltau(full_rank, lodo_rank)
    rho, rho_p = spearmanr(full_rank, lodo_rank)
    rank_difference = (lodo_rank - full_rank).abs()
    return {
        'kendall_tau': tau,
        'kendall_p_value': tau_p,
        'spearman_rho': rho,
        'spearman_p_value': rho_p,
        'max_rank_change': rank_difference.max(),
        'num_rank_changes': int((rank_difference > 0).sum()),
        'mean_rank_change': rank_difference.mean(),
        'median_rank_change': rank_difference.median(),
    }


def save_bar_plot(x, y, xlabel, ylabel, title, filename, rotation=35):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=rotation)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_lodo_analysis(data, task_cols, competitor_col, label, lodo_output, full_output, rank_matrix_output):
    work = data[[competitor_col, *task_cols]].copy()
    work['full_score'] = work[task_cols].mean(axis=1)
    work['full_rank'] = rank_series(work['full_score'])
    work['_row_order'] = np.arange(len(work))

    full_ranking = (
        work[[competitor_col, 'full_score', 'full_rank', '_row_order']]
        .sort_values(['full_score', '_row_order'], ascending=[False, True])
        .drop(columns='_row_order')
        .reset_index(drop=True)
    )
    full_ranking['rank_position'] = np.arange(1, len(full_ranking) + 1)
    full_ranking = full_ranking[['rank_position', competitor_col, 'full_score', 'full_rank']]
    full_ranking.to_csv(full_output, index=False)

    full_winner_idx = work['full_score'].idxmax()
    full_winner = work.loc[full_winner_idx, competitor_col]
    full_winner_score = work.loc[full_winner_idx, 'full_score']

    lodo_results = []
    rank_columns = {'full_rank': work['full_rank']}
    full_top_5 = set(work.nsmallest(5, 'full_rank').index)

    for removed_task in task_cols:
        remaining_tasks = [task for task in task_cols if task != removed_task]
        lodo_score = work[remaining_tasks].mean(axis=1)
        lodo_rank = rank_series(lodo_score)
        rank_columns[f'without_{removed_task}'] = lodo_rank
        metrics = ranking_metrics(work['full_rank'], lodo_rank)
        lodo_winner_idx = lodo_score.idxmax()
        lodo_winner = work.loc[lodo_winner_idx, competitor_col]
        full_winner_lodo_rank = lodo_rank.loc[full_winner_idx]
        lodo_top_5 = set(lodo_rank.nsmallest(5).index)
        lodo_results.append({
            'removed_task': removed_task,
            'num_remaining_tasks': len(remaining_tasks),
            **metrics,
            'full_winner': full_winner,
            'lodo_winner': lodo_winner,
            'winner_changed': lodo_winner != full_winner,
            'full_winner_lodo_rank': full_winner_lodo_rank,
            'top_5_overlap': len(full_top_5 & lodo_top_5),
        })

    results = pd.DataFrame(lodo_results)
    results['task_order'] = pd.Categorical(results['removed_task'], categories=task_cols, ordered=True)
    results = results.sort_values('task_order').drop(columns='task_order').reset_index(drop=True)
    results.to_csv(lodo_output, index=False)

    rank_matrix = pd.DataFrame(rank_columns)
    rank_matrix.insert(0, competitor_col, work[competitor_col].values)
    rank_matrix.to_csv(rank_matrix_output, index=False)
    return results, full_ranking, full_winner, full_winner_score


print('=' * 70)
print('IMAGENET LODO ANALYSIS')
print('=' * 70)
print(f'\nInput file:\n{IMAGENET_FILE}')
if not IMAGENET_FILE.exists():
    raise FileNotFoundError(IMAGENET_FILE)

df = pd.read_csv(IMAGENET_FILE, sep='\t')
print('\nShape:', df.shape)
print('Columns:', df.columns[:10].tolist(), '...')

TASK_COLS = [c for c in df.columns if c.startswith('acc_')]
if len(TASK_COLS) != 1000:
    raise ValueError(f'Expected 1000 acc_* task columns, found {len(TASK_COLS)}')

for c in TASK_COLS + ['acc', 'Acc@1']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
if df[TASK_COLS].isna().any().any():
    raise ValueError('Missing/non-numeric class accuracy values found.')

# Verify stored overall accuracy.
df['computed_average'] = df[TASK_COLS].mean(axis=1)
max_average_difference = (df['acc'] - df['computed_average']).abs().max()
print('\nAccuracy consistency check')
print('Maximum |acc - mean(acc_0..acc_999)|:', max_average_difference)
if max_average_difference > 1e-6:
    raise ValueError('Stored acc does not match class-wise mean.')

print('\nNumber of class tasks:', len(TASK_COLS))
print('Number of weight rows:', len(df))
print('Unique Weight identifiers:', df[COMPETITOR_COL].nunique())
print('Unique architectures:', df[ARCHITECTURE_COL].nunique())

# Row/weight-level analysis: each checkpoint/weight is a competitor.
row_df = df[[COMPETITOR_COL, *TASK_COLS]].copy()
row_results, row_full_ranking, row_full_winner, row_full_winner_score = run_lodo_analysis(
    row_df, TASK_COLS, COMPETITOR_COL, 'Weights', LODO_OUTPUT, FULL_OUTPUT, RANK_MATRIX_OUTPUT
)

# Architecture-level robustness: average variants belonging to the same architecture.
arch_df = df.groupby(ARCHITECTURE_COL, as_index=False, sort=False)[TASK_COLS].mean()
arch_df = arch_df.rename(columns={ARCHITECTURE_COL: COMPETITOR_COL})
unique_results, unique_full_ranking, unique_full_winner, unique_full_winner_score = run_lodo_analysis(
    arch_df, TASK_COLS, COMPETITOR_COL, 'Architectures', UNIQUE_LODO_OUTPUT, UNIQUE_FULL_OUTPUT, UNIQUE_RANK_MATRIX_OUTPUT
)

comparison_rows = []
for analysis_name, results, n_items in [
    ('Weights', row_results, len(row_df)),
    ('Unique architectures', unique_results, len(arch_df)),
]:
    for _, r in results.iterrows():
        comparison_rows.append({
            'analysis': analysis_name,
            'n_competitors': n_items,
            'removed_task': r['removed_task'],
            'kendall_tau': r['kendall_tau'],
            'spearman_rho': r['spearman_rho'],
            'max_rank_change': r['max_rank_change'],
            'num_rank_changes': r['num_rank_changes'],
            'mean_rank_change': r['mean_rank_change'],
            'median_rank_change': r['median_rank_change'],
            'top_5_overlap': r['top_5_overlap'],
            'winner_changed': r['winner_changed'],
            'full_winner': r['full_winner'],
            'lodo_winner': r['lodo_winner'],
        })
pd.DataFrame(comparison_rows).to_csv(COMPARISON_OUTPUT, index=False)


def summary_lines(title, results, n_items, winner, winner_score):
    most_k = results.loc[results['kendall_tau'].idxmin()]
    largest = results.loc[results['max_rank_change'].idxmax()]
    winners = results.loc[results['winner_changed'], 'removed_task'].tolist()
    return [
        title,
        '=' * len(title),
        f'Competitors: {n_items}',
        f'Tasks: {len(TASK_COLS)} (ImageNet classes acc_0..acc_999)',
        f'Full benchmark winner: {winner}',
        f'Full benchmark score: {winner_score:.6f}',
        '',
        'Most influential task by lowest Kendall tau:',
        f"  {most_k['removed_task']}",
        f"  Kendall tau = {most_k['kendall_tau']:.6f}",
        f"  Spearman rho = {most_k['spearman_rho']:.6f}",
        f"  Mean rank change = {most_k['mean_rank_change']:.2f}",
        f"  Max rank change = {most_k['max_rank_change']:.0f}",
        f"  Num rank changes = {most_k['num_rank_changes']}",
        f"  LODO winner = {most_k['lodo_winner']}",
        '',
        'Largest maximum rank change:',
        f"  {largest['removed_task']}",
        f"  Max rank change = {largest['max_rank_change']:.0f}",
        f"  LODO winner = {largest['lodo_winner']}",
        '',
        'Winner-changing removals:',
        f"  {', '.join(winners) if winners else 'None'}",
    ]

SUMMARY_OUTPUT.write_text('\n'.join(summary_lines('ImageNet LODO summary — weight competitors', row_results, len(row_df), row_full_winner, row_full_winner_score)), encoding='utf-8')
UNIQUE_SUMMARY_OUTPUT.write_text('\n'.join(summary_lines('ImageNet LODO summary — unique architectures', unique_results, len(arch_df), unique_full_winner, unique_full_winner_score)), encoding='utf-8')

# Same plot set as OpenLLM, with compact x-axis handling for 1000 tasks.
def save_lodo_plots(results, prefix):
    safe = prefix.lower().replace(' ', '_')
    for metric, ylabel, filename_suffix, title_metric in [
        ('kendall_tau', 'Kendall tau', 'lodo_kendall_tau.png', 'Kendall tau'),
        ('spearman_rho', 'Spearman rho', 'lodo_spearman_rho.png', 'Spearman rho'),
    ]:
        p = results.sort_values(metric)
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.plot(np.arange(len(p)), p[metric].to_numpy(), linewidth=1)
        ax.set_xlabel('Removed ImageNet class (sorted by metric)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'ImageNet LODO ({prefix}): {title_metric} vs. full ranking')
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f'imagenet_{safe}_{filename_suffix}', dpi=300, bbox_inches='tight')
        plt.close(fig)

    p = results.sort_values('mean_rank_change', ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(p)); width = 0.38
    ax.plot(x, p['mean_rank_change'].to_numpy(), label='Mean absolute rank change')
    ax.plot(x, p['max_rank_change'].to_numpy(), label='Maximum rank change')
    ax.set_xlabel('Removed ImageNet class (sorted)')
    ax.set_ylabel('Rank change')
    ax.set_title(f'ImageNet LODO ({prefix}): rank movement')
    ax.legend(); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / f'imagenet_{safe}_lodo_rank_change.png', dpi=300, bbox_inches='tight'); plt.close(fig)

    p = results.sort_values('num_rank_changes', ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(np.arange(len(p)), p['num_rank_changes'].to_numpy(), linewidth=1)
    ax.set_xlabel('Removed ImageNet class (sorted)')
    ax.set_ylabel('Number of competitors with changed rank')
    ax.set_title(f'ImageNet LODO ({prefix}): changed ranks')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / f'imagenet_{safe}_lodo_num_rank_changes.png', dpi=300, bbox_inches='tight'); plt.close(fig)

save_lodo_plots(row_results, 'Weights')
save_lodo_plots(unique_results, 'Unique architectures')

print('\nFULL WEIGHT BENCHMARK')
print('Winner:', row_full_winner)
print('Winner score:', row_full_winner_score)
print('\nFULL ARCHITECTURE BENCHMARK')
print('Winner:', unique_full_winner)
print('Winner score:', unique_full_winner_score)
print('\nLOWEST KENDALL TAU (weights)')
print(row_results.loc[row_results['kendall_tau'].idxmin()][['removed_task','kendall_tau','spearman_rho','max_rank_change','num_rank_changes','lodo_winner']].to_string())
print('\nLOWEST KENDALL TAU (architectures)')
print(unique_results.loc[unique_results['kendall_tau'].idxmin()][['removed_task','kendall_tau','spearman_rho','max_rank_change','num_rank_changes','lodo_winner']].to_string())
print('\nSaved results:')
for p in [LODO_OUTPUT,FULL_OUTPUT,RANK_MATRIX_OUTPUT,SUMMARY_OUTPUT,UNIQUE_LODO_OUTPUT,UNIQUE_FULL_OUTPUT,UNIQUE_RANK_MATRIX_OUTPUT,UNIQUE_SUMMARY_OUTPUT,COMPARISON_OUTPUT]:
    print(' ', p)
print('\nSaved plots to:', PLOTS_DIR)