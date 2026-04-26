"""Bootstrap CI and paired t-test for time-aware metric comparisons."""
import math
import numpy as np
from scipy import stats
import config


def bootstrap_ci(values: list, n_boot: int = config.N_BOOTSTRAP,
                 ci_level: float = config.CI_LEVEL) -> tuple:
    """Return (mean, lower_ci, upper_ci) via percentile bootstrap."""
    finite = [v for v in values if v is not None and not (isinstance(v, float) and math.isinf(v)) and not math.isnan(v)]
    if not finite:
        return float('nan'), float('nan'), float('nan')
    arr = np.array(finite, dtype=float)
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
                  for _ in range(n_boot)]
    alpha = 1 - ci_level
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(arr)), lo, hi


def aggregate_metrics(metrics_list: list) -> dict:
    """
    From a list of per-seed metric dicts, compute mean ± 95% CI for
    each numeric metric.

    Returns dict:  metric_key -> {'mean': float, 'ci_lo': float, 'ci_hi': float}
    """
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0] if k != 'seed']
    agg = {}
    for key in keys:
        vals = [m[key] for m in metrics_list]
        try:
            vals_f = [float(v) for v in vals]
            mean, lo, hi = bootstrap_ci(vals_f)
            agg[key] = {'mean': mean, 'ci_lo': lo, 'ci_hi': hi}
        except (TypeError, ValueError):
            agg[key] = {'mean': vals[0], 'ci_lo': None, 'ci_hi': None}
    return agg


def paired_ttest(values_a: list, values_b: list,
                 alpha_bonferroni: float = 0.05) -> dict:
    """
    Paired t-test between two sets of per-seed metric values.

    values_a, values_b : lists of floats (same length = number of seeds).
    alpha_bonferroni   : significance level (after Bonferroni correction
                         applied externally by the caller).

    Returns dict with t_stat, p_value, significant (bool), n.
    """
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {'t_stat': float('nan'), 'p_value': float('nan'),
                'significant': False, 'n': len(a)}
    t_stat, p_val = stats.ttest_rel(a, b)
    return {'t_stat': float(t_stat), 'p_value': float(p_val),
            'significant': bool(p_val < alpha_bonferroni), 'n': int(len(a))}


def pairwise_comparisons(all_metrics: dict, metric_key: str = 'dr_at_5s',
                         n_comparisons: int = None) -> dict:
    """
    Run all pairwise Bonferroni-corrected paired t-tests for `metric_key`
    across all models.

    all_metrics : {model_name: [metric_dict, ...]} from evaluate_all_models
    n_comparisons : total number of tests (for Bonferroni). If None, uses
                    C(n_models, 2).

    Returns nested dict: {(model_a, model_b): ttest_result_dict}
    """
    model_names = list(all_metrics.keys())
    n = len(model_names)
    if n_comparisons is None:
        n_comparisons = max(1, n * (n - 1) // 2)
    alpha_adj = 0.05 / n_comparisons

    results = {}
    for i in range(n):
        for j in range(i + 1, n):
            ma, mb = model_names[i], model_names[j]
            vals_a = [m[metric_key] for m in all_metrics[ma]]
            vals_b = [m[metric_key] for m in all_metrics[mb]]
            results[(ma, mb)] = paired_ttest(vals_a, vals_b, alpha_bonferroni=alpha_adj)
    return results


def print_summary_table(all_metrics: dict) -> None:
    """Print a human-readable summary table of all models and metrics."""
    header = (f"{'Model':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} "
              f"{'DR@5s':>6} {'ADD(s)':>7} {'MTBFA(h)':>9}")
    print('\n' + '=' * len(header))
    print(header)
    print('-' * len(header))
    for model_name, metrics_list in all_metrics.items():
        agg = aggregate_metrics(metrics_list)
        def fmt(k):
            v = agg.get(k, {}).get('mean', float('nan'))
            if math.isnan(v): return '  NaN'
            if math.isinf(v): return '  inf'
            return f'{v:>6.3f}'
        print(f"{model_name:<14} {fmt('precision')} {fmt('recall')} {fmt('f1')} "
              f"{fmt('dr_at_5s')} {fmt('add_s'):>7} {fmt('mtbfa_h'):>9}")
    print('=' * len(header))
