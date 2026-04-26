"""
Generate LaTeX result tables from matlab_pipeline/output/all_metrics.pkl.

Usage:
    cd ~/projects/UAV
    python matlab_pipeline/generate_tables.py

Outputs:
    matlab_pipeline/output/tables/
        table_conv_metrics.tex
        table_timeaware_metrics.tex
        table_deployability.tex
        table_pairwise_tests.tex
"""
import sys, os, pickle, math, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from statistical_tests import aggregate_metrics, pairwise_comparisons

OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
TABLE_DIR  = os.path.join(OUT_DIR, 'tables')
METRICS_PKL = os.path.join(OUT_DIR, 'all_metrics.pkl')
os.makedirs(TABLE_DIR, exist_ok=True)

MODEL_ORDER = ['CNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'TCN', 'Transformer']

DR_PASS = 0.85
MTBFA_PASS = 0.10


def _fmt(v, fmt=':.3f'):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return '---'
    return format(v, fmt.lstrip(':'))


def _fmt_ci(agg, key, unit='', fmt='.3f'):
    m  = agg.get(key, {}).get('mean', float('nan'))
    lo = agg.get(key, {}).get('ci_lo', float('nan'))
    hi = agg.get(key, {}).get('ci_hi', float('nan'))
    if math.isnan(m):
        return '---'
    if math.isinf(m):
        return r'$\infty$'
    m_s  = format(m,  fmt)
    err  = max(m - lo, hi - m)
    if math.isnan(err):
        return f'{m_s}{unit}'
    err_s = format(err, fmt)
    return f'${m_s} \\pm {err_s}${unit}'


def table_conv(all_metrics: dict) -> str:
    lines = [
        r'\begin{table}[!t]',
        r'\caption{Overall performance under conventional metrics (mean $\pm$ 95\% CI, 5 seeds).}',
        r'\label{tab:overall-conv}',
        r'\centering',
        r'\begin{tabular}{lccc}',
        r'\hline',
        r'\textbf{Model} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} \\',
        r'\hline',
    ]
    for mn in MODEL_ORDER:
        if mn not in all_metrics:
            continue
        agg = aggregate_metrics(all_metrics[mn])
        p  = _fmt_ci(agg, 'precision')
        r  = _fmt_ci(agg, 'recall')
        f1 = _fmt_ci(agg, 'f1')
        lines.append(f'{mn} & {p} & {r} & {f1} \\\\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def table_timeaware(all_metrics: dict) -> str:
    lines = [
        r'\begin{table}[!t]',
        r'\caption{Overall performance under time-aware metrics (mean $\pm$ 95\% CI, 5 seeds).}',
        r'\label{tab:overall-timeaware}',
        r'\centering',
        r'\resizebox{\columnwidth}{!}{%',
        r'\begin{tabular}{lcccccc}',
        r'\hline',
        (r'\textbf{Model} & \textbf{DR@1s} & \textbf{DR@3s} & \textbf{DR@5s} '
         r'& \textbf{DR@10s} & \textbf{ADD (s)} & \textbf{MTBFA (h)} \\'),
        r'\hline',
    ]
    for mn in MODEL_ORDER:
        if mn not in all_metrics:
            continue
        agg = aggregate_metrics(all_metrics[mn])
        dr1  = _fmt_ci(agg, 'dr_at_1s')
        dr3  = _fmt_ci(agg, 'dr_at_3s')
        dr5  = _fmt_ci(agg, 'dr_at_5s')
        dr10 = _fmt_ci(agg, 'dr_at_10s')
        add  = _fmt_ci(agg, 'add_s', fmt='.2f')
        mt   = _fmt_ci(agg, 'mtbfa_h', fmt='.3f')
        lines.append(f'{mn} & {dr1} & {dr3} & {dr5} & {dr10} & {add} & {mt} \\\\')
    lines += [r'\hline', r'\end{tabular}%', r'}', r'\end{table}']
    return '\n'.join(lines)


def table_deployability(all_metrics: dict) -> str:
    lines = [
        r'\begin{table}[!t]',
        r'\caption{Deployability decisions under conventional and time-aware frameworks.}',
        r'\label{tab:deployability}',
        r'\centering',
        r'\resizebox{\columnwidth}{!}{%',
        r'\begin{tabular}{llll}',
        r'\hline',
        r'\textbf{Model} & \textbf{Conventional} & \textbf{Time-aware} & \textbf{Reason (if fail)} \\',
        r'\hline',
    ]
    for mn in MODEL_ORDER:
        if mn not in all_metrics:
            continue
        agg = aggregate_metrics(all_metrics[mn])
        p  = agg.get('precision', {}).get('mean', float('nan'))
        r  = agg.get('recall',    {}).get('mean', float('nan'))
        f1 = agg.get('f1',        {}).get('mean', float('nan'))
        dr5   = agg.get('dr_at_5s', {}).get('mean', float('nan'))
        mtbfa = agg.get('mtbfa_h',  {}).get('mean', float('nan'))

        conv_pass = all(v >= DR_PASS for v in [p, r, f1] if not math.isnan(v))
        ta_dr_ok  = (not math.isnan(dr5))  and dr5   >= DR_PASS
        ta_mt_ok  = (not math.isnan(mtbfa)) and (math.isinf(mtbfa) or mtbfa >= MTBFA_PASS)
        ta_pass   = ta_dr_ok and ta_mt_ok

        conv_str = r'\checkmark Pass' if conv_pass else r'$\times$ Fail'
        ta_str   = r'\checkmark Pass' if ta_pass   else r'$\times$ Fail'

        reason = '--'
        if not ta_dr_ok and not math.isnan(dr5):
            reason = f'DR@5s $= {dr5:.1%} < 85\\%$'
        elif not ta_mt_ok and not math.isnan(mtbfa) and not math.isinf(mtbfa):
            reason = f'MTBFA $= {mtbfa:.3f}\\,\\mathrm{{h}} < 0.1\\,\\mathrm{{h}}$'

        lines.append(f'{mn} & {conv_str} & {ta_str} & {reason} \\\\')
    lines += [r'\hline', r'\end{tabular}%', r'}', r'\end{table}']
    return '\n'.join(lines)


def table_pairwise(all_metrics: dict) -> str:
    comparisons = pairwise_comparisons(all_metrics, metric_key='dr_at_5s')
    lines = [
        r'\begin{table}[!t]',
        r'\caption{Pairwise Bonferroni-corrected paired $t$-tests on DR@5s ($\alpha=0.05/21$).}',
        r'\label{tab:pairwise}',
        r'\centering',
        r'\begin{tabular}{llcc}',
        r'\hline',
        r'\textbf{Model A} & \textbf{Model B} & \textbf{$p$-value} & \textbf{Significant} \\',
        r'\hline',
    ]
    for (ma, mb), res in sorted(comparisons.items()):
        sig = r'\checkmark' if res['significant'] else '---'
        p_str = f"{res['p_value']:.4f}" if not math.isnan(res['p_value']) else '---'
        lines.append(f'{ma} & {mb} & {p_str} & {sig} \\\\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    if not os.path.exists(METRICS_PKL):
        print(f'ERROR: {METRICS_PKL} not found. Run main pipeline first.')
        return

    with open(METRICS_PKL, 'rb') as f:
        all_metrics = pickle.load(f)

    tables = {
        'table_conv_metrics.tex': table_conv(all_metrics),
        'table_timeaware_metrics.tex': table_timeaware(all_metrics),
        'table_deployability.tex': table_deployability(all_metrics),
        'table_pairwise_tests.tex': table_pairwise(all_metrics),
    }

    for fname, content in tables.items():
        path = os.path.join(TABLE_DIR, fname)
        with open(path, 'w') as f:
            f.write(content)
        print(f'  Wrote {path}')

    print(f'\nAll tables written to {TABLE_DIR}/')


if __name__ == '__main__':
    main()
