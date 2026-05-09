# Codex Handoff — 2026-05-01

This file records what Codex actually did in this takeover session so Claude can continue paper writing without redoing or trusting speculative local edits.

## Important Scope Correction

- Codex initially misunderstood the task and made speculative local code/script edits.
- The user correctly stated that the authoritative scripts and experiment results are on the remote machine `abee`.
- Codex then reverted all local speculative code/script edits.
- Current local repository tracked files were clean before creating this handoff file and `CLAUDE.md`.
- The only pre-existing untracked local content left untouched is `step9_realworld/data/`.

## Local Edits That Were Reverted

These were created/modified locally by Codex, then removed/restored after user instruction:

- Restored tracked files to `HEAD`:
  - `.gitignore`
  - `docs/main.tex`
  - `step10_ablation/attack_param_sweep.py`
  - `step9_realworld/cross_domain_eval.py`
- Removed speculative local files:
  - `matlab_pipeline/config_600flight.py`
  - `matlab_pipeline/config_alfa_matched.py`
  - `matlab_pipeline/main_600flight.py`
  - `matlab_pipeline/main_alfa_matched.py`
  - `matlab_sim/config_600flight.m`
  - `matlab_sim/config_alfa_matched.m`
  - `matlab_sim/generate_mission_alfa_matched.m`
  - `matlab_sim/generate_one_flight_alfa_matched.m`
  - `matlab_sim/inject_attack_alfa_matched.m`
  - `matlab_sim/mass_runner_30k.m`
  - `matlab_sim/mass_runner_600flight.m`
  - `matlab_sim/mass_runner_alfa_matched.m`
  - `matlab_sim/simulate_flight_alfa_matched.m`
  - `run_alfa_matched.sh`
  - `step9_realworld/cross_domain_eval_600flight.py`
  - `step9_realworld/cross_domain_eval_alfa_matched.py`
  - `step9_realworld/extract_realworld_csvs.py`
- Removed local empty/accidental output directories:
  - `matlab_pipeline/output_600flight`
  - `matlab_pipeline/output_alfa_matched`
  - `matlab_pipeline/output/cross_domain`

Do not rely on these removed local files. Use remote `abee` outputs/scripts as source of truth.

## Remote Machine and Source of Truth

- Remote host: `administrator@100.100.84.61`
- Remote project root: `/home/administrator/projects/UAV`
- Remote venv: `source ~/TSL/venv/bin/activate`
- ROCm env used for GPU jobs: `export HSA_OVERRIDE_GFX_VERSION=11.0.0`

Claude previously ran the matched-ALFA and 600-flight controlled experiments on this remote. Codex only audited/completed missing outputs and generated paper-ready artifacts on the remote.

## Experiments Completed

### Already present when Codex audited remote

Remote outputs found:

- Main 5 m/s, 3000-flight pipeline:
  - `matlab_pipeline/output/aggregated_metrics.json`
  - `matlab_pipeline/output/all_metrics.pkl`
  - `matlab_pipeline/output/cross_domain/cross_domain_summary.json`
  - `matlab_pipeline/output/cross_domain/cross_domain_metrics.pkl`
- 5 m/s, 600-flight controlled pipeline:
  - `matlab_pipeline/output_600flight/aggregated_metrics.json`
  - `matlab_pipeline/output_600flight/all_metrics.pkl`
  - `matlab_pipeline/output_600flight/cross_domain/cross_domain_summary.json`
  - `matlab_pipeline/output_600flight/cross_domain/cross_domain_metrics.pkl`
- 20 m/s matched-ALFA, 600-flight pipeline:
  - `matlab_pipeline/output_alfa_matched/aggregated_metrics.json`
  - `matlab_pipeline/output_alfa_matched/all_metrics.pkl`
  - `matlab_pipeline/output_alfa_matched/cross_domain/cross_domain_summary.json`
  - `matlab_pipeline/output_alfa_matched/cross_domain/cross_domain_metrics.pkl`
- Existing ablations:
  - `matlab_pipeline/output/ablation_attack_params.json`
  - `matlab_pipeline/output/ablation_threshold_grid.json`

### Codex-completed missing experiment

Codex identified `window_step_sweep.py` as missing and ran it remotely:

```bash
cd ~/projects/UAV
source ~/TSL/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python -u step10_ablation/window_step_sweep.py > window_step_sweep.log 2>&1
```

Final output:

- `matlab_pipeline/output/ablation_window_step.json`
- Log: `window_step_sweep.log`

Window/step DR@5s table generated from the sweep:

```tex
\begin{tabular}{lccc}
\toprule
Window size & Step 3 & Step 5 & Step 10 \\
\midrule
30 & 0.951 & 0.948 & 0.949 \\
50 & 0.953 & 0.940 & 0.924 \\
80 & 0.951 & 0.930 & 0.965 \\
\bottomrule
\end{tabular}
```

Corresponding CSV:

- `matlab_pipeline/output/paper_ready/table_window_step_metrics.csv`

## Codex-generated Remote Analysis Artifacts

Codex generated paper-ready outputs on `abee`.

### Transfer analysis

Directory:

- `matlab_pipeline/output/transfer_analysis/`

Files:

- `transfer_alfa_sim_vs_real.csv`
- `transfer_alfa_between_regime_tests.csv`
- `transfer_all_domains.csv`
- `transfer_summary.json`
- `table_alfa_controlled_dr5.tex`

Key ALFA controlled-comparison DR@5s values:

| Model | 5 m/s 3000-flight | 5 m/s 600-flight | 20 m/s matched 600-flight |
|---|---:|---:|---:|
| CNN | 0.960 | 0.960 | 0.000 |
| LSTM | 0.920 | 0.960 | 0.600 |
| BiLSTM | 0.840 | 0.880 | 0.600 |
| GRU | 0.880 | 0.840 | 0.480 |
| CNN-LSTM | 0.960 | 0.920 | 0.080 |
| TCN | 1.000 | 0.680 | 0.160 |
| Transformer | 0.760 | 0.640 | 0.280 |

### Paper-ready tables and figures

Directory:

- `matlab_pipeline/output/paper_ready/`

Files:

- `table_per_attack_dr5.tex`
- `table_per_attack_metrics.csv`
- `table_threshold_grid_mtbfa01.tex`
- `table_threshold_grid_mtbfa01.csv`
- `table_window_step_dr5.tex`
- `table_window_step_metrics.csv`
- `fig_alfa_controlled_dr5.pdf`
- `fig_alfa_controlled_dr5.png`
- `fig_cross_domain_dr5_original.pdf`
- `fig_cross_domain_dr5_original.png`
- `fig_window_step_dr5_heatmap.pdf`
- `fig_window_step_dr5_heatmap.png`

### Final archive

Remote final archive:

- `matlab_pipeline/output/result_archives/uav_results_final_20260501_182507.tar.gz`

Local temporary backup copied by Codex:

- `C:\Users\Jawk\AppData\Local\Temp\UAV_remote_results\uav_results_final_20260501_182507.tar.gz`

This archive contains the main metrics, cross-domain results, ablations, transfer analysis, paper-ready tables/figures, and relevant logs.

## Current Experimental Claim Candidates

Use these as candidate findings, but Claude should decide final framing for the paper.

1. Time-aware metrics expose deployment-relevant failures hidden by conventional simulation metrics.
2. Sim-best models do not necessarily transfer best to real domains; architecture ranking can change under zero-shot transfer.
3. The controlled ALFA comparison shows a counter-intuitive effect: nominal platform-speed matching (`20 m/s matched 600-flight`) worsened ALFA transfer relative to both `5 m/s 3000-flight` and `5 m/s 600-flight`.
4. Since `5 m/s 600-flight` remains strong on ALFA, the matched-ALFA degradation is not explained only by reduced training-set size.

Suggested cautious wording:

> Matching nominal simulation speed to the target platform does not guarantee improved sim-to-real transfer. In our controlled ALFA comparison, the slower 5 m/s simulation regime transferred more robustly than the 20 m/s matched regime, suggesting that richer multi-timescale spoofing signatures may matter more than nominal platform-speed matching.

## Main Remaining Task

Write the paper revision using the remote results, especially:

- `docs/main.tex`
  - Fill cross-domain section around the existing §V-G placeholder.
  - Fill per-attack table placeholder.
  - Fill threshold/deployability table/figure references.
  - Fill window/step ablation placeholder.
  - Update limitations/discussion to include the controlled matched-vs-unmatched finding.
- `docs/response_to_reviewers.tex`
  - Replace placeholders for cross-domain DR@5s drops and Table IV-specific numbers.

Do not rerun long experiments unless a reviewer-specific gap remains after writing.

