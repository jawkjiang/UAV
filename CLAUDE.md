# CLAUDE Handoff

Read `CODEX_HANDOFF_2026-05-01.md` before continuing.

## What Codex Did

- Codex initially made speculative local code/script edits, then fully rolled them back after the user clarified that the authoritative scripts/results are on remote `abee`.
- Local tracked repo was clean before this handoff file and `CODEX_HANDOFF_2026-05-01.md` were added.
- Codex did not preserve local speculative scripts. Do not rely on local `*_alfa_matched*` or `*_600flight*` files unless you explicitly fetch them from `abee`.
- Codex completed the missing remote `window_step_sweep.py` experiment on `abee`.
- Codex generated remote transfer-analysis tables, paper-ready tables/figures, and a final archive.

## Authoritative Remote Outputs

Remote:

- Host: `administrator@100.100.84.61`
- Root: `/home/administrator/projects/UAV`
- Final archive: `matlab_pipeline/output/result_archives/uav_results_final_20260501_182507.tar.gz`
- Paper-ready directory: `matlab_pipeline/output/paper_ready/`
- Transfer-analysis directory: `matlab_pipeline/output/transfer_analysis/`

Local backup of final archive:

- `C:\Users\Jawk\AppData\Local\Temp\UAV_remote_results\uav_results_final_20260501_182507.tar.gz`

## Current Priority

The next task is paper writing, not experiment execution.

Update:

- `docs/main.tex`
- `docs/response_to_reviewers.tex`

Use the remote/paper-ready outputs to fill:

- cross-domain / sim-to-real section,
- controlled ALFA comparison,
- per-attack table,
- threshold/deployability table,
- window/step ablation table/heatmap,
- reviewer-response placeholders.

Main experimental framing to consider:

> Time-aware and cross-domain evaluation reveal deployment failures missed by simulation-only benchmarks; additionally, the controlled ALFA comparison shows that nominal platform-speed matching can worsen transfer, while the 5 m/s simulation regime transfers more robustly to ALFA than the 20 m/s matched regime.

