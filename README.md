# COLM-Rebuttal

Anonymous repository for our COLM 2026 rebuttal. This repository also contains code for fully reproducing our paper's results, from inference to plotting!

## Reviewer 1

| Concern | What we provide | Where |
|---|---|---|
| **W3** Verifier Gain is asymptotic, not actual post-verification performance | 2×3 bar plot: theoretical Verifier Gain (top) vs empirical 10-attempt rejection-sampling gain (bottom), across self / intra-family / cross-family verification | `viz/R1-W3.pdf` |
| **W5** No per-(solver, verifier, dataset) numerical results | Per-(solver, verifier, dataset) CSVs, additional data, and plotting code that reproduce every plot in the paper. | `plotting_data/`, `plotter.py` |
| **W6 / Q1** How does counting filtered outputs as incorrect affect main results? | Main Figs (1, 3, 5, 6) redone with filtered solver outputs counted as incorrect | `viz/R1-W6-Q1_fig1.pdf`, `viz/R1-W6-Q1_fig3.pdf`, `viz/R1-W6-Q1_fig5.pdf`, `viz/R1-W6-Q1_fig6.pdf` |

## Repo layout

- `plotter.py`: generates all plots; reads from `plotting_data/`, writes to `viz/`.
- `plotting_data/`: numerical CSVs (one folder per metric, one file per dataset).
- `src/`: full inference + verifier + rejection-sampling pipeline. See `src/README.md` and `src/requirements.txt` for inference setup (requires GPU).
- `viz/`: PDF and PNG output. Files named `figure_N` correspond to paper figure numbers; files prefixed `R1-*` are rebuttal plots.

## Running Solver/Verifier Inference

See `src/README.md` for setup details and commands.

## Reproducing Paper/Rebuttal Plots with `plotting_data/`

```bash
pip install numpy matplotlib
python plotter.py
```

This regenerates all 12 paper figures (`viz/figure_1.pdf` ... `viz/figure_12.pdf`) and the four rebuttal plots (`viz/R1-W3.pdf`, `viz/R1-W6-Q1_fig{1,3,5,6}.pdf`), both as PDF and PNG.