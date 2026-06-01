# COLM-Rebuttal

Anonymous repository for our COLM 2026 rebuttal. This repository also contains code and data for fully reproducing our paper's results, from inference to plotting!

## Reviewer 1

| Concern | What we provide | Where |
|---|---|---|
| **W1** Verifier gain ignores acceptance rate, so a strict verifier might score high by accepting very few outputs | Per-setting verifier gain, false-negative rate, and average candidates to acceptance: the highest-gain verification setting has the lowest false-negative rate | `r1_w1.py` |
| **W3** Verifier Gain is asymptotic, not actual post-verification performance | 2×3 bar plot: theoretical Verifier Gain (top) vs empirical 10-attempt rejection-sampling gain (bottom), across self / intra-family / cross-family verification | `viz/R1-W3.pdf` |
| **W5** No per-(solver, verifier, dataset) numerical results | Per-(solver, verifier, dataset) CSVs, additional data, and plotting code that reproduce every plot in the paper. | `plotting_data/`, `plotter.py` |
| **W6 / Q1** How does counting filtered outputs as incorrect affect main results? | Main Figs (1, 3, 5, 6) redone with filtered solver outputs counted as incorrect | `viz/R1-W6-Q1_fig1.pdf`, `viz/R1-W6-Q1_fig3.pdf`, `viz/R1-W6-Q1_fig5.pdf`, `viz/R1-W6-Q1_fig6.pdf` |

## Reviewer 2

| Concern | What we provide | Where |
|---|---|---|
| **W2** Post-training analysis covers only Qwen2.5 and Qwen3, so trends may reflect family-specific recipes and data | Llama3 base vs post-trained verifier gain by verification setting: Llama3-Base is negative in every setting (it cannot follow the verification instruction) | `r2_w2.py` |

## Reviewer 3

| Concern | What we provide | Where |
|---|---|---|
| **W3** Verifier Gain validation lacks confidence intervals, significance tests, and per-dataset robustness checks | Significance tests and bootstrap CIs for the Figure 2 correlation (pair-level plus a more conservative model-level resampling), a per-dataset breakdown, and a Figure 5 spot-check | `r3_w3.py` |

## Repo layout

- `plotter.py`: generates all plots; reads from `plotting_data/`, writes to `viz/`.
- `r1_w1.py`: prints the Reviewer 1 / W1 numbers (verifier gain, false-negative rate, and candidates to acceptance per verification setting) from `plotting_data/`.
- `r2_w2.py`: prints the Reviewer 2 / W2 numbers (verifier gain by verification setting, base vs post-trained) from `plotting_data/`.
- `r3_w3.py`: prints the Reviewer 3 / W3 numbers (significance, bootstrap CIs, per-dataset robustness) from `plotting_data/`.
- `plotting_data/`: numerical CSVs (one folder per metric, one file per dataset).
- `src/`: full inference + verifier + rejection-sampling pipeline. See `src/README.md` and `src/requirements.txt` for inference setup (requires GPU).
- `viz/`: PDF and PNG output. Files named `figure_N` correspond to paper figure numbers; files prefixed `R1-*` are rebuttal plots.

## Running Solver/Verifier Inference

See `src/README.md` for setup details and commands.

## Reproducing Paper/Rebuttal Plots and Analyses

```bash
pip install numpy matplotlib scipy
python plotter.py
```

This regenerates all 12 paper figures (`viz/figure_1.pdf` ... `viz/figure_12.pdf`) and the four rebuttal plots (`viz/R1-W3.pdf`, `viz/R1-W6-Q1_fig{1,3,5,6}.pdf`), both as PDF and PNG.

The per-reviewer rebuttal analyses print their numbers to stdout (each script also reproduces its output in a header comment):

```bash
python r1_w1.py
python r2_w2.py
python r3_w3.py
```