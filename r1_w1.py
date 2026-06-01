"""CLI output of this script:

W1: 12-model subset, averaged over 9 datasets and over solver-verifier pairs per verification setting.

verification setting    verifier gain    FNR    avg. number of candidates
self                            +1.2%    38%    2.8
intra-family                    +2.2%    42%    3.0
cross-family                    +6.2%    36%    3.3

Run with: python r1_w1.py
"""
import csv
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plotting_data")
DATASETS = ["sat", "matmul", "sudoku", "aime", "csqa", "gpqa", "gsm",
            "mmlu_social_sciences", "mmlu_stem"]


def read_matrix(metric, ds):
    with open(f"{DATA}/{metric}/{ds}.csv") as f:
        rows = list(csv.reader(f))
    labels = [r[0] for r in rows[1:]]
    return labels, np.array([[float(x) for x in r[1:]] for r in rows[1:]])


def avg_matrix(metric):
    """Per-(solver, verifier) matrix averaged over the 9 datasets, with its row labels."""
    labels = read_matrix(metric, DATASETS[0])[0]
    return labels, np.nanmean([read_matrix(metric, ds)[1] for ds in DATASETS], axis=0)


def family(nick):
    n = nick.lower()
    return ("DeepSeek" if "deepseek" in n else "Llama3" if "llama" in n
            else "Qwen3" if "qwen3" in n else "Qwen2.5")


def setting(solver, verifier):
    if solver == verifier:
        return "self"
    return "intra" if family(solver) == family(verifier) else "cross"


# 12-model subset: the models in the rejection-sampling CSVs
subset, _ = avg_matrix("empirical_gain_K=10")          # 12 nicknames, in order
fnr_labels, fnr = avg_matrix("verifier_fnr")           # 37 x 37
gain_labels, gain = avg_matrix("verifier_gain")        # 37 x 37
_, cand = avg_matrix("candidates_to_acceptance")       # 12 x 12, same order as `subset`
fnr = fnr[np.ix_([fnr_labels.index(m) for m in subset], [fnr_labels.index(m) for m in subset])]
gain = gain[np.ix_([gain_labels.index(m) for m in subset], [gain_labels.index(m) for m in subset])]


def by_setting(matrix):
    return {s: np.nanmean([matrix[i, j] for i in range(12) for j in range(12)
                           if setting(subset[i], subset[j]) == s])
            for s in ["self", "intra", "cross"]}


gain_s, fnr_s, cand_s = by_setting(gain), by_setting(fnr), by_setting(cand)

print("W1: 12-model subset, averaged over 9 datasets and over solver-verifier pairs per verification setting.\n")
print(f"{'verification setting':<22}{'verifier gain':>15}{'FNR':>7}    avg. number of candidates")
for s, label in [("self", "self"), ("intra", "intra-family"), ("cross", "cross-family")]:
    print(f"{label:<22}{gain_s[s] * 100:>+14.1f}%{fnr_s[s] * 100:>6.0f}%    {cand_s[s]:.1f}")
