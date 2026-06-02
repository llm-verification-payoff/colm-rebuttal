"""CLI output of this script:

Theoretical Verifier Gain vs empirical 10-attempt rejection-sampling gain, 12-model subset.

1. Pooled across the 9 datasets (Figure 2, n = 144 pairs)
   slope +0.405   Pearson r +0.892   two-sided t-test p < 0.001
   pair-level  bootstrap 95% CI:  slope [+0.358, +0.451]   r [+0.815, +0.939]
   model-level bootstrap 95% CI:  slope [+0.266, +0.510]   r [+0.695, +0.977]

2. Within each dataset, no averaging
   dataset             slope        r         p
   --------------------------------------------
   AIME               +0.153   +0.863   < 0.001
   GSM8K              +0.450   +0.902   < 0.001
   3SAT               +0.705   +0.878   < 0.001
   Sudoku             +0.640   +0.638   < 0.001
   Matrix Mult.       +0.244   +0.414   < 0.001
   CSQA               +0.240   +0.629   < 0.001
   MMLU (STEM)        +0.400   +0.619   < 0.001
   MMLU (Social)      +0.234   +0.525   < 0.001
   GPQA               +0.025   +0.051     0.542

3. Figure 5 spot-check: cross-family similarity vs verifier FPR
   n = 108   r +0.613   p < 0.001

Run with: python correlation_significance.py
"""
import csv
import os

import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plotting_data")
SEED, BOOT = 0, 1000
rng = np.random.default_rng(SEED)

# (csv key, display label) in the order of the rebuttal table.
DATASETS = [("aime", "AIME"), ("gsm", "GSM8K"), ("sat", "3SAT"), ("sudoku", "Sudoku"),
            ("matmul", "Matrix Mult."), ("csqa", "CSQA"), ("mmlu_stem", "MMLU (STEM)"),
            ("mmlu_social_sciences", "MMLU (Social)"), ("gpqa", "GPQA")]
KEYS = [k for k, _ in DATASETS]


def read_matrix(metric, ds):
    with open(f"{DATA}/{metric}/{ds}.csv") as f:
        rows = list(csv.reader(f))
    labels = [r[0] for r in rows[1:]]
    return labels, np.array([[float(x) for x in r[1:]] for r in rows[1:]])


def avg_matrix(metric):
    """Per-(solver, verifier) matrix averaged over the 9 datasets, with its row labels."""
    labels = read_matrix(metric, KEYS[0])[0]
    return labels, np.nanmean([read_matrix(metric, ds)[1] for ds in KEYS], axis=0)


def family(nick):
    n = nick.lower()
    return ("DeepSeek" if "deepseek" in n else "Llama3" if "llama" in n
            else "Qwen3" if "qwen3" in n else "Qwen2.5")


def fit(x, y):
    """Slope, Pearson r, two-sided t-test p (H0: r = 0), and n."""
    x, y = np.asarray(x), np.asarray(y)
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return np.polyfit(x, y, 1)[0], r, 2 * stats.t.sf(abs(t), n - 2), n


def ci(v):
    return f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"


def pfmt(p):
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


# x = theoretical verifier gain, y = empirical 10-attempt gain, over the 12-model subset
subset, eg = avg_matrix("empirical_gain_K=10")           # 12 nicknames + 12 x 12
gain_labels, gain37 = avg_matrix("verifier_gain")        # 37 x 37
ix = [gain_labels.index(m) for m in subset]
vg = gain37[np.ix_(ix, ix)]                              # 12 x 12, same order as subset
x, y = vg.flatten(), eg.flatten()

print("Theoretical Verifier Gain vs empirical 10-attempt rejection-sampling gain, "
      "12-model subset.\n")

# 1. Significance test and the two bootstrap confidence intervals.
slope, r, p, n = fit(x, y)
print(f"1. Pooled across the 9 datasets (Figure 2, n = {n} pairs)")
print(f"   slope {slope:+.3f}   Pearson r {r:+.3f}   two-sided t-test p {pfmt(p)}")
bs = np.array([fit(x[s], y[s])[:2] for s in (rng.integers(0, n, n) for _ in range(BOOT))])
print(f"   pair-level  bootstrap 95% CI:  slope {ci(bs[:, 0])}   r {ci(bs[:, 1])}")
mb = []
for _ in range(BOOT):
    s = rng.integers(0, len(subset), len(subset))
    xx, yy = vg[np.ix_(s, s)].flatten(), eg[np.ix_(s, s)].flatten()
    if np.std(xx) > 1e-9:
        mb.append(fit(xx, yy)[:2])
mb = np.array(mb)
print(f"   model-level bootstrap 95% CI:  slope {ci(mb[:, 0])}   r {ci(mb[:, 1])}\n")

# 2. Same test within each dataset, no averaging.
print("2. Within each dataset, no averaging")
print(f"   {'dataset':<16}{'slope':>9}{'r':>9}{'p':>10}")
print("   " + "-" * 44)
for key, label in DATASETS:
    g_labels, g = read_matrix("verifier_gain", key)
    gix = [g_labels.index(m) for m in subset]
    vgd = g[np.ix_(gix, gix)].flatten()
    egd = read_matrix("empirical_gain_K=10", key)[1].flatten()
    s_, r_, p_, _ = fit(vgd, egd)
    print(f"   {label:<16}{s_:>+9.3f}{r_:>+9.3f}{pfmt(p_):>10}")
print()

# 3. Spot-check that another correlation reported in the paper also holds.
sim_labels, sim = avg_matrix("similarity")               # 12 x 12
fpr_labels, fpr37 = avg_matrix("verifier_fpr")           # 37 x 37
fix = [fpr_labels.index(m) for m in sim_labels]
fpr = fpr37[np.ix_(fix, fix)]                            # 12 x 12, same order as sim_labels
cross = [(i, j) for i in range(len(sim_labels)) for j in range(len(sim_labels))
         if i != j and family(sim_labels[i]) != family(sim_labels[j])]
s_, r_, p_, n_ = fit([sim[i, j] for i, j in cross], [fpr[i, j] for i, j in cross])
print("3. Figure 5 spot-check: cross-family similarity vs verifier FPR")
print(f"   n = {n_}   r {r_:+.3f}   p {pfmt(p_)}")
