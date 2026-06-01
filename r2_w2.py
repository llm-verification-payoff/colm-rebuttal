"""CLI output of this script:

Llama3 base vs post-trained verifier gain, by verification setting

                     self    intra    cross
  Llama3-Base      -0.083   -0.082   -0.527
  Llama3           +0.020   +0.055   +0.032

Run with: python r2_w2.py
"""
import numpy as np

import plotter as P  # reuse the repo's CSV loaders and family logic

SETTINGS = ["self", "intra", "cross"]

# verifier gain averaged over datasets, then per verifier within each setting (as in Figure 6)
rows, cols, M = P.matrix_avg_across_datasets("verifier_gain")
pv = {s: P.per_verifier_avg(M, rows, cols, s) for s in SETTINGS}


def gain(family, training):
    """Mean verifier gain (self, intra, cross) over one family's base or post-trained models."""
    idx = [i for i, c in enumerate(cols) if P.categorize_model(c) == (family, training)]
    return [float(np.mean([pv[s][i] for i in idx])) for s in SETTINGS]


def show(label, v):
    print(f"  {label:14s}" + "".join(f"{x:>+9.3f}" for x in v))


print("Llama3 base vs post-trained verifier gain, by verification setting")
print(f"  {'':14s}" + "".join(f"{h:>9}" for h in SETTINGS))
show("Llama3-Base", gain("Llama3", "Base"))
show("Llama3", gain("Llama3", "PostTrained"))
