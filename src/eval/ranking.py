"""Expected-value ranking metrics under uniform random tie-breaking.

The legacy protocol broke score-ties with a single fixed permutation
(``sub.sample(frac=1.0, random_state=0)`` then a sort). Because
pairs.parquet writes the gold successor as row 0 of every anchor group,
that fixed permutation lands the gold at the same deterministic slot
within every same-sized tie group — systematically depressing Hits@1
for discrete scorers (the LLM 0-10 rubric, exact_name, token overlaps)
by ~2-13 pp. Continuous scorers (cross-encoder, E5 cosine) are
unaffected (<0.5 pp).

These helpers instead compute the exact EXPECTATION of each metric over
uniformly random permutations within each tie group: deterministic,
seed-free, and fair to every scorer.

E[Hits@k]: probability that at least one positive lands in the top k.
E[RR]:     expected reciprocal rank of the first positive, using
           P(first positive at offset x within its tie group of size t
           containing g positives) = C(t-x, g-1) / C(t, g).
"""
from math import comb

import numpy as np


def expected_metrics(scores, labels, ks=(1, 3)):
    """Exact E[Hits@k] for each k and E[1/rank of first positive].

    Returns (dict k -> E[Hits@k], E[RR]). Anchors with no positive or a
    single candidate should be filtered by the caller (eligibility rule).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    lab = labels[order]
    n = len(s)
    hks = {k: 0.0 for k in ks}
    err = 0.0
    offset = 0
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        t = j - i
        g = int(lab[i:j].sum())
        if g > 0:
            # The first positive lies in this tie group.
            cg = comb(t, g)
            for x in range(1, t - g + 2):
                err += (comb(t - x, g - 1) / cg) / (offset + x)
            for k in ks:
                if offset >= k:
                    pass                       # group entirely below cutoff
                elif offset + t <= k:
                    hks[k] += 1.0              # group entirely above cutoff
                else:
                    m = k - offset             # slots visible from this group
                    hks[k] += 1.0 - comb(t - g, m) / comb(t, m)
            break
        offset += t
        i = j
    return hks, err


def per_anchor_expected(df_test, ks=(1, 3)):
    """Yield (anchor_lei, {k: E[Hits@k]}, E[RR]) for eligible anchors
    (>=1 positive, >=2 candidates) — the shared eligibility rule."""
    for anchor, sub in df_test.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        hks, err = expected_metrics(sub["score"].values, sub["label"].values, ks)
        yield anchor, hks, err
