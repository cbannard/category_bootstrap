"""
Tests whether NOUN/VERB precision and recall differ significantly between
pattern types at each seed-set size, using the per-fold rows that
category_bootstrap.py writes to summary.csv / summary_parts/.

Why a PAIRED test across folds
------------------------------
make_kfold_sentence_splits() builds the folds deterministically from
--split-seed, so fold k is the SAME train/test split for every pattern
type, mode and seed step. Comparing pattern types at one seed step is
therefore a within-fold comparison: for each fold we take the difference
(pattern_type A - pattern_type B) and test whether the mean of those
k differences is zero. Pairing removes the fold-to-fold variation that
affects every pattern type alike (some test folds are simply easier),
which is usually most of the variance.

Two versions of the paired t-test are reported
----------------------------------------------
  p_paired     - the ordinary paired t-test on the k fold differences
                 (df = k - 1).
  p_corrected  - the Nadeau & Bengio (2003) corrected resampled t-test.
                 In k-fold CV the training sets overlap heavily (each pair
                 of folds shares (k-2)/(k-1) of its training data), so the
                 k differences are not independent and the ordinary test's
                 variance is too small, inflating false positives. The
                 correction replaces the 1/k variance factor with
                 1/k + n_test/n_train = 1/k + 1/(k-1). It is the more
                 conservative choice and is used for the significance flags
                 by default (--test corrected); use --test paired to flag on
                 the ordinary test instead.

Omnibus test
------------
With 3 pattern types, a one-way repeated-measures ANOVA (folds as
subjects) is also run at each step: "do the pattern types differ at all
here?". The pairwise tests then say which pairs differ.

Multiple comparisons
--------------------
Each (model, mode, metric) is a family of (#seed steps x #pairs) tests -
several hundred with the default sweep - so p-values are adjusted with the
Benjamini-Hochberg false-discovery-rate procedure within each family
(columns ending _fdr). The omnibus p-values get their own BH adjustment per
(model, mode, metric). Significance flags use the FDR-adjusted values.

Power caveat
------------
With the default 5 folds there are only 4 degrees of freedom, so only
fairly consistent differences will reach significance. More folds
(--n-folds 10) gives more power. A non-parametric alternative (sign test
or exact permutation test on the fold differences) cannot reach p < .05
at all with 5 folds - its smallest possible two-sided p is 2/2^5 = .0625 -
which is why it isn't used here.

Outputs (in --out-dir, default SUMMARY_DIR/pattern_type_significance/)
----------------------------------------------------------------------
  pairwise_tests.csv  one row per (model, mode, metric, seed step, pair):
                      means, mean difference, its 95% CI (corrected SE),
                      t, raw and FDR-adjusted p for both test versions,
                      and a `significant` flag.
  omnibus_tests.csv   one row per (model, mode, metric, seed step): the
                      repeated-measures ANOVA F, p and FDR-adjusted p.
  significance_summary.txt
                      per (mode, metric, pair): how many seed steps showed
                      a significant difference, and in which direction.
  diff_<mode>_<metric>.png
                      one figure per mode x metric: a panel per pair showing
                      the mean difference (A - B) with its 95% CI against
                      seed count; filled markers = significant after FDR.
                      The CI is per-step and NOT multiplicity-adjusted, so a
                      CI can exclude zero while the point stays hollow.

Usage:
    python3 compare_pattern_types.py SUMMARY_DIR [--out-dir DIR]
        [--pattern *.csv] [--alpha 0.05] [--test corrected|paired]
        [--metrics NOUN_precision NOUN_recall ...] [--no-plots]
"""

import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from plot_precision_recall import (
    METRICS,
    SWEPT_MODES,
    REQUIRED_COLUMNS,
    find_csv_files,
    group_files_by_model,
)

OWN_OUTPUT_DIR_NAME = "pattern_type_significance"
STEP_KEYS = ["num_noun_seeds", "num_verb_seeds"]
DEFAULT_METRICS = [m[0] for m in METRICS]
METRIC_X = {m[0]: (m[1], m[2]) for m in METRICS}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_fold_rows(summary_dir, pattern):
    """Return every individual-fold row (fold = 0..k-1, not the "mean"
    row) from every summary CSV under summary_dir, with a `model` column
    (same grouping rule as plot_precision_recall.py)."""
    paths = [
        p for p in find_csv_files(summary_dir, pattern)
        if OWN_OUTPUT_DIR_NAME not in os.path.normpath(p).split(os.sep)
    ]
    if not paths:
        raise FileNotFoundError(f"No file(s) matching '{pattern}' found under {summary_dir}")
    groups = group_files_by_model(paths, summary_dir)

    frames = []
    print(f"Found {len(paths)} CSV file(s) under {summary_dir}, grouped into {len(groups)} model(s):")
    for label, group_paths in sorted(groups.items()):
        model_frames = []
        for path in group_paths:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"  [{label}] skipping {path}: could not read as CSV ({e})")
                continue
            if REQUIRED_COLUMNS - set(df.columns):
                continue  # not a summary file (confusion_words_*, pattern_usage_*, ...)
            model_frames.append(df)
        if not model_frames:
            print(f"  [{label}] no summary CSVs")
            continue
        raw = pd.concat(model_frames, ignore_index=True)
        fold_str = raw["fold"].astype(str)
        is_fold = (fold_str != "mean") & (raw["n_folds"] > 1) & raw["fold"].notna()
        rows = raw[is_fold].copy()
        rows["fold"] = pd.to_numeric(rows["fold"], errors="coerce").astype("Int64")
        rows.insert(0, "model", label)
        n_single = int((raw["n_folds"] == 1).sum())
        if n_single:
            print(f"  [{label}] ignoring {n_single} row(s) from --n-folds 1 runs "
                  f"(a single split gives nothing to test across)")
        print(f"  [{label}] {len(group_paths)} file(s) -> {len(rows)} per-fold row(s)")
        frames.append(rows)

    if not frames or all(f.empty for f in frames):
        raise ValueError(
            "No per-fold rows found. Significance testing needs cross-validated "
            "runs (--n-folds >= 2, the default) whose individual fold rows are in "
            "summary.csv / summary_parts/."
        )
    df = pd.concat(frames, ignore_index=True)

    # A job that was re-run appends a second copy of its fold rows; keep the
    # most recent one so a fold isn't counted twice.
    key = ["model", "mode", "pattern_type"] + STEP_KEYS + ["fold"]
    if "time" in df.columns:
        df = df.sort_values("time")
    n_before = len(df)
    df = df.drop_duplicates(subset=key, keep="last")
    if len(df) < n_before:
        print(f"Note: dropped {n_before - len(df)} duplicate fold row(s) "
              f"(re-run jobs) - kept the most recent of each.")
    return df


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def paired_tests(diffs, n_folds):
    """Paired t-test and Nadeau-Bengio corrected t-test on per-fold
    differences. Returns a dict of statistics."""
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    out = {"n_folds_used": k, "mean_diff": np.nan, "sd_diff": np.nan,
           "t_paired": np.nan, "p_paired": np.nan,
           "t_corrected": np.nan, "p_corrected": np.nan,
           "ci_low": np.nan, "ci_high": np.nan}
    if k < 2:
        return out
    mean = d.mean()
    sd = d.std(ddof=1)
    out["mean_diff"], out["sd_diff"] = mean, sd
    dof = k - 1
    # Training/test ratio of a k-fold split: n_test/n_train = 1/(k_total-1).
    # Use the job's own n_folds (folds missing a value don't change the
    # split sizes).
    rho = 1.0 / (n_folds - 1)
    var_plain = sd ** 2 / k
    var_corr = sd ** 2 * (1.0 / k + rho)
    tcrit = stats.t.ppf(0.975, dof)

    if sd == 0:
        # Every fold gave exactly the same difference.
        p = 1.0 if mean == 0 else 0.0
        t = 0.0 if mean == 0 else np.sign(mean) * np.inf
        out.update(t_paired=t, p_paired=p, t_corrected=t, p_corrected=p,
                   ci_low=mean, ci_high=mean)
        return out

    t_plain = mean / np.sqrt(var_plain)
    t_corr = mean / np.sqrt(var_corr)
    out["t_paired"] = t_plain
    out["p_paired"] = 2 * stats.t.sf(abs(t_plain), dof)
    out["t_corrected"] = t_corr
    out["p_corrected"] = 2 * stats.t.sf(abs(t_corr), dof)
    half = tcrit * np.sqrt(var_corr)
    out["ci_low"], out["ci_high"] = mean - half, mean + half
    return out


def rm_anova(wide):
    """One-way repeated-measures ANOVA. `wide`: rows = folds (subjects),
    columns = pattern types, no NaNs. Returns (F, df1, df2, p)."""
    y = wide.to_numpy(dtype=float)
    n, a = y.shape
    if n < 2 or a < 2:
        return np.nan, np.nan, np.nan, np.nan
    grand = y.mean()
    ss_cond = n * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_subj = a * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_total = ((y - grand) ** 2).sum()
    ss_err = ss_total - ss_cond - ss_subj
    df1, df2 = a - 1, (a - 1) * (n - 1)
    if ss_err <= 1e-15:
        p = 1.0 if ss_cond <= 1e-15 else 0.0
        return (np.inf if p == 0 else 0.0), df1, df2, p
    F = (ss_cond / df1) / (ss_err / df2)
    return F, df1, df2, stats.f.sf(F, df1, df2)


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment, NaN-aware."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def run_tests(df, metrics, alpha, test_kind):
    df = df[df["mode"].isin(SWEPT_MODES)]
    if df.empty:
        raise ValueError(f"No per-fold rows for the swept modes {SWEPT_MODES}.")

    pair_rows, omni_rows = [], []
    group_cols = ["model", "mode"] + STEP_KEYS
    for (model, mode, nn, nv), sub in df.groupby(group_cols, sort=True):
        n_folds = int(sub["n_folds"].max())
        pattern_types = sorted(sub["pattern_type"].dropna().unique())
        if len(pattern_types) < 2:
            continue
        for metric in metrics:
            if metric not in sub.columns:
                continue
            wide = sub.pivot_table(index="fold", columns="pattern_type",
                                   values=metric, aggfunc="first")
            base = {"model": model, "mode": mode, "metric": metric,
                    "num_noun_seeds": nn, "num_verb_seeds": nv,
                    "n_folds": n_folds}

            complete = wide.dropna()
            if complete.shape[1] >= 3:
                F, df1, df2, p = rm_anova(complete)
                omni_rows.append({**base, "n_pattern_types": complete.shape[1],
                                  "n_folds_used": len(complete),
                                  "F": F, "df1": df1, "df2": df2, "p": p})

            for a, b in itertools.combinations(pattern_types, 2):
                if a not in wide.columns or b not in wide.columns:
                    continue
                pair = wide[[a, b]].dropna()
                res = paired_tests(pair[a] - pair[b], n_folds)
                pair_rows.append({
                    **base,
                    "pattern_type_A": int(a), "pattern_type_B": int(b),
                    "mean_A": pair[a].mean() if len(pair) else np.nan,
                    "mean_B": pair[b].mean() if len(pair) else np.nan,
                    **res,
                })

    pairwise = pd.DataFrame(pair_rows)
    omnibus = pd.DataFrame(omni_rows)

    family = ["model", "mode", "metric"]
    if not pairwise.empty:
        for col in ["p_paired", "p_corrected"]:
            pairwise[f"{col}_fdr"] = pairwise.groupby(family)[col].transform(
                lambda s: pd.Series(bh_adjust(s.values), index=s.index))
        flag_col = "p_corrected_fdr" if test_kind == "corrected" else "p_paired_fdr"
        pairwise["significant"] = pairwise[flag_col] < alpha
        pairwise["better"] = np.where(
            ~pairwise["significant"], "",
            np.where(pairwise["mean_diff"] > 0,
                     "p" + pairwise["pattern_type_A"].astype(str),
                     "p" + pairwise["pattern_type_B"].astype(str)))
    if not omnibus.empty:
        omnibus["p_fdr"] = omnibus.groupby(family)["p"].transform(
            lambda s: pd.Series(bh_adjust(s.values), index=s.index))
        omnibus["significant"] = omnibus["p_fdr"] < alpha
    return pairwise, omnibus


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_summary(pairwise, omnibus, path, alpha, test_kind):
    lines = [
        "Pattern-type comparisons at each seed step",
        f"Test: {'Nadeau-Bengio corrected' if test_kind == 'corrected' else 'ordinary'} "
        f"paired t-test across folds; BH-FDR adjusted within each model x mode x metric; "
        f"alpha = {alpha}",
        "",
    ]
    if pairwise.empty:
        lines.append("No pairwise comparisons could be made.")
    for (model, mode, metric), g in pairwise.groupby(["model", "mode", "metric"], sort=True):
        lines.append(f"[{model}] {mode}  {metric}")
        if not omnibus.empty:
            o = omnibus[(omnibus["model"] == model) & (omnibus["mode"] == mode)
                        & (omnibus["metric"] == metric)]
            if len(o):
                lines.append(f"  omnibus (any difference among pattern types): "
                             f"{int(o['significant'].sum())}/{len(o)} seed steps significant")
        for (a, b), gp in g.groupby(["pattern_type_A", "pattern_type_B"]):
            n = len(gp)
            sig = gp[gp["significant"]]
            a_wins = int((sig["mean_diff"] > 0).sum())
            b_wins = int((sig["mean_diff"] < 0).sum())
            lines.append(
                f"  p{a} vs p{b}: {len(sig)}/{n} steps significant "
                f"(p{a} higher at {a_wins}, p{b} higher at {b_wins}); "
                f"median diff (p{a}-p{b}) = {gp['mean_diff'].median():+.4f}")
            xcol = METRIC_X.get(metric, ("num_noun_seeds", ""))[0]
            if len(sig):
                xs = sorted(sig[xcol].unique())
                lines.append(f"      significant at {xcol} = {_compress_ranges(xs)}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def _compress_ranges(xs):
    xs = [int(x) for x in xs]
    if not xs:
        return ""
    parts, start, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = x
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(parts)


def plot_differences(pairwise, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#222222", "#6b6b6b", "#e3e3e3"
    sig_color = "#1f5fa8"
    written = []
    for (model, mode, metric), g in pairwise.groupby(["model", "mode", "metric"], sort=True):
        xcol, xlabel = METRIC_X.get(metric, ("num_noun_seeds", "Noun seed count"))
        pairs = sorted(g.groupby(["pattern_type_A", "pattern_type_B"]).groups.keys())
        fig, axes = plt.subplots(len(pairs), 1, figsize=(8.5, 2.6 * len(pairs)),
                                 sharex=True, squeeze=False)
        for ax, (a, b) in zip(axes[:, 0], pairs):
            gp = g[(g["pattern_type_A"] == a) & (g["pattern_type_B"] == b)].sort_values(STEP_KEYS)
            x = gp[xcol].to_numpy(dtype=float)
            y = gp["mean_diff"].to_numpy(dtype=float)
            ax.axhline(0, color=muted, linewidth=1)
            if gp[xcol].duplicated().any():
                # Several seed steps share this x (typically verb counts,
                # which repeat across consecutive noun counts): show each
                # step as its own point + CI, spread slightly sideways,
                # rather than a line that zig-zags vertically.
                rank = gp.groupby(xcol).cumcount().to_numpy()
                size = gp.groupby(xcol)[xcol].transform("size").to_numpy()
                x = x + (rank - (size - 1) / 2) * np.minimum(0.6 / np.maximum(size, 1), 0.12)
                ax.errorbar(x, y, yerr=[y - gp["ci_low"], gp["ci_high"] - y],
                            fmt="none", ecolor=sig_color, alpha=0.35, linewidth=1)
            else:
                ax.fill_between(x, gp["ci_low"], gp["ci_high"], color=sig_color,
                                alpha=0.12, linewidth=0)
                ax.plot(x, y, color=sig_color, linewidth=1.5)
            sig = gp["significant"].to_numpy(dtype=bool)
            ax.scatter(x[sig], gp["mean_diff"].to_numpy()[sig], s=22,
                       color=sig_color, zorder=3, label="significant (FDR)")
            ax.scatter(x[~sig], gp["mean_diff"].to_numpy()[~sig], s=22,
                       facecolor="white", edgecolor=sig_color, linewidth=1,
                       zorder=3, label="not significant")
            ax.set_ylabel(f"p{a} − p{b}", color=ink)
            ax.grid(axis="y", color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.text(0.99, 0.95, f"{int(sig.sum())}/{len(sig)} steps significant",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=muted)
        axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
        axes[-1, 0].set_xlabel(xlabel)
        title = f"{metric}: pattern-type differences, {mode}"
        if pairwise["model"].nunique() > 1:
            title += f" [{model}]"
        fig.suptitle(title + "\nmean fold difference with 95% CI (corrected)",
                     fontsize=10, color=ink)
        fig.tight_layout()
        suffix = f"_{model}" if pairwise["model"].nunique() > 1 else ""
        path = os.path.join(out_dir, f"diff_{mode}_{metric}{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paired significance tests of precision/recall differences "
                    "between pattern types at each seed-set size, across CV folds.")
    parser.add_argument("summary_dir", help="Folder to search recursively for summary CSVs.")
    parser.add_argument("--out-dir", default=None,
                        help=f"Default: SUMMARY_DIR/{OWN_OUTPUT_DIR_NAME}")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level applied to FDR-adjusted p-values. Default 0.05.")
    parser.add_argument("--test", choices=["corrected", "paired"], default="corrected",
                        help="Which t-test drives the `significant` flag: the Nadeau-Bengio "
                             "corrected test (default) or the ordinary paired t-test. "
                             "Both are always reported.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        help=f"Metric columns to test. Default: {' '.join(DEFAULT_METRICS)}. "
                             "Any METRIC_COLS column works, e.g. macro_precision.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.summary_dir, OWN_OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    df = load_fold_rows(args.summary_dir, args.pattern)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pairwise, omnibus = run_tests(df, args.metrics, args.alpha, args.test)

    pw_path = os.path.join(out_dir, "pairwise_tests.csv")
    om_path = os.path.join(out_dir, "omnibus_tests.csv")
    pairwise.to_csv(pw_path, index=False)
    omnibus.to_csv(om_path, index=False)
    print(f"\nwrote {pw_path} ({len(pairwise)} rows)")
    print(f"wrote {om_path} ({len(omnibus)} rows)")

    summary = write_summary(pairwise, omnibus,
                            os.path.join(out_dir, "significance_summary.txt"),
                            args.alpha, args.test)
    print("\n" + summary)

    if not args.no_plots and not pairwise.empty:
        for p in plot_differences(pairwise, out_dir):
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
