"""
Plots NOUN/VERB precision and recall against seed-list size from
category_bootstrap.py summary CSVs, producing exactly 4 plots total (one
per metric: NOUN precision, NOUN recall, VERB precision, VERB recall).

Each plot puts EVERY pattern_type and BOTH require_tag_match settings on
the same axes, without pooling/mixing them: each (mode, pattern_type)
combination gets its own line and shaded 95% CI band (color = pattern_type,
line style = require_tag_match mode), so up to 6 lines appear per plot.
The legend is placed outside the plot, to the right.

all_tagged_nouns_verbs is deliberately NOT drawn as a line - its
num_noun_seeds/num_verb_seeds aren't a swept seed-list size (that mode
ignores seed lists entirely - it's the count of distinct tagged surface
words seen in the training corpus, far larger than the curated seed
counts the other two modes sweep over). Sharing an x-axis with it would
compress the swept curves into an unreadable sliver. Instead its pooled
value (with its own 95% CI) is reported as a text annotation in the
bottom-right corner.

Handles two input layouts, auto-detected per matched file's own
containing directory (every matching CSV directly inside the SAME
directory is treated as ONE model and concatenated together before
extracting mean rows):

  1. An already-merged `summary.csv` sitting alone in a directory (the
     output of `category_bootstrap.py --merge`) - that directory is one
     model, one file.
  2. A `summary_parts/` directory containing many small per-job CSVs -
     one file per (mode, pattern_type, seed-step) job. All files
     directly inside that one directory are treated as ONE model and
     concatenated together first.

If SUMMARY_DIR contains multiple such directories (e.g. several
`<out_dir>/summary_parts/` folders, or several `<out_dir>/summary.csv`
files, one per sweep), each directory is its own model - this is how you
compare multiple independent runs on one plot. At each (mode,
pattern_type, x) point:
  - If MULTIPLE models contribute a point there, the CI is computed
    ACROSS those per-model mean values (t-distribution).
  - If only ONE model contributes that point (the common case), the band
    instead falls back to that model's own WITHIN-model cross-validation
    spread, using the <metric>_std/n_folds columns category_bootstrap.py
    already computed for that mean row. If that model was run with
    --n-folds 1, there's no fold-level spread either, so the point is
    plotted with no band at all.

"Mean row" = the row category_bootstrap.py already wrote for that
(mode, pattern_type, num_noun_seeds, num_verb_seeds) combination once
fold results were aggregated: fold=="mean" for a cross-validated run
(--n-folds > 1), or the lone row itself when the model was run with
--n-folds 1. A job that's still mid-run (not all folds finished, no
"mean" row written yet) contributes 0 rows and is silently skipped.

Any CSV that doesn't have this pipeline's summary-file columns (e.g.
confusion_words_*.csv or pattern_usage_*.csv, which can sit alongside
summary_parts/ in the same --out-dir) is skipped with a warning rather
than aborting the whole run - so --pattern can safely stay broad
("*.csv").

Usage:
    python3 plot_precision_recall.py SUMMARY_DIR [--out-dir DIR]
        [--pattern *.csv]

Requires matplotlib and scipy in addition to pandas/numpy:
    pip install matplotlib scipy pandas numpy
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


MODES = ["all_tagged_nouns_verbs", "require_tag_match_false", "require_tag_match_true"]
SWEPT_MODES = ["require_tag_match_false", "require_tag_match_true"]
ALL_TAGGED_MODE = "all_tagged_nouns_verbs"
# Color now encodes pattern_type (not mode) - each pattern_type gets its own
# color, shared across both require_tag_match settings.
PATTERN_TYPE_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
_COLOR_CYCLE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
# Line style now encodes the require_tag_match mode (not pattern_type).
MODE_LINESTYLES = {"require_tag_match_false": "-", "require_tag_match_true": "--"}
# Neutral color for the all_tagged_nouns_verbs annotation, which isn't
# drawn as a line/color-coded pattern_type at all - see module docstring.
ALL_TAGGED_ANNOTATION_COLOR = "#555555"
# Random-guess baseline (category_bootstrap.py's baseline_<metric> columns).
# It depends only on the train/test composition, not on pattern_type, so
# it is drawn once per mode, in neutral grey, without markers.
BASELINE_COLOR = "#7f7f7f"
BASELINE_LINESTYLES = {"require_tag_match_false": ":", "require_tag_match_true": "-."}


def _color_for_pattern_type(pattern_type, all_pattern_types):
    if pattern_type in PATTERN_TYPE_COLORS:
        return PATTERN_TYPE_COLORS[pattern_type]
    idx = list(all_pattern_types).index(pattern_type) % len(_COLOR_CYCLE)
    return _COLOR_CYCLE[idx]

# (metric column, x-axis column, x-axis label, y-axis label, plot title)
METRICS = [
    ("NOUN_precision", "num_noun_seeds", "Noun seed count", "Precision", "NOUN precision"),
    ("NOUN_recall", "num_noun_seeds", "Noun seed count", "Recall", "NOUN recall"),
    ("VERB_precision", "num_verb_seeds", "Verb seed count", "Precision", "VERB precision"),
    ("VERB_recall", "num_verb_seeds", "Verb seed count", "Recall", "VERB recall"),
]
REQUIRED_COLUMNS = {"fold", "n_folds", "mode", "pattern_type", "num_noun_seeds", "num_verb_seeds"}
OWN_OUTPUT_DIR_NAME = "precision_recall_plots"
# category_bootstrap.py --merge writes summary_means.csv next to summary.csv:
# a copy of summary.csv's mean rows. Reading both would count every run
# twice (and, being identical, shrink every CI band to zero), so it is
# always skipped - summary.csv already holds the same rows.
SKIP_FILE_NAMES = {"summary_means.csv"}

# Word-type count table under each plot (see draw_type_table): for each
# metric, the class whose <CLASS>_<type_col> column is shown.
TYPE_CLASS = {"NOUN_precision": "NOUN", "NOUN_recall": "NOUN",
              "VERB_precision": "VERB", "VERB_recall": "VERB"}
TYPE_COL_CHOICES = ["TP_types", "FP_types", "FN_types", "pred_types", "true_types"]
TYPE_COL_LABELS = {"TP_types": "TP types", "FP_types": "FP types", "FN_types": "FN types",
                   "pred_types": "predicted types", "true_types": "true types"}


def find_csv_files(summary_dir, pattern):
    matches = glob.glob(os.path.join(summary_dir, "**", pattern), recursive=True)
    matches += glob.glob(os.path.join(summary_dir, pattern))
    matches = {
        os.path.normpath(m) for m in matches
        if OWN_OUTPUT_DIR_NAME not in os.path.normpath(m).split(os.sep)
        and os.path.basename(m) not in SKIP_FILE_NAMES
    }
    return sorted(matches)


def group_files_by_model(paths, summary_dir):
    groups = {}
    for path in paths:
        parent_dir = os.path.normpath(os.path.dirname(path))
        label = os.path.basename(parent_dir)
        if label == "summary_parts":
            grandparent_label = os.path.basename(os.path.dirname(parent_dir))
            if grandparent_label:
                label = grandparent_label
        if not label:
            label = os.path.basename(os.path.normpath(summary_dir)) or "model"
        groups.setdefault(label, []).append(path)
    return groups


def load_mean_rows_for_model(paths, model_label):
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [{model_label}] skipping {path}: could not read as CSV ({e})")
            continue
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"  [{model_label}] skipping {path}: not a summary CSV (missing {sorted(missing)})")
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS) + ["model"])

    raw = pd.concat(frames, ignore_index=True)
    is_mean = (raw["fold"].astype(str) == "mean") | (raw["n_folds"] == 1)
    mean_rows = raw[is_mean].copy()
    mean_rows.insert(0, "model", model_label)
    return mean_rows


def combine_summaries(summary_dir, pattern):
    paths = find_csv_files(summary_dir, pattern)
    if not paths:
        raise FileNotFoundError(f"No file(s) matching '{pattern}' found under {summary_dir}")
    groups = group_files_by_model(paths, summary_dir)
    frames = []
    print(f"Found {len(paths)} CSV file(s) under {summary_dir}, grouped into {len(groups)} model(s):")
    for label, group_paths in sorted(groups.items()):
        rows = load_mean_rows_for_model(group_paths, label)
        frames.append(rows)
        print(f"  [{label}] {len(group_paths)} file(s) -> {len(rows)} mean row(s)")
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError(
            f"No usable mean rows found under {summary_dir} - every matched CSV either "
            f"didn't look like a summary file, or no job there has finished all its folds yet."
        )
    return combined


def _point_estimate(sub, metric_col):
    values = sub[metric_col].dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))

    if n >= 2:
        sem = np.std(values, ddof=1) / np.sqrt(n)
        return mean, float(sem * stats.t.ppf(0.975, df=n - 1))

    row = sub.iloc[0]
    n_folds = row.get("n_folds", 1)
    std_col = f"{metric_col}_std"
    std = row.get(std_col, np.nan)
    if pd.notna(std) and pd.notna(n_folds) and n_folds and n_folds > 1:
        sem = std / np.sqrt(n_folds)
        return mean, float(sem * stats.t.ppf(0.975, df=n_folds - 1))
    return mean, np.nan


def annotate_all_tagged(ax, df, metric_col, show_baseline=True, type_col=None, type_label=""):
    """Report all_tagged_nouns_verbs as one value PER PATTERN TYPE (each
    in its pattern_type colour, with a 95% CI from its own folds), in a box
    outside the axes, under the legend. Pattern types are kept separate
    because they can differ a lot - pooling them would give a mean that
    describes none of them and a CI across pattern types, not folds.
    With type_col (e.g. NOUN_TP_types), each line also gives that word-type
    count's fold mean."""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

    sub = df[df["mode"] == ALL_TAGGED_MODE]
    if sub.empty or sub[metric_col].isna().all():
        return
    all_pattern_types = sorted(df["pattern_type"].dropna().unique())
    lines = [TextArea(f"{ALL_TAGGED_MODE}\n(mean ± 95% CI across folds)",
                      textprops=dict(fontsize=8, color=ALL_TAGGED_ANNOTATION_COLOR))]
    for pattern_type in sorted(sub["pattern_type"].dropna().unique()):
        mean, half_width = _point_estimate(sub[sub["pattern_type"] == pattern_type], metric_col)
        if np.isnan(mean):
            continue
        pt_label = int(pattern_type) if float(pattern_type).is_integer() else pattern_type
        text = f"pattern_type={pt_label}: {mean:.3f}"
        if not np.isnan(half_width):
            text += f" ± {half_width:.3f}"
        if type_col and type_col in sub.columns:
            n_types = sub.loc[sub["pattern_type"] == pattern_type, type_col].mean()
            if pd.notna(n_types):
                text += f"  [{type_label}: {n_types:.0f}]"
        lines.append(TextArea(text, textprops=dict(
            fontsize=8, color=_color_for_pattern_type(pattern_type, all_pattern_types))))
    base_col = f"baseline_{metric_col}"
    if len(lines) > 1 and show_baseline and base_col in sub.columns and sub[base_col].notna().any():
        lines.append(TextArea(f"random baseline: {sub[base_col].mean():.3f}",
                              textprops=dict(fontsize=8, color=BASELINE_COLOR)))
    if len(lines) == 1:
        return
    box = AnchoredOffsetbox(
        loc="lower left", child=VPacker(children=lines, align="left", pad=0, sep=3),
        pad=0.4, borderpad=0, frameon=True,
        bbox_to_anchor=(1.02, 0.0), bbox_transform=ax.transAxes,
    )
    box.patch.set_boxstyle("round,pad=0.2")
    box.patch.set_edgecolor("0.7")
    ax.add_artist(box)


MODE_SHORT_LABELS = {"require_tag_match_false": "match=F", "require_tag_match_true": "match=T"}


def compute_significance(summary_dir, pattern, alpha, test_kind):
    """Run compare_pattern_types.py's paired tests on the per-fold rows.
    Returns the pairwise-results DataFrame, or None (with a printed reason)
    if significance can't be shown - no per-fold rows (--n-folds 1 runs),
    or more than one model (the strip would be ambiguous: which model's
    folds?)."""
    # Imported here, not at module top, because compare_pattern_types
    # itself imports from this module.
    import warnings
    import compare_pattern_types as cpt
    try:
        folds = cpt.load_fold_rows(summary_dir, pattern)
    except (ValueError, FileNotFoundError) as e:
        print(f"Significance strip skipped: {e}")
        return None
    if folds["model"].nunique() > 1:
        print("Significance strip skipped: more than one model found - run "
              "compare_pattern_types.py on each model's folder separately.")
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pairwise, _ = cpt.run_tests(folds, [m[0] for m in METRICS], alpha, test_kind)
    return pairwise


def draw_significance_strip(ax, sig, x_col, all_pattern_types, xlim):
    """One row per (mode, pattern-type pair). A mark at x wherever that pair
    differs significantly (FDR-adjusted) at that seed step, coloured by the
    pattern type that is BETTER there. Several steps sharing one x (verb
    counts repeat across noun steps) are spread slightly sideways."""
    rows = []
    for mode in SWEPT_MODES:
        pairs = sig[sig["mode"] == mode][["pattern_type_A", "pattern_type_B"]].drop_duplicates()
        for a, b in sorted(map(tuple, pairs.to_numpy())):
            rows.append((mode, int(a), int(b)))
    n = len(rows)
    for i, (mode, a, b) in enumerate(rows):
        y = n - 1 - i
        g = sig[(sig["mode"] == mode) & (sig["pattern_type_A"] == a) & (sig["pattern_type_B"] == b)]
        g = g.sort_values(["num_noun_seeds", "num_verb_seeds"])
        x = g[x_col].to_numpy(dtype=float)
        if g[x_col].duplicated().any():
            rank = g.groupby(x_col).cumcount().to_numpy()
            size = g.groupby(x_col)[x_col].transform("size").to_numpy()
            x = x + (rank - (size - 1) / 2) * np.minimum(0.6 / size, 0.12)
        s = g["significant"].to_numpy(dtype=bool)
        winner = np.where(g["mean_diff"].to_numpy() > 0, a, b)
        for pt in (a, b):
            m = s & (winner == pt)
            if m.any():
                ax.scatter(x[m], np.full(m.sum(), y), marker="|", s=60, linewidths=2,
                           color=_color_for_pattern_type(pt, all_pattern_types))
        ax.axhline(y, color="0.9", linewidth=0.6, zorder=0)
    # Separate the two modes' blocks.
    n_first = sum(1 for r in rows if r[0] == rows[0][0]) if rows else 0
    if 0 < n_first < n:
        ax.axhline(n - n_first - 0.5, color="0.6", linewidth=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{MODE_SHORT_LABELS.get(m, m)}  {a} vs {b}" for m, a, b in reversed(rows)],
                       fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(xlim)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return n


def select_type_xs(df, x_col, n_steps):
    """n_steps seed sizes, evenly spaced through the swept modes' distinct
    x values (first and last always included), for the type-count table."""
    xs = np.array(sorted(df.loc[df["mode"].isin(SWEPT_MODES), x_col].dropna().unique()))
    if len(xs) == 0 or n_steps <= 0:
        return []
    if n_steps >= len(xs):
        return list(xs)
    idx = np.unique(np.round(np.linspace(0, len(xs) - 1, n_steps)).astype(int))
    return list(xs[idx])


def type_table_rows(df, type_col):
    """(mode, pattern_type) pairs that have this type-count column."""
    rows = []
    for mode in SWEPT_MODES:
        for pt in sorted(df.loc[df["mode"] == mode, "pattern_type"].dropna().unique()):
            sub = df[(df["mode"] == mode) & (df["pattern_type"] == pt)]
            if type_col in sub.columns and sub[type_col].notna().any():
                rows.append((mode, pt))
    return rows


def draw_type_table(ax, df, type_col, x_col, xs, rows, all_pattern_types, xlim, show_mode):
    """A row of numbers per (mode, pattern_type): the fold-mean word-type
    count (type_col, e.g. NOUN_TP_types) at each seed size in xs, placed at
    that x so it lines up with the plot above, in the pattern type's colour.
    Steps sharing one x (verb counts repeat across noun steps) are averaged,
    as the plotted lines are."""
    n = len(rows)
    for i, (mode, pt) in enumerate(rows):
        y = n - 1 - i
        sub = df[(df["mode"] == mode) & (df["pattern_type"] == pt)]
        by_x = sub.groupby(x_col)[type_col].mean()
        color = _color_for_pattern_type(pt, all_pattern_types)
        for x in xs:
            v = by_x.get(x, np.nan)
            if pd.notna(v):
                ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=color)
    ax.set_yticks(range(n))
    labels = []
    for mode, pt in reversed(rows):
        pt_label = int(pt) if float(pt).is_integer() else pt
        labels.append((f"{MODE_SHORT_LABELS.get(mode, mode)}  " if show_mode else "") + f"pattern_type={pt_label}")
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(xlim)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def draw_baselines(ax, df, metric_col, x_col):
    """One grey line per swept mode for the random-guess baseline. Values
    are averaged over pattern types (identical anyway) and over seed steps
    sharing an x, the same way the metric lines are."""
    base_col = f"baseline_{metric_col}"
    if base_col not in df.columns:
        return
    for mode in SWEPT_MODES:
        sub = df[(df["mode"] == mode) & df[base_col].notna()]
        if sub.empty:
            continue
        line = sub.groupby(x_col)[base_col].mean().sort_index()
        ax.plot(line.index, line.values, color=BASELINE_COLOR,
                linestyle=BASELINE_LINESTYLES.get(mode, ":"), linewidth=1.5,
                label=f"random baseline ({mode})", zorder=1)


def plot_metric_by_pattern_type(df, metric_col, x_col, x_label, y_label, title, out_path,
                                sig=None, sig_note="", show_baseline=True,
                                type_col=None, type_steps=6):
    """Every pattern_type and both require_tag_match modes on ONE plot -
    each (mode, pattern_type) combination gets its own line and CI band:
    color encodes pattern_type (PATTERN_TYPE_COLORS), line style encodes
    the require_tag_match mode (MODE_LINESTYLES). Legend is placed outside
    the axes, to the right, so it never sits on top of a line.

    If type_col (e.g. "TP_types") is given and the summary has the matching
    <CLASS>_<type_col> column, a table of that word-type count's fold means
    at type_steps evenly spaced seed sizes is added underneath (see
    draw_type_table), with faint guide lines at those seed sizes.

    If `sig` (compare_pattern_types pairwise results for this metric) is
    given, a significance strip is added underneath, sharing the x-axis."""
    all_pattern_types = sorted(df["pattern_type"].dropna().unique())
    type_full = f"{TYPE_CLASS[metric_col]}_{type_col}" if type_col and metric_col in TYPE_CLASS else None
    type_rows = type_table_rows(df, type_full) if type_full else []
    type_xs = select_type_xs(df, x_col, type_steps) if type_rows else []
    show_types = bool(type_rows and type_xs)
    type_label = TYPE_COL_LABELS.get(type_col, type_col) if type_col else ""

    show_sig = sig is not None and not sig.empty
    panels = [("main", 5.5)]
    if show_types:
        panels.append(("types", 0.22 * len(type_rows) + 0.3))
    if show_sig:
        n_rows = sig[["mode", "pattern_type_A", "pattern_type_B"]].drop_duplicates().shape[0]
        panels.append(("sig", 0.22 * n_rows + 0.25))
    heights = [h for _, h in panels]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(8.5, sum(heights)), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": heights, "hspace": 0.08})
    axes_by = {name: a for (name, _), a in zip(panels, axes[:, 0])}
    ax = axes_by["main"]
    plotted_any = False

    for mode in SWEPT_MODES:
        sub_mode = df[df["mode"] == mode]
        if sub_mode.empty:
            continue
        linestyle = MODE_LINESTYLES.get(mode, "-")
        for pattern_type in all_pattern_types:
            sub = sub_mode[sub_mode["pattern_type"] == pattern_type]
            if sub.empty or sub[metric_col].isna().all():
                continue

            xs = sorted(sub[x_col].dropna().unique())
            means, halfwidths = [], []
            for x in xs:
                mean, half_width = _point_estimate(sub[sub[x_col] == x], metric_col)
                means.append(mean)
                halfwidths.append(half_width)

            color = _color_for_pattern_type(pattern_type, all_pattern_types)
            pt_label = int(pattern_type) if float(pattern_type).is_integer() else pattern_type
            label = f"{mode} (pattern_type={pt_label})"
            ax.plot(
                xs, means, label=label, color=color, linestyle=linestyle,
                marker="o", markersize=3, linewidth=1.5,
            )
            lo = [m - (0.0 if np.isnan(h) else h) for m, h in zip(means, halfwidths)]
            hi = [m + (0.0 if np.isnan(h) else h) for m, h in zip(means, halfwidths)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    if show_baseline:
        draw_baselines(ax, df, metric_col, x_col)
    annotate_all_tagged(ax, df, metric_col, show_baseline=show_baseline,
                        type_col=type_full if show_types else None, type_label=type_label)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)
    xlim = ax.get_xlim()
    if show_types:
        for x in type_xs:
            ax.axvline(x, color="0.85", linewidth=0.8, linestyle=":", zorder=0)
        ax_t = axes_by["types"]
        show_mode = len({m for m, _ in type_rows}) > 1
        draw_type_table(ax_t, df, type_full, x_col, type_xs, type_rows, all_pattern_types,
                        xlim, show_mode)
        ax_t.text(1.02, 0.5,
                  f"Mean {TYPE_CLASS[metric_col]} {type_label}\nper test fold, at the\n"
                  "dotted seed sizes",
                  transform=ax_t.transAxes, ha="left", va="center", fontsize=7, color="0.35")
    if show_sig:
        ax_sig = axes_by["sig"]
        draw_significance_strip(ax_sig, sig, x_col, all_pattern_types, xlim)
        ax_sig.text(1.02, 0.5,
                    "Significant pattern-type\ndifferences " + sig_note +
                    "\nmark colour = better pattern type"
                    "\nmatch=F/T: require_tag_match",
                    transform=ax_sig.transAxes, ha="left", va="center", fontsize=7, color="0.35")
    axes[-1, 0].set_xlabel(x_label)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def make_plots(combined, out_dir, pairwise=None, sig_note="", show_baseline=True,
               type_col="TP_types", type_steps=6):
    combined = combined[combined["mode"].isin(MODES)].copy()
    if type_col and not any(f"{c}_{type_col}" in combined.columns for c in ("NOUN", "VERB")):
        print(f"  Note: no {type_col} columns in these summaries (runs made before the type "
              "counts were added) - plotting without the type-count table.")
        type_col = None
    written = []
    for metric_col, x_col, x_label, y_label, title in METRICS:
        out_path = os.path.join(out_dir, f"{metric_col}.png")
        sig = None
        if pairwise is not None and not pairwise.empty:
            sig = pairwise[pairwise["metric"] == metric_col]
        ok = plot_metric_by_pattern_type(combined, metric_col, x_col, x_label, y_label, title,
                                         out_path, sig=sig, sig_note=sig_note,
                                         show_baseline=show_baseline,
                                         type_col=type_col, type_steps=type_steps)
        if ok:
            written.append(out_path)
            print(f"  wrote {out_path}")
        else:
            print(f"  skipped {out_path} (no data for any mode)")
    return written


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot NOUN/VERB precision and recall against seed-list size from "
            "category_bootstrap.py summary CSVs - exactly 4 plots, each "
            "showing every pattern_type and require_tag_match setting "
            "together, with a legend to the side."
        )
    )
    parser.add_argument("summary_dir", help="Folder to search recursively for summary CSVs.")
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the 4 plot PNGs. Default: SUMMARY_DIR/precision_recall_plots.",
    )
    parser.add_argument(
        "--pattern", default="*.csv",
        help="Filename glob to search for. Default: *.csv (non-summary CSVs are skipped with a warning).",
    )
    parser.add_argument(
        "--no-significance", action="store_true",
        help="Don't add the pattern-type significance strip under each plot.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for the strip (FDR-adjusted p). Default 0.05.",
    )
    parser.add_argument(
        "--test", choices=["corrected", "paired"], default="corrected",
        help="Paired t-test used for the strip: Nadeau-Bengio corrected (default) or ordinary. "
             "See compare_pattern_types.py.",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Don't draw the random-guess baseline (baseline_<metric> columns).",
    )
    parser.add_argument(
        "--type-count", choices=TYPE_COL_CHOICES, default="TP_types",
        help="Word-type count shown in the table under each plot (fold mean of "
             "<CLASS>_<this> from summary.csv). Default: TP_types. Use pred_types for the "
             "types behind precision's denominator, true_types for recall's.",
    )
    parser.add_argument(
        "--type-steps", type=int, default=6,
        help="How many seed sizes (evenly spaced, first and last included) to show type "
             "counts at. Default 6.",
    )
    parser.add_argument(
        "--no-types", action="store_true",
        help="Leave out the type-count table.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.summary_dir, "precision_recall_plots")
    os.makedirs(out_dir, exist_ok=True)

    combined = combine_summaries(args.summary_dir, args.pattern)

    unexpected_modes = sorted(set(combined["mode"].dropna()) - set(MODES))
    if unexpected_modes:
        print(f"Note: ignoring mode(s) not in {MODES} for plotting: {unexpected_modes}")

    pairwise, sig_note = None, ""
    if not args.no_significance:
        print("\nComputing pattern-type significance from per-fold rows...")
        pairwise = compute_significance(args.summary_dir, args.pattern, args.alpha, args.test)
        test_name = "corrected paired t" if args.test == "corrected" else "paired t"
        sig_note = f"\n({test_name}, BH-FDR q<{args.alpha:g})"

    print("\nGenerating plots...")
    make_plots(combined, out_dir, pairwise, sig_note, show_baseline=not args.no_baseline,
               type_col=None if args.no_types else args.type_count, type_steps=args.type_steps)


if __name__ == "__main__":
    main()
