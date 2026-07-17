import re
import random
import argparse
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import os
import time
import gc
import pandas as pd
from scipy import sparse
import inspect


def extract_context_patterns_fast(corpus, seeds, window_size=2, dtype=np.int32, pattern_type=1,
                                   corpus_tags=None, require_tag_match=False,
                                   all_tagged_nouns_verbs=False):
    """
    require_tag_match:
        False (default) - a word counts as a noun/verb whenever it appears in
            seeds['nouns']/seeds['verbs'], regardless of its corpus tag (original behavior).
        True - a word only counts as a noun/verb if it is BOTH in the corresponding
            seed list AND tagged as that category (tag starting with "N"/"V") in
            corpus_tags at that position. Requires corpus_tags to be provided and
            aligned index-for-index with corpus.

    all_tagged_nouns_verbs:
        False (default) - noun/verb status is decided by seeds (+ require_tag_match
            as above).
        True - ignore seeds entirely. Every corpus token counts as a noun/verb
            whenever its own corpus tag says so (tag starting "N"/"V"), i.e. every
            tagged noun and verb in the training corpus is used, not just seeds.
            Takes precedence over require_tag_match. Requires corpus_tags.
    """
    if (require_tag_match or all_tagged_nouns_verbs) and corpus_tags is None:
        raise ValueError("corpus_tags must be provided when require_tag_match=True or all_tagged_nouns_verbs=True")
    if (require_tag_match or all_tagged_nouns_verbs) and len(corpus_tags) != len(corpus):
        raise ValueError("corpus_tags must be the same length as corpus")

    types = sorted(set(corpus))
    types.extend(["NOUN", "VERB"])
    types_to_idx = {t: i for i, t in enumerate(types)}

    seeds = seeds or {}
    noun_set = set(seeds.get('nouns', []))
    verb_set = set(seeds.get('verbs', []))

    def is_noun(w, idx):
        if all_tagged_nouns_verbs:
            return bool(re.match(r"^N", corpus_tags[idx]))
        if w not in noun_set:
            return False
        if require_tag_match:
            return bool(re.match(r"^N", corpus_tags[idx]))
        return True

    def is_verb(w, idx):
        if all_tagged_nouns_verbs:
            return bool(re.match(r"^V", corpus_tags[idx]))
        if w not in verb_set:
            return False
        if require_tag_match:
            return bool(re.match(r"^V", corpus_tags[idx]))
        return True

    contexts = []
    context_to_idx = {}
    rows = []
    cols = []
    data = []

    for i, word in enumerate(corpus):
        if word not in ["{" , "}"]:
            begin = max(i - window_size, 0)
            end = min(i + window_size, len(corpus) - 1)
            context_indices = list(range(begin, end + 1))
            del context_indices[i-begin]

            # verb membership takes priority over noun on overlap
            context = []
            for idx in context_indices:
                w = corpus[idx]
                if is_verb(w, idx):
                    context.append("verb")
                elif is_noun(w, idx):
                    context.append("noun")
                else:
                    context.append(w)

            if len(context) == 4:
                # noun membership takes priority over verb on overlap
                
                if is_noun(word, i):
                    seed_id = types_to_idx["NOUN"]
                elif is_verb(word, i):
                    seed_id = types_to_idx["VERB"]
                else:
                    seed_id = types_to_idx[word]

                if pattern_type == 1:
                    p1 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X_" + context[2]))
                    p1a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X"))
                    for p in (p1, p1a):
                        idx = context_to_idx.get(p)
                        if idx is None:
                            idx = len(contexts)
                            contexts.append(p)
                            context_to_idx[p] = idx
                        rows.append(seed_id)
                        cols.append(idx)
                        data.append(1)
                elif pattern_type == 2:
                    p2 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", "X_" + context[2] + "_" + context[3]))
                    p2a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", "X_" + context[2]))
                    for p in (p2, p2a):
                        idx = context_to_idx.get(p)
                        if idx is None:
                            idx = len(contexts)
                            contexts.append(p)
                            context_to_idx[p] = idx
                        rows.append(seed_id)
                        cols.append(idx)
                        data.append(1)
                elif pattern_type == 3:
                    p3 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[0] + "_" + context[1] + "_X"))
                    p3a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X"))

                    for p in (p3, p3a):
                        idx = context_to_idx.get(p)
                        if idx is None:
                            idx = len(contexts)
                            contexts.append(p)
                            context_to_idx[p] = idx
                        rows.append(seed_id)
                        cols.append(idx)
                        data.append(1)
    if not data:
        return pd.DataFrame(np.zeros((len(types), 0), dtype=int), index=types, columns=[])

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    data = np.asarray(data, dtype=dtype)
    coo = coo_matrix((data, (rows, cols)), shape=(len(types), len(contexts)), dtype=dtype)
    # Keep this sparse rather than densifying: with small seed lists, context
    # words rarely get abstracted to "noun"/"verb", so the pattern space can
    # run into the hundreds of thousands of columns. Densifying that (as this
    # used to do via .toarray()) tries to allocate a dense int32 array of
    # len(types) x len(contexts), which can require several GB and crash for
    # no benefit since the matrix is almost entirely zeros.
    df = pd.DataFrame.sparse.from_spmatrix(coo, index=types, columns=contexts)

    return df


def _trim_braces_fast(s: str) -> str:
    i = s.find('{')
    if i != -1:
        j = s.rfind('}')
        if j >= i:
            return s[i:j+1]
    return s


def add_proportion(df, freq_col=None, prop_col="PROPORTION"):
    """
    Add a column giving each word's frequency as a proportion of the total
    frequency across the WHOLE dataframe passed in - i.e. the full
    vocabulary. Call this once, before any Include-based filtering, so every
    word's proportion is fixed relative to the full vocabulary rather than
    to whatever subset is later used to pick seeds.

    freq_col: name of the frequency/count column to use. If not given, looks
    for a column named "Freq" first, then "Count" (the two names this
    pipeline's seed files have used), and raises if neither is present.
    """
    if freq_col is None:
        freq_col = next((c for c in ("Freq", "Count") if c in df.columns), None)
        if freq_col is None:
            raise KeyError(
                f"Could not find a frequency column (looked for 'Freq' or 'Count') "
                f"in columns: {list(df.columns)}"
            )

    total = df[freq_col].sum()
    df[prop_col] = df[freq_col] / total
    return df


def add_cumulative_proportion(df, prop_col="PROPORTION", cum_col=None):
    """
    Compute, for each row, the cumulative proportion accounted for from the
    top of the (already-filtered) dataframe down to that row: a running sum
    of the PRECOMPUTED per-word proportion column (see add_proportion) - not
    recomputed relative to this subset. Call this on the fly, after
    filtering down to whichever words are actually eligible to be seeds
    (e.g. Include==1), so the running total only accumulates over eligible
    words while each word's individual proportion still reflects its share
    of the full vocabulary.

    Overwrites the existing cumulative-proportion column in place (whatever
    it's named - matched case-insensitively on "CUMULATIVE"), or adds a new
    "CUMULATIVE_PROPORTION" column if none exists. Row order is not changed;
    the dataframe is used exactly as it currently appears (post-filtering).
    """
    if prop_col not in df.columns:
        raise KeyError(
            f"Expected a precomputed '{prop_col}' column (see add_proportion), "
            f"got columns: {list(df.columns)}"
        )

    cum_col = cum_col or next((c for c in df.columns if "CUMULATIVE" in str(c).upper()), "CUMULATIVE_PROPORTION")
    df[cum_col] = df[prop_col].cumsum()
    return df


def run_extract_and_evaluate(
    train_tokens,
    test_tokens,
    test_tags,
    selected_noun_seeds,
    selected_verb_seeds,
    token_counts,
    sorted_noun_tokens,
    sorted_verb_tokens,
    target_prob_cutoff=0.0005,
    window_size=2, pattern_type=1,
    train_tags=None, require_tag_match=False,
    all_tagged_nouns_verbs=False,
):
    """
    Run extraction on train_tokens, categorize test_tokens (with test_tags),
    and compute strict precision/recall.
    Returns: (results, metrics, df_contexts)
    - results: list of (token, pred, true_tag) tuples from categorize_with_contexts_fast
    - metrics: output of strict_precision_recall(results)
    - df_contexts: DataFrame from extract_context_patterns_fast

    require_tag_match: if True, a training-corpus word only counts as a noun/verb
        when it is also tagged as that category in train_tags (see
        extract_context_patterns_fast). Requires train_tags to be provided.

    all_tagged_nouns_verbs: if True, seeds are ignored entirely and every
        tagged noun/verb in the training corpus (per train_tags) is used to
        extract patterns, rather than only seed words. Requires train_tags
        and test_tags. Takes precedence over require_tag_match.
    """
    seeds = {'nouns': selected_noun_seeds, 'verbs': selected_verb_seeds}

    df_contexts = extract_context_patterns_fast(
        train_tokens, seeds, window_size=window_size, pattern_type=pattern_type,
        corpus_tags=train_tags, require_tag_match=require_tag_match,
        all_tagged_nouns_verbs=all_tagged_nouns_verbs,
    )

    corpus_total = sum(token_counts.values())
    token_probs = {k: (v / corpus_total) for k, v in token_counts.items()}
    targets = [k for k, p in token_probs.items() if p < target_prob_cutoff]

    results = categorize_with_contexts_fast(
        df_contexts,
        test_tokens,
        targets,
        selected_noun_seeds,
        selected_verb_seeds,
        sorted_noun_tokens,
        sorted_verb_tokens,
        window_size=window_size,
        pattern_type=pattern_type,
        tags=test_tags,
        all_tagged_nouns_verbs=all_tagged_nouns_verbs,
    )

    metrics = strict_precision_recall(results, train_tags=train_tags)
    return metrics


def get_max_count_item(this_pattern,df):
    if this_pattern in df.columns:
        #print(this_pattern)
        counts = df[this_pattern]
        max_count = counts.max()
        # Find all categories with the max count
        max_labels = [label for label, val in counts.items() if val == max_count and max_count > 0]
        if len(max_labels) == 1:
            return(max_labels[0])
        else:
            return("OTHER")
    else:
        return None

def categorize_with_contexts_fast(df, tokens, targets,
                                  selected_noun_seeds, selected_verb_seeds,
                                  sorted_noun_tokens, sorted_verb_tokens,
                                  window_size=2, tags=None, pattern_type=1,
                                  all_tagged_nouns_verbs=False):
    """
    all_tagged_nouns_verbs: when True, context words are marked "noun"/"verb"
    based on their own corpus tag (tags[idx] starting "N"/"V") instead of
    seed-list membership - mirrors extract_context_patterns_fast's
    all_tagged_nouns_verbs mode, so patterns built that way at train time
    actually line up with contexts built at test/categorization time.
    Requires tags to be provided and aligned with tokens.
    """
    if all_tagged_nouns_verbs and tags is None:
        raise ValueError("tags must be provided when all_tagged_nouns_verbs=True")
    if all_tagged_nouns_verbs and len(tags) != len(tokens):
        raise ValueError("tags must be the same length as tokens")

    toks_to_ignore = {"{", "}"}
    token_count = len(tokens)
    #print("CALLED!")
    #print(tags)
    selected_noun_set = set(selected_noun_seeds)
    selected_verb_set = set(selected_verb_seeds)
    targets_set = set(targets)
    df_loc = df

    results = []
    trim = _trim_braces_fast
    for i, word in enumerate(tokens):
        if word.lower() in toks_to_ignore or word.lower() not in targets_set:
            continue

        begin = max(i - window_size, 0)
        end = min(i + window_size, token_count - 1)
        context_indices = list(range(begin, end + 1))
        del context_indices[i-begin]

        context = []
        for idx in context_indices:
            w = tokens[idx]
            if all_tagged_nouns_verbs:
                # verb-first, matching extract_context_patterns_fast's priority
                if re.match(r"^V", tags[idx]):
                    context.append("verb")
                elif re.match(r"^N", tags[idx]):
                    context.append("noun")
                else:
                    context.append(w)
            else:
                if w in selected_noun_set:
                    context.append("noun")
                elif w in selected_verb_set:
                    context.append("verb")
                else:
                    context.append(w)

        if len(context) != 4:
            continue
        p1 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X_" + context[2]))
        p1a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X"))
        p2 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", "X_" + context[2] + "_" + context[3]))
        p2a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", "X_" + context[2]))
        p3 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[0] + "_" + context[1] + "_X"))
        p3a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X"))

        if pattern_type == 1:
            cat = get_max_count_item(p1,df)
            if cat is None:
                cat =  get_max_count_item(p1a,df)
            if cat not in ("NOUN", "VERB"):
                cat = "OTHER"
        if pattern_type == 2:
            cat = get_max_count_item(p2,df)
            if cat is None:
                cat =  get_max_count_item(p2a,df)
            if cat not in ("NOUN", "VERB"):
                cat = "OTHER"
        if pattern_type == 3:
            cat =  get_max_count_item(p3,df)
            if cat is None:
                cat =  get_max_count_item(p3a,df)
            if cat not in ("NOUN", "VERB"):
                cat = "OTHER"
        

        # produce triple if tags provided, else pair
        if tags is not None:
            results.append((word, cat, tags[i]))
        else:
            results.append((word, cat))

    return results


SUMMARY_COLS = [
    "time", "mode", "pattern_type", "num_noun_seeds", "num_verb_seeds", "runtime_s",
    "NOUN_precision", "NOUN_recall",
    "VERB_precision", "VERB_recall",
    "macro_precision", "macro_recall",
    "micro_precision", "micro_recall",
    "baseline_NOUN_precision", "baseline_NOUN_recall",
    "baseline_VERB_precision", "baseline_VERB_recall",
    "baseline_macro_precision", "baseline_macro_recall",
    "baseline_micro_precision", "baseline_micro_recall",
]


def compute_all_tagged_counts(train_tokens, train_tags):
    """
    Returns (num_nouns, num_verbs): the count of distinct noun-/verb-tagged
    word types in the training corpus (per train_tags) - what
    all_tagged_nouns_verbs=True actually uses instead of a seed list. Used
    both by sweep_and_save_runs and the standalone single-run CLI mode, so
    that mode logs a meaningful "how many nouns/verbs" number.
    """
    if train_tags is None:
        raise ValueError("train_tags must be provided for all_tagged_nouns_verbs mode")
    actual_nouns = {w for w, t in zip(train_tokens, train_tags) if re.match(r"^N", t)}
    actual_verbs = {w for w, t in zip(train_tokens, train_tags) if re.match(r"^V", t)}
    return len(actual_nouns), len(actual_verbs)


def compute_seed_steps(noun_seeds_df, verb_seeds_df, cum_prop_threshold=0.1, max_sweep_steps=None):
    """
    Filters both seed dataframes down to Include==1 (only words eligible to
    be seeds), recomputes cumulative proportion on that filtered,
    frequency-ordered subset (see add_cumulative_proportion), then returns
    the list of (num_nouns, num_verbs) seed-list sizes used by the sweep:
    starting from the smallest size allowed by cum_prop_threshold, then
    doubling each step, capped at the full filtered seed-list size and at
    max_sweep_steps steps (None means no cap - continue until the full list
    is covered).

    Returns (steps, noun_seeds_df, verb_seeds_df) - the seed dataframes
    returned are the filtered/annotated ones, so callers can slice them
    directly, e.g. noun_seeds_df.iloc[:num_nouns]['Word'].
    """
    for label, df in (("noun_seeds_df", noun_seeds_df), ("verb_seeds_df", verb_seeds_df)):
        if "Include" not in df.columns:
            raise KeyError(f"{label} has no 'Include' column: {list(df.columns)}")

    noun_seeds_df = noun_seeds_df[noun_seeds_df["Include"] == 1].reset_index(drop=True)
    verb_seeds_df = verb_seeds_df[verb_seeds_df["Include"] == 1].reset_index(drop=True)
    noun_seeds_df = add_cumulative_proportion(noun_seeds_df)
    verb_seeds_df = add_cumulative_proportion(verb_seeds_df)

    def _find_cumcol(df):
        for c in df.columns:
            if "CUMULATIVE" in str(c).upper():
                return c
        raise KeyError("No cumulative-proportion column found")

    m_col = _find_cumcol(noun_seeds_df)
    n_col = _find_cumcol(verb_seeds_df)
    base_m = max(1, int((noun_seeds_df[m_col] < cum_prop_threshold).sum()))
    base_n = max(1, int((verb_seeds_df[n_col] < cum_prop_threshold).sum()))

    total_noun = len(noun_seeds_df)
    total_verb = len(verb_seeds_df)

    steps = []
    multiplier = 1
    seen = set()
    step = 0
    while True:
        if max_sweep_steps is not None and step >= max_sweep_steps:
            break
        num_nouns = min(total_noun, base_m * multiplier)
        num_verbs = min(total_verb, base_n * multiplier)
        if (num_nouns, num_verbs) in seen:
            break
        seen.add((num_nouns, num_verbs))
        steps.append((num_nouns, num_verbs))
        step += 1
        if num_nouns >= total_noun and num_verbs >= total_verb:
            break
        multiplier *= 2

    return steps, noun_seeds_df, verb_seeds_df


def evaluate_single_run(
    run_fn, train_tokens, test_tokens, test_tags,
    selected_nouns, selected_verbs, num_nouns, num_verbs,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    target_prob_cutoff=0.0005, window_size=2, pattern_type=1,
    train_tags=None, require_tag_match=False, all_tagged_nouns_verbs=False,
    run_mode="run",
):
    """
    Runs a single (mode, pattern_type, seed-set) configuration and returns
    everything needed to log it - (row, confusion_text, confusion_words) -
    WITHOUT writing to any file. This is the atomic unit of work shared by:
      - sweep_and_save_runs, which appends the result to a shared
        summary.csv/confusion_matrices.txt (safe since it runs sequentially
        in a single process), and
      - the standalone single-job CLI mode (--mode ...), which writes the
        result to its own uniquely-named per-run files instead, so many
        instances of this script can run in parallel - e.g. one per core on
        a cluster (see run_cluster.sh) - without corrupting a shared file
        through concurrent writes.
    """
    print(f"\nRunning [{run_mode}] pattern_type={pattern_type} with num_noun_seeds={num_nouns}, num_verb_seeds={num_verbs}...")
    t0 = time.time()
    metrics = run_fn(
        train_tokens, test_tokens, test_tags,
        selected_nouns, selected_verbs,
        token_counts, sorted_noun_tokens, sorted_verb_tokens,
        target_prob_cutoff=target_prob_cutoff, window_size=window_size,
        pattern_type=pattern_type,
        train_tags=train_tags, require_tag_match=require_tag_match,
        all_tagged_nouns_verbs=all_tagged_nouns_verbs,
    )
    t1 = time.time()

    per_class = metrics['per_class']
    macro = metrics['macro']
    micro = metrics['micro']
    confusion = metrics['confusion']
    confusion_words = metrics.get('confusion_words')
    baseline = metrics.get('baseline', {})

    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
        "mode": run_mode,
        "pattern_type": pattern_type,
        "num_noun_seeds": num_nouns,
        "num_verb_seeds": num_verbs,
        "runtime_s": t1 - t0,
        "NOUN_precision": float(per_class.loc['NOUN', 'precision']),
        "NOUN_recall": float(per_class.loc['NOUN', 'recall']),
        "VERB_precision": float(per_class.loc['VERB', 'precision']),
        "VERB_recall": float(per_class.loc['VERB', 'recall']),
        "macro_precision": float(macro['precision']),
        "macro_recall": float(macro['recall']),
        "micro_precision": float(micro['precision']),
        "micro_recall": float(micro['recall']),
        "baseline_NOUN_precision": float(baseline.get('NOUN_precision', 0.0)),
        "baseline_NOUN_recall": float(baseline.get('NOUN_recall', 0.0)),
        "baseline_VERB_precision": float(baseline.get('VERB_precision', 0.0)),
        "baseline_VERB_recall": float(baseline.get('VERB_recall', 0.0)),
        "baseline_macro_precision": float(baseline.get('macro_precision', 0.0)),
        "baseline_macro_recall": float(baseline.get('macro_recall', 0.0)),
        "baseline_micro_precision": float(baseline.get('micro_precision', 0.0)),
        "baseline_micro_recall": float(baseline.get('micro_recall', 0.0)),
    }

    confusion_text = (
        f"mode={run_mode}, pattern_type={pattern_type}, "
        f"num_noun_seeds={num_nouns}, num_verb_seeds={num_verbs}, time={row['time']}\n"
        + confusion.to_string() + "\n\n"
    )

    return row, confusion_text, confusion_words


def sweep_and_save_runs(
    run_fn, train_tokens, test_tokens, test_tags,
    noun_seeds_df, verb_seeds_df,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    out_dir="sweep_runs",
    cum_prop_threshold=0.1,
    target_prob_cutoff=0.0005,
    window_size=2, pattern_type=1,
    train_tags=None, require_tag_match=False,
    all_tagged_nouns_verbs=False,
    force_full_seeds=False,
    max_sweep_steps=None,
    run_mode=None,
):
    """
    force_full_seeds: run a single pass using the entire (Include==1) seed
        list, rather than sweeping over increasing seed-list sizes. Ignored
        (implied True) when all_tagged_nouns_verbs=True.

    max_sweep_steps: cap the seed-count sweep to at most this many steps
        (e.g. 6 for "the first 6 different seed sets"), instead of doubling
        all the way up to the full seed list. None (default) means no cap -
        sweep until the full list is covered, as before. Ignored when
        all_tagged_nouns_verbs=True or force_full_seeds=True (both are
        already single-pass).

    run_mode: a short label identifying what kind of run this is (e.g.
        "all_tagged_nouns_verbs", "full_seeds", "require_tag_match_true",
        "require_tag_match_false"), recorded in a "mode" column in
        summary.csv, in the confusion_matrices.txt entry header, and in the
        confusion-words CSV filename. If not given, it's derived from
        all_tagged_nouns_verbs/require_tag_match/force_full_seeds.
    """
    if run_mode is None:
        if all_tagged_nouns_verbs:
            run_mode = "all_tagged_nouns_verbs"
        elif force_full_seeds:
            run_mode = "full_seeds"
        elif require_tag_match:
            run_mode = "require_tag_match_true"
        else:
            run_mode = "require_tag_match_false"
    run_mode_safe = re.sub(r"[^A-Za-z0-9_-]+", "_", run_mode)

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    confusion_path = os.path.join(out_dir, "confusion_matrices.txt")

    # Only words marked Include==1 are eligible to be used as seeds. See
    # compute_seed_steps for the filtering/cumulative-proportion/doubling
    # logic - it's shared with the standalone single-job CLI mode so the
    # exact same seed-count progression is available to both.
    steps, noun_seeds_df, verb_seeds_df = compute_seed_steps(
        noun_seeds_df, verb_seeds_df,
        cum_prop_threshold=cum_prop_threshold, max_sweep_steps=max_sweep_steps,
    )
    total_noun = len(noun_seeds_df)
    total_verb = len(verb_seeds_df)

    if not os.path.exists(summary_path):
        pd.DataFrame(columns=SUMMARY_COLS).to_csv(summary_path, index=False)
    else:
        existing_header = pd.read_csv(summary_path, nrows=0).columns.tolist()
        if existing_header != SUMMARY_COLS:
            raise ValueError(
                f"{summary_path} already exists with columns {existing_header}, "
                f"which don't match the current schema {SUMMARY_COLS} (this "
                f"schema now includes 'mode', 'pattern_type' and 'baseline_*' "
                f"columns). Move, rename, or delete the old file, or point "
                f"out_dir somewhere new, before re-running."
            )

    def _log(row, confusion_text, confusion_words, num_nouns, num_verbs):
        pd.DataFrame([row]).to_csv(summary_path, mode="a", header=False, index=False)

        with open(confusion_path, "a", encoding="utf-8") as f:
            f.write(confusion_text)

        if confusion_words is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            words_csv_path = os.path.join(
                out_dir,
                f"confusion_words_{run_mode_safe}_p{pattern_type}_n{num_nouns}_v{num_verbs}_{ts}.csv",
            )
            confusion_words.to_csv(words_csv_path)
            print(f"Word-level confusion breakdown written to {words_csv_path}")

    def _run_and_log(selected_nouns, selected_verbs, num_nouns, num_verbs):
        row, confusion_text, confusion_words = evaluate_single_run(
            run_fn, train_tokens, test_tokens, test_tags,
            selected_nouns, selected_verbs, num_nouns, num_verbs,
            token_counts, sorted_noun_tokens, sorted_verb_tokens,
            target_prob_cutoff=target_prob_cutoff, window_size=window_size, pattern_type=pattern_type,
            train_tags=train_tags, require_tag_match=require_tag_match,
            all_tagged_nouns_verbs=all_tagged_nouns_verbs, run_mode=run_mode,
        )
        _log(row, confusion_text, confusion_words, num_nouns, num_verbs)

    if all_tagged_nouns_verbs or force_full_seeds:
        # Single full pass, no sweep over increasing seed-list sizes.
        selected_nouns = noun_seeds_df['Word'].tolist()
        selected_verbs = verb_seeds_df['Word'].tolist()
        if all_tagged_nouns_verbs:
            # Seed identity is irrelevant in this mode - every tagged
            # noun/verb in the training corpus is used - so the seed-list
            # sizes (total_noun/total_verb) would be a misleading thing to
            # log. Log the actual count of distinct noun-/verb-tagged word
            # types found in the training corpus instead.
            num_nouns_display, num_verbs_display = compute_all_tagged_counts(train_tokens, train_tags)
        else:
            # force_full_seeds: seeds ARE what's driving noun/verb status
            # here, so log the actual full seed-list sizes used.
            num_nouns_display, num_verbs_display = total_noun, total_verb
        _run_and_log(selected_nouns, selected_verbs, num_nouns_display, num_verbs_display)
        return summary_path, confusion_path

    for num_nouns, num_verbs in steps:
        selected_nouns = noun_seeds_df.iloc[:num_nouns]['Word'].tolist()
        selected_verbs = verb_seeds_df.iloc[:num_verbs]['Word'].tolist()
        _run_and_log(selected_nouns, selected_verbs, num_nouns, num_verbs)

    return summary_path, confusion_path


def run_mode_comparison(
    run_fn, train_tokens, test_tokens, test_tags,
    noun_seeds_df, verb_seeds_df,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    out_dir="sweep_runs",
    cum_prop_threshold=0.1,
    target_prob_cutoff=0.0005,
    window_size=2,
    pattern_types=(1, 2, 3),
    train_tags=None,
    num_sweep_steps=6,
):
    """
    For EACH pattern_type in pattern_types (all three by default), runs
    three passes in sequence, sharing the same summary.csv/
    confusion_matrices.txt (distinguished by "mode" and "pattern_type"
    columns/labels) and out_dir for the confusion-words CSVs:

      1. all_tagged_nouns_verbs=True - one full run using every tagged
         noun/verb in the training corpus (mode="all_tagged_nouns_verbs").
      2. require_tag_match=True, swept across num_sweep_steps seed-list
         sizes, starting from the smallest allowed (base_m/base_n, from
         cum_prop_threshold) and doubling num_sweep_steps-1 more times - i.e.
         num_sweep_steps=6 gives 6 runs (mode="require_tag_match_true").
      3. require_tag_match=False, swept across the SAME num_sweep_steps
         seed-list sizes (mode="require_tag_match_false"). "Same" holds
         because the seed-count progression (base_m/base_n and the doubling
         sequence) is derived purely from noun_seeds_df/verb_seeds_df +
         cum_prop_threshold, which are identical across passes 2 and 3 (and
         across pattern types, since pattern_type doesn't affect seed
         selection).

    So with the default pattern_types=(1, 2, 3), this runs 3 * (1 + 2 *
    num_sweep_steps) passes total. pattern_type is recorded in its own
    summary.csv column, in the confusion_matrices.txt entry header, and in
    the confusion-words CSV filename, so every row/entry is traceable to
    exactly which (pattern_type, mode, seed-set) combination produced it.

    Returns the (summary_path, confusion_path) from the very last pass.
    """
    common = dict(
        run_fn=run_fn, train_tokens=train_tokens, test_tokens=test_tokens, test_tags=test_tags,
        noun_seeds_df=noun_seeds_df, verb_seeds_df=verb_seeds_df,
        token_counts=token_counts, sorted_noun_tokens=sorted_noun_tokens, sorted_verb_tokens=sorted_verb_tokens,
        out_dir=out_dir, cum_prop_threshold=cum_prop_threshold,
        target_prob_cutoff=target_prob_cutoff, window_size=window_size,
        train_tags=train_tags,
    )

    summary_path = confusion_path = None
    for pattern_type in pattern_types:
        sweep_and_save_runs(
            all_tagged_nouns_verbs=True, run_mode="all_tagged_nouns_verbs",
            pattern_type=pattern_type, **common,
        )
        sweep_and_save_runs(
            all_tagged_nouns_verbs=False, require_tag_match=True, max_sweep_steps=num_sweep_steps,
            run_mode="require_tag_match_true", pattern_type=pattern_type, **common,
        )
        summary_path, confusion_path = sweep_and_save_runs(
            all_tagged_nouns_verbs=False, require_tag_match=False, max_sweep_steps=num_sweep_steps,
            run_mode="require_tag_match_false", pattern_type=pattern_type, **common,
        )

    return summary_path, confusion_path

def baseline_random_scores(true_mapped, guess_probs=None):
    """
    Expected precision/recall/F1 of a baseline that assigns NOUN/VERB/OTHER
    to each instance in true_mapped independently at random, with
    probabilities guess_probs (a dict with keys 'NOUN'/'VERB'/'OTHER').
    Computed analytically in closed form (no simulation, no confusion
    matrix), using the same scoring convention as strict_precision_recall: a
    predicted OTHER is never counted as a TP, and every true-OTHER instance
    counts fully as a miss (FN) regardless of what was predicted.

    true_mapped: an iterable of 'NOUN'/'VERB'/'OTHER' labels (already
    collapsed - anything not NOUN/VERB should already be 'OTHER') for the
    set actually being scored (i.e. the test set) - this determines n and
    the per-class true counts that TP/FP/FN are computed against.

    guess_probs: dict of guess probabilities for 'NOUN'/'VERB'/'OTHER'
    (should sum to ~1). This is deliberately independent of true_mapped, so
    the guess distribution can come from a different set than the one being
    scored - e.g. guess in proportion to the TRAINING set's tag frequencies,
    then score those guesses against the test set's actual labels. If None,
    defaults to true_mapped's own empirical frequency (guess in proportion
    to the scored set's own tag frequencies).

    Returns a flat dict: NOUN_precision, NOUN_recall, VERB_precision,
    VERB_recall, macro_precision, macro_recall, micro_precision,
    micro_recall.
    """
    true_mapped = pd.Series(list(true_mapped))
    keys = ['NOUN_precision', 'NOUN_recall', 'VERB_precision', 'VERB_recall',
            'macro_precision', 'macro_recall', 'micro_precision', 'micro_recall']
    n = len(true_mapped)
    if n == 0:
        return {k: 0.0 for k in keys}

    counts = true_mapped.value_counts()
    n_noun = int(counts.get('NOUN', 0))
    n_verb = int(counts.get('VERB', 0))
    n_other = int(counts.get('OTHER', 0))
    if guess_probs is None:
        p_noun = n_noun / n
        p_verb = n_verb / n
        p_other = n_other / n
    else:
        p_noun = guess_probs.get('NOUN', 0.0)
        p_verb = guess_probs.get('VERB', 0.0)
        p_other = guess_probs.get('OTHER', 0.0)

    def class_stats(n_true_l, p_l):
        TP = n_true_l * p_l
        FP = (n - n_true_l) * p_l
        FN = n_true_l * (1 - p_l)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return TP, FP, FN, prec, rec

    TP_n, FP_n, FN_n, prec_n, rec_n = class_stats(n_noun, p_noun)
    TP_v, FP_v, FN_v, prec_v, rec_v = class_stats(n_verb, p_verb)

    # OTHER follows the same "never a TP, every true-OTHER is a miss"
    # convention as strict_precision_recall - its own precision/recall are
    # therefore not meaningful (and not returned here), but its FP/FN still
    # feed into the micro totals below.
    FP_o = (n - n_other) * p_other
    FN_o = n_other

    TP_sum = TP_n + TP_v
    FP_sum = FP_n + FP_v + FP_o
    FN_sum = FN_n + FN_v + FN_o
    micro_p = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0.0
    micro_r = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0.0

    macro_p = (prec_n + prec_v) / 2
    macro_r = (rec_n + rec_v) / 2

    return {
        'NOUN_precision': prec_n, 'NOUN_recall': rec_n,
        'VERB_precision': prec_v, 'VERB_recall': rec_v,
        'macro_precision': macro_p, 'macro_recall': macro_r,
        'micro_precision': micro_p, 'micro_recall': micro_r,
    }


def strict_precision_recall(results, train_tags=None):
    """
    Scoring per your rules:
      - map preds -> 'NOUN'/'VERB' else 'OTHER'
      - map trues -> 'NOUN'/'VERB' else 'OTHER'
      - For class L in {NOUN, VERB}:
          TP = pred==L and true==L
          FP = pred==L and true!=L
          FN = true==L and pred!=L
      - Predictions == OTHER are never counted as TP.
    Precision/recall/F1 (per_class, micro, macro) are computed on the
    collapsed 3-way NOUN/VERB/OTHER split, as before. The returned
    'confusion' matrix keeps the original OTHER column: rows are the actual
    corpus tags (PRON, DET, ADJ, NOUN, VERB, ...), not collapsed, but columns
    stay NOUN/VERB/OTHER. A separate, more granular breakdown is returned as
    'confusion_words': same rows, but every word the categorizer didn't put
    in NOUN/VERB gets its own column instead of being lumped into OTHER.
    Also returns 'baseline': the scores a random guesser would get if it
    guessed NOUN/VERB/OTHER in proportion to the TRAINING set's tag
    frequencies (train_tags), scored against this run's actual test-set
    labels (see baseline_random_scores) - no confusion matrix is built for
    it. If train_tags isn't provided, falls back to guessing in proportion
    to the test set's own tag frequencies instead.
    Returns dict {per_class, micro, macro, confusion, confusion_words, baseline}.
    """
    df = pd.DataFrame(results, columns=['token','pred','true'])
    df['pred_mapped'] = df['pred'].where(df['pred'].isin(['NOUN','VERB']), 'OTHER')
    df['true_mapped'] = df['true'].where(df['true'].isin(['NOUN','VERB']), 'OTHER')

    labels = ['NOUN', 'VERB', 'OTHER']

    # Reporting confusion matrix: true axis uses the raw corpus tag (not
    # collapsed), pred axis stays collapsed to NOUN/VERB/OTHER. Rows ordered
    # by frequency, most common tag first.
    detailed_confusion = pd.crosstab(df['true'], df['pred_mapped']).reindex(columns=labels, fill_value=0)
    detailed_confusion = detailed_confusion.loc[detailed_confusion.sum(axis=1).sort_values(ascending=False).index]

    # Word-level breakdown: same true-tag rows, but pred axis is NOUN/VERB
    # where predicted as such, else the literal word, so OTHER predictions
    # are broken out per word rather than collapsed into one column.
    df['pred_expanded'] = df['pred'].where(df['pred'].isin(['NOUN', 'VERB']), df['token'])
    confusion_words = pd.crosstab(df['true'], df['pred_expanded'])
    noun_verb_cols = [c for c in ('NOUN', 'VERB') if c in confusion_words.columns]
    word_cols = [c for c in confusion_words.columns if c not in ('NOUN', 'VERB')]
    word_cols = confusion_words[word_cols].sum(axis=0).sort_values(ascending=False).index.tolist()
    confusion_words = confusion_words[noun_verb_cols + word_cols]
    confusion_words = confusion_words.loc[confusion_words.sum(axis=1).sort_values(ascending=False).index]

    # Collapsed confusion matrix, used only to compute precision/recall/F1.
    conf = pd.crosstab(df['true_mapped'], df['pred_mapped']).reindex(index=labels, columns=labels, fill_value=0)

    classes = ['NOUN','VERB']
    rows = []
    for cls in classes:
        TP = int(conf.at[cls, cls])
        FP = int(conf[cls].sum() - TP)
        FN = int(conf.loc[cls].sum() - TP)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append((cls, TP, FP, FN, prec, rec, f1))

    # include OTHER row for completeness (OTHER has TP=0 by rule)
    TP_o = 0
    FP_o = int(conf['OTHER'].sum() - conf.at['OTHER','OTHER']) if 'OTHER' in conf.columns else 0
    FN_o = int(conf.loc['OTHER'].sum()) if 'OTHER' in conf.index else 0
    prec_o = 0.0
    rec_o = 0.0
    f1_o = 0.0
    rows.append(('OTHER', TP_o, FP_o, FN_o, prec_o, rec_o, f1_o))

    per_class = pd.DataFrame(rows, columns=['label','TP','FP','FN','precision','recall','f1']).set_index('label')

    TP_sum = per_class['TP'].sum()
    FP_sum = per_class['FP'].sum()
    FN_sum = per_class['FN'].sum()
    micro_p = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0.0
    micro_r = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    macro_p = per_class.loc[classes, 'precision'].mean()
    macro_r = per_class.loc[classes, 'recall'].mean()
    macro_f = per_class.loc[classes, 'f1'].mean()

    guess_probs = None
    if train_tags is not None:
        # Exclude "{"/"}" sentence-boundary markers (tagged BOS/EOS) - they
        # aren't real word tokens and are likewise excluded on the test side
        # (categorize_with_contexts_fast never scores them), so leaving them
        # in here would inflate OTHER relative to the test set's convention.
        train_true = pd.Series([t for t in train_tags if t not in ('BOS', 'EOS')])
        if len(train_true) > 0:
            train_mapped = train_true.where(train_true.isin(['NOUN', 'VERB']), 'OTHER')
            train_counts = train_mapped.value_counts()
            n_train = len(train_mapped)
            guess_probs = {
                'NOUN': train_counts.get('NOUN', 0) / n_train,
                'VERB': train_counts.get('VERB', 0) / n_train,
                'OTHER': train_counts.get('OTHER', 0) / n_train,
            }
    baseline = baseline_random_scores(df['true_mapped'], guess_probs=guess_probs)

    print(detailed_confusion)
    return {
        'per_class': per_class,
        'micro': {'precision': micro_p, 'recall': micro_r, 'f1': micro_f},
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f},
        'confusion': detailed_confusion,
        'confusion_words': confusion_words,
        'baseline': baseline,
    }

### RUNTIME CODE STARTS HERE ###

def load_corpus_and_split(corpus_file, split_seed=42, test_fraction=0.2):
    """
    Reads the WORD_LEMMA_TAG corpus file, builds the flat token/tag lists
    (lemma-lowercased, sentence-bounded by "{"/"}"), then splits it into
    train/test by randomly holding out test_fraction of sentences (seeded by
    split_seed, so the same split is reproduced deterministically by every
    process that calls this with the same arguments - this is what lets
    run_cluster.sh dispatch each (mode, pattern_type, seed-set) combination
    to its own independent process/core without sharing any state: each
    process just redoes this same deterministic corpus load+split itself).

    Returns (train, test, train_tags, test_tags, token_counts,
    sorted_noun_tokens, sorted_verb_tokens).
    """
    noun_tokens = defaultdict(int)
    verb_tokens = defaultdict(int)
    token_counts = defaultdict(int)
    tokens = []
    tags = []
    with open(corpus_file) as file:
        for line in file:
            tokens.append("{")
            tags.append("BOS")
            line_array = line.split()
            for element in line_array:
                # File format is WORD_LEMMA_TAG (e.g. "thought_think_VERB");
                # use the lemma (middle field), matching how the seed lists
                # in noun_selection.xlsx/verb_selection.xlsx were built in
                # from_tagged_corpus_to_seeds.py. Splitting only on the last
                # underscore would glue WORD_LEMMA together and never match
                # any seed.
                la = re.match(r"[^ ]+_([^ ]+)_([^ ]+)", element)
                w = la.group(1)
                t = la.group(2)
                tokens.append(str.lower(w))
                tags.append(t)
                token_counts[str.lower(w)] += 1
                if re.match(r"^N", t):
                    noun_tokens[str.lower(w)] += 1
                if re.match(r"^V", t):
                    verb_tokens[str.lower(w)] += 1
            tokens.append("}")
            tags.append("EOS")

    sorted_noun_counts = sorted(noun_tokens.items(), key=lambda item: item[1], reverse=True)
    sorted_verb_counts = sorted(verb_tokens.items(), key=lambda item: item[1], reverse=True)
    sorted_noun_tokens = [x for x, _ in sorted_noun_counts]
    sorted_verb_tokens = [x for x, _ in sorted_verb_counts]
    excluded_nouns = ["mummy", "daddy", "john", "carl", "dominic"]
    excluded_verbs = [""]
    sorted_noun_tokens = [x for x in sorted_noun_tokens if x not in excluded_nouns and noun_tokens[x] > verb_tokens[x]]
    sorted_verb_tokens = [x for x in sorted_verb_tokens if x not in excluded_verbs and verb_tokens[x] > noun_tokens[x]]

    # Split by utterance (sentence), selecting a random test_fraction of
    # sentences for test. Find all indices where a sentence ends ("}"), then
    # derive (start, end) bounds for each sentence (a sentence runs from
    # just after the previous "}" through its own "}", inclusive).
    random.seed(split_seed)
    sentence_end_indices = [i for i, tok in enumerate(tokens) if tok == "}"]
    sentence_bounds = []
    start = 0
    for end in sentence_end_indices:
        sentence_bounds.append((start, end))
        start = end + 1

    n_sentences = len(sentence_bounds)
    n_test = int(n_sentences * test_fraction)
    test_sentence_idx = set(random.sample(range(n_sentences), n_test))

    train, test, train_tags, test_tags = [], [], [], []
    for i, (s, e) in enumerate(sentence_bounds):
        if i in test_sentence_idx:
            test.extend(tokens[s:e + 1])
            test_tags.extend(tags[s:e + 1])
        else:
            train.extend(tokens[s:e + 1])
            train_tags.extend(tags[s:e + 1])

    return train, test, train_tags, test_tags, token_counts, sorted_noun_tokens, sorted_verb_tokens


def merge_parts(out_dir):
    """
    Combines the per-job outputs written by single-job CLI runs (--mode ...)
    under out_dir/summary_parts/*.csv and out_dir/confusion_parts/*.txt into
    the final out_dir/summary.csv and out_dir/confusion_matrices.txt - this
    is the step run_cluster.sh runs once, after all its parallel jobs finish.
    """
    summary_path = os.path.join(out_dir, "summary.csv")
    confusion_path = os.path.join(out_dir, "confusion_matrices.txt")
    parts_dir = os.path.join(out_dir, "summary_parts")
    conf_parts_dir = os.path.join(out_dir, "confusion_parts")

    part_files = sorted(glob.glob(os.path.join(parts_dir, "*.csv")))
    if part_files:
        parts_df = pd.concat([pd.read_csv(p) for p in part_files], ignore_index=True)
        parts_df = parts_df[SUMMARY_COLS]
        if not os.path.exists(summary_path):
            parts_df.to_csv(summary_path, index=False)
        else:
            existing_header = pd.read_csv(summary_path, nrows=0).columns.tolist()
            if existing_header != SUMMARY_COLS:
                raise ValueError(
                    f"{summary_path} already exists with columns {existing_header}, "
                    f"which don't match the current schema {SUMMARY_COLS}. Move, "
                    f"rename, or delete the old file, or point out_dir somewhere "
                    f"new, before merging."
                )
            parts_df.to_csv(summary_path, mode="a", header=False, index=False)
        print(f"Merged {len(part_files)} summary part(s) ({len(parts_df)} row(s)) into {summary_path}")
    else:
        print(f"No summary parts found in {parts_dir}")

    conf_part_files = sorted(glob.glob(os.path.join(conf_parts_dir, "*.txt")))
    if conf_part_files:
        with open(confusion_path, "a", encoding="utf-8") as out_f:
            for p in conf_part_files:
                with open(p, encoding="utf-8") as in_f:
                    out_f.write(in_f.read())
        print(f"Merged {len(conf_part_files)} confusion part(s) into {confusion_path}")
    else:
        print(f"No confusion parts found in {conf_parts_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract noun/verb context patterns from a tagged corpus and evaluate "
            "them. With no arguments, runs the full in-process mode/pattern-type "
            "comparison sequentially, writing straight to <out-dir>/summary.csv and "
            "confusion_matrices.txt. Pass --mode to run exactly ONE configuration "
            "instead: this is the unit of work run_cluster.sh dispatches in "
            "parallel, one process per (pattern_type, mode, seed-step) combination, "
            "each writing its own uniquely-named file under "
            "<out-dir>/summary_parts/ and <out-dir>/confusion_parts/ rather than a "
            "shared file, to avoid concurrent-write corruption. Pass --merge "
            "afterwards to combine those parts into the final summary.csv/"
            "confusion_matrices.txt."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["all_tagged_nouns_verbs", "require_tag_match_true", "require_tag_match_false"],
        default=None,
        help="Run exactly this one mode as a single job, instead of the full in-process comparison.",
    )
    parser.add_argument("--pattern-type", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument(
        "--seed-step", type=int, default=None,
        help="0-indexed seed-set step to run. Required for --mode require_tag_match_true/"
             "require_tag_match_false; ignored for all_tagged_nouns_verbs.",
    )
    parser.add_argument("--num-sweep-steps", type=int, default=6)
    parser.add_argument("--cum-prop-threshold", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--out-dir", default="sweep_out")
    parser.add_argument("--corpus-file", default="manchester_input_tagged_trf_word_and_lemma_postprocessed.txt")
    parser.add_argument("--noun-seeds-file", default="noun_selection.xlsx")
    parser.add_argument("--verb-seeds-file", default="verb_selection.xlsx")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge <out-dir>/summary_parts and confusion_parts into the final "
             "summary.csv/confusion_matrices.txt, then exit (skips everything else).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.merge:
        merge_parts(args.out_dir)
        return

    (train, test, train_tags, test_tags, token_counts,
     sorted_noun_tokens, sorted_verb_tokens) = load_corpus_and_split(
        args.corpus_file, split_seed=args.split_seed, test_fraction=args.test_fraction,
    )

    noun_seeds = pd.read_excel(args.noun_seeds_file)
    verb_seeds = pd.read_excel(args.verb_seeds_file)
    # Precompute each word's proportion of the FULL vocabulary once, up
    # front. The cumulative proportion used to pick seeds is computed later,
    # on the fly (see compute_seed_steps), after filtering down to
    # Include==1 words.
    noun_seeds = add_proportion(noun_seeds)
    verb_seeds = add_proportion(verb_seeds)

    if args.mode is None:
        # Original single-machine behavior: run the full in-process
        # comparison across all three modes and all three pattern types,
        # sequentially, writing straight to the shared summary.csv/
        # confusion_matrices.txt.
        summary_csv = run_mode_comparison(
            run_extract_and_evaluate,
            train, test, test_tags,
            noun_seeds, verb_seeds,
            token_counts, sorted_noun_tokens, sorted_verb_tokens,
            out_dir=args.out_dir,
            pattern_types=(1, 2, 3),
            train_tags=train_tags,
            num_sweep_steps=args.num_sweep_steps,
            cum_prop_threshold=args.cum_prop_threshold,
            window_size=args.window_size,
        )
        print("summary written to", summary_csv)
        return

    # Single-job mode: run exactly one (pattern_type, mode, seed-step)
    # configuration and write it to its own per-run files under
    # summary_parts/ and confusion_parts/ - safe to run in parallel across
    # many processes/cores (see run_cluster.sh), since no file is shared.
    os.makedirs(args.out_dir, exist_ok=True)
    parts_dir = os.path.join(args.out_dir, "summary_parts")
    conf_parts_dir = os.path.join(args.out_dir, "confusion_parts")
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(conf_parts_dir, exist_ok=True)

    if args.mode == "all_tagged_nouns_verbs":
        _, noun_seeds_f, verb_seeds_f = compute_seed_steps(
            noun_seeds, verb_seeds, cum_prop_threshold=args.cum_prop_threshold,
        )
        selected_nouns = noun_seeds_f['Word'].tolist()
        selected_verbs = verb_seeds_f['Word'].tolist()
        num_nouns, num_verbs = compute_all_tagged_counts(train, train_tags)
        require_tag_match = False
        all_tagged = True
        step_label = "full"
    else:
        require_tag_match = (args.mode == "require_tag_match_true")
        all_tagged = False
        if args.seed_step is None:
            raise SystemExit(f"--seed-step is required for --mode {args.mode}")
        steps, noun_seeds_f, verb_seeds_f = compute_seed_steps(
            noun_seeds, verb_seeds, cum_prop_threshold=args.cum_prop_threshold,
            max_sweep_steps=args.num_sweep_steps,
        )
        if not (0 <= args.seed_step < len(steps)):
            raise SystemExit(
                f"--seed-step {args.seed_step} out of range: only {len(steps)} "
                f"step(s) available for this seed list/threshold/--num-sweep-steps"
            )
        num_nouns, num_verbs = steps[args.seed_step]
        selected_nouns = noun_seeds_f.iloc[:num_nouns]['Word'].tolist()
        selected_verbs = verb_seeds_f.iloc[:num_verbs]['Word'].tolist()
        step_label = f"step{args.seed_step}"

    row, confusion_text, confusion_words = evaluate_single_run(
        run_extract_and_evaluate, train, test, test_tags,
        selected_nouns, selected_verbs, num_nouns, num_verbs,
        token_counts, sorted_noun_tokens, sorted_verb_tokens,
        window_size=args.window_size, pattern_type=args.pattern_type,
        train_tags=train_tags, require_tag_match=require_tag_match,
        all_tagged_nouns_verbs=all_tagged, run_mode=args.mode,
    )

    run_mode_safe = re.sub(r"[^A-Za-z0-9_-]+", "_", args.mode)
    job_id = f"{run_mode_safe}_p{args.pattern_type}_{step_label}_n{num_nouns}_v{num_verbs}"

    pd.DataFrame([row])[SUMMARY_COLS].to_csv(os.path.join(parts_dir, f"{job_id}.csv"), index=False)
    with open(os.path.join(conf_parts_dir, f"{job_id}.txt"), "w", encoding="utf-8") as f:
        f.write(confusion_text)

    if confusion_words is not None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        words_csv_path = os.path.join(args.out_dir, f"confusion_words_{job_id}_{ts}.csv")
        confusion_words.to_csv(words_csv_path)
        print(f"Word-level confusion breakdown written to {words_csv_path}")

    print(f"Single-run result written to {parts_dir}/{job_id}.csv and {conf_parts_dir}/{job_id}.txt")


if __name__ == "__main__":
    main()

