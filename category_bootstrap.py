import re
import random
import argparse
import glob
import sys
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


_WORD_CHAR_RE = re.compile(r"[A-Za-z]")


def _is_word_token(tok):
    """
    True if tok contains at least one letter, i.e. counts as an actual word
    rather than punctuation, a sentence-boundary brace ("{"/"}"), or any
    other non-alphabetic token. Abstracted tokens ("noun"/"verb") always
    count as words.
    """
    return bool(_WORD_CHAR_RE.search(tok))


def _is_word_context(tok):
    """
    True if tok is usable as a "real word" neighbor when deciding whether to
    emit a context pattern - i.e. it is NOT the "PUNCT" placeholder and NOT a
    sentence-boundary brace ("{"/"}"). Everything else (literal words, and
    the "noun"/"verb" abstraction labels) counts as a word here, since those
    always contain letters. Use this instead of _is_word_token on context
    strings, because _is_word_token("PUNCT") would otherwise be True (it
    contains letters) even though PUNCT stands in for a punctuation mark,
    not a word.
    """
    return tok not in ("{", "}", "PUNCT")


def _split_word_lemma_tag(element):
    """
    Split one WORD_LEMMA_TAG corpus token (e.g. "thought_think_VERB") into
    its (surface_word, lemma, tag) fields.

    tag is always the LAST underscore-separated segment - this corpus's
    tagset (ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON,
    PUNCT, SCONJ, VERB, X) never itself contains an underscore, so this part
    is unambiguous no matter what word/lemma contain.

    word and lemma are harder: both can themselves contain underscores, for
    multi-word compounds (proper nouns, fixed phrases) - e.g. the surface
    form "Thomas train" is recorded as "thomas_train", and when its own
    lemma is ALSO that same multi-word compound (common for names/fixed
    phrases the lemmatizer doesn't otherwise normalize), the full raw token
    becomes "thomas_train_thomas_train_NOUN": four underscore-separated
    segments before the tag, not the usual two ("thought_think").

    A plain regex can't tell where word ends and lemma begins in a case
    like that from the string alone. This function resolves it the way an
    empirical scan of this pipeline's corpus file showed these compounds
    actually work:
      - exactly 1 segment before the tag: word = lemma = that segment.
      - exactly 2 segments: word = first, lemma = second (the ordinary
        single-word case, e.g. "thought"/"think").
      - more than 2 segments (a compound): if it splits into two equal-
        length halves that are IDENTICAL word-for-word (word == lemma as a
        whole compound - true for 10,774 of the 11,131 >2-segment tokens in
        this corpus, e.g. "thomas_train_thomas_train"), word = lemma = that
        half.
      - otherwise (an uneven/non-matching compound - a small remainder,
        ~360 tokens/~160 distinct types in this corpus, mostly two-word
        proper names whose auto-generated lemma doesn't cleanly mirror the
        surface form, e.g. "fireman_sam_fireman's_fireman"): there's no
        reliable rule to recover the true split, so word and lemma are BOTH
        set to the entire original compound string, rather than guessing a
        boundary that could be silently wrong.

    This deliberately replaces an earlier greedy-regex approach that always
    guessed some split, which produced garbled results in exactly the
    common (equal-halves) compound case above - it used to make the WORD
    "thomas_train_thomas" and the LEMMA just "train", losing the actual
    compound entirely. This function gets that case exactly right, and
    refuses to guess (falling back to the whole compound for both fields)
    rather than risk the same kind of silent corruption on the harder,
    genuinely ambiguous remainder.

    Returns (surface_word, lemma, tag), all still in their original case -
    the call site lowercases as needed.
    """
    parts = element.split("_")
    tag = parts[-1]
    rest = parts[:-1]
    if len(rest) <= 2:
        if len(rest) == 1:
            return rest[0], rest[0], tag
        return rest[0], rest[1], tag
    if len(rest) % 2 == 0:
        half = len(rest) // 2
        a, b = rest[:half], rest[half:]
        if a == b:
            word = "_".join(a)
            return word, word, tag
    # Ambiguous compound - refuse to guess a split; keep the whole thing as
    # both fields (see docstring).
    whole = "_".join(rest)
    return whole, whole, tag


def extract_context_patterns_fast(corpus, seeds, corpus_words=None, window_size=2, dtype=np.int32, pattern_type=1,
                                   corpus_tags=None, require_tag_match=False,
                                   all_tagged_nouns_verbs=False, abstract_context=True,
                                   track_target_words=False):
    """
    corpus: the LEMMA sequence - used only to match seeds to tokens (is_noun/
        is_verb membership against seeds['nouns']/seeds['verbs'], and the
        require_tag_match/all_tagged_nouns_verbs per-occurrence tag checks).

    corpus_words: the SURFACE WORD FORM sequence, aligned index-for-index
        with corpus. This is what actually gets used to build pattern text
        and to record the lexical filler for a target/context word that
        isn't abstracted to "noun"/"verb" - i.e. everywhere this function
        used to write a literal word into a pattern or a df_contexts row
        label, it now writes the surface form instead of the lemma. Required
        (must be the same length as corpus).

    abstract_context:
        True (default) - context words are abstracted to "noun"/"verb" when
            they qualify per is_noun()/is_verb() (original behavior).
        False - context words are left as their literal surface form; no
            noun/verb abstraction is applied to context tokens. Only the
            TARGET word's row label (NOUN/VERB/literal type) is unaffected
            by this flag - abstraction of the target is controlled
            separately and always applied.

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

    track_target_words:
        False (default) - only the usual (rows=fillers, columns=patterns) matrix
            is returned, where a target word that qualifies as a noun/verb is
            collapsed to the literal row label "NOUN"/"VERB" (original behavior,
            unaffected by this flag).
        True - ALSO build and return a second (rows=(filler, category),
            columns=patterns) sparse matrix, over the same pattern columns,
            that keeps the target's actual surface word in 'filler' always -
            never collapsed to "NOUN"/"VERB" - and records its status in a
            parallel 'category' level ("NOUN"/"VERB"/"OTHER"). This is purely
            an additional reporting output (see df_target_words_to_long) for
            inspecting which literal words a pattern was learned from; it does
            not change the main df_contexts matrix or anything derived from
            it (categorization/scoring still use the collapsed matrix exactly
            as before). When True, the function returns (df, df_target_words)
            instead of just df.
    """
    if (require_tag_match or all_tagged_nouns_verbs) and corpus_tags is None:
        raise ValueError("corpus_tags must be provided when require_tag_match=True or all_tagged_nouns_verbs=True")
    if (require_tag_match or all_tagged_nouns_verbs) and len(corpus_tags) != len(corpus):
        raise ValueError("corpus_tags must be the same length as corpus")
    if corpus_words is None:
        raise ValueError("corpus_words (surface word forms) must be provided")
    if len(corpus_words) != len(corpus):
        raise ValueError("corpus_words must be the same length as corpus")

    # Row/type universe is the surface word forms (plus NOUN/VERB) - a
    # target/context word that isn't abstracted to noun/verb is recorded
    # under its literal surface form, not its lemma.
    types = sorted(set(corpus_words))
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

    target_word_types = []
    target_word_to_idx = {}
    tw_rows = []
    tw_cols = []
    tw_data = []

    for i, word in enumerate(corpus):
        word_form = corpus_words[i]
        if word_form not in ("{", "}") and _is_word_token(word_form):
            begin = max(i - window_size, 0)
            end = min(i + window_size, len(corpus) - 1)
            context_indices = list(range(begin, end + 1))
            del context_indices[i-begin]

            # verb membership takes priority over noun on overlap. Note:
            # is_verb/is_noun match on the LEMMA (w); the literal fallback
            # text uses the surface form (w_form) instead.
            context = []
            for idx in context_indices:
                w = corpus[idx]
                w_form = corpus_words[idx]
                if abstract_context and is_verb(w, idx):
                    context.append("verb")
                elif abstract_context and is_noun(w, idx):
                    context.append("noun")
                elif w_form in ("{", "}"):
                    # sentence-boundary braces stay literal - the pattern
                    # trimming regexes below look for these exact characters
                    context.append(w_form)
                elif not _is_word_token(w_form):
                    # any other non-word token is punctuation - normalize all
                    # punctuation marks to a single shared "PUNCT" label so
                    # e.g. "," and "." aren't treated as different context
                    # words, and so it's obvious PUNCT is not a real word
                    context.append("PUNCT")
                else:
                    context.append(w_form)

            if len(context) == 4:
                # noun membership takes priority over verb on overlap. As
                # above, is_noun/is_verb match on the lemma (word); the
                # literal-filler fallback uses the surface form (word_form).
                if is_noun(word, i):
                    seed_id = types_to_idx["NOUN"]
                    target_category = "NOUN"
                elif is_verb(word, i):
                    seed_id = types_to_idx["VERB"]
                    target_category = "VERB"
                else:
                    seed_id = types_to_idx[word_form]
                    target_category = "OTHER"

                if track_target_words:
                    tw_key = (word_form, target_category)
                    tw_id = target_word_to_idx.get(tw_key)
                    if tw_id is None:
                        tw_id = len(target_word_types)
                        target_word_types.append(tw_key)
                        target_word_to_idx[tw_key] = tw_id

                if pattern_type == 1:
                    p1 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X_" + context[2]))
                    p1a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X"))
                    candidates = []
                    if _is_word_context(context[1]) or _is_word_context(context[2]):
                        candidates.append(p1)
                    if _is_word_context(context[1]):
                        candidates.append(p1a)
                    for p in candidates:
                        idx = context_to_idx.get(p)
                        if idx is None:
                            idx = len(contexts)
                            contexts.append(p)
                            context_to_idx[p] = idx
                        rows.append(seed_id)
                        cols.append(idx)
                        data.append(1)
                        if track_target_words:
                            tw_rows.append(tw_id)
                            tw_cols.append(idx)
                            tw_data.append(1)
                elif pattern_type == 2:
                    p2 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", "X_" + context[2] + "_" + context[3]))
                    p2a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", "X_" + context[2]))
                    candidates = []
                    if _is_word_context(context[2]) or _is_word_context(context[3]):
                        candidates.append(p2)
                    if _is_word_context(context[2]):
                        candidates.append(p2a)
                    for p in candidates:
                        idx = context_to_idx.get(p)
                        if idx is None:
                            idx = len(contexts)
                            contexts.append(p)
                            context_to_idx[p] = idx
                        rows.append(seed_id)
                        cols.append(idx)
                        data.append(1)
                        if track_target_words:
                            tw_rows.append(tw_id)
                            tw_cols.append(idx)
                            tw_data.append(1)
                elif pattern_type == 3:
                    p3 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[0] + "_" + context[1] + "_X"))
                    p3a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X"))

                    candidates = []
                    if _is_word_context(context[0]) or _is_word_context(context[1]):
                        candidates.append(p3)
                    if _is_word_context(context[1]):
                        candidates.append(p3a)
                    for p in candidates:
                        idx = context_to_idx.get(p)
                        if idx is None:
                            idx = len(contexts)
                            contexts.append(p)
                            context_to_idx[p] = idx
                        rows.append(seed_id)
                        cols.append(idx)
                        data.append(1)
                        if track_target_words:
                            tw_rows.append(tw_id)
                            tw_cols.append(idx)
                            tw_data.append(1)
    if not data:
        df = pd.DataFrame(np.zeros((len(types), 0), dtype=int), index=types, columns=[])
        if track_target_words:
            empty_index = pd.MultiIndex.from_tuples([], names=["filler", "category"])
            df_target_words = pd.DataFrame(np.zeros((0, 0), dtype=int), index=empty_index, columns=[])
            return df, df_target_words
        return df

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

    if track_target_words:
        if not tw_data:
            empty_index = pd.MultiIndex.from_tuples([], names=["filler", "category"])
            df_target_words = pd.DataFrame(np.zeros((0, len(contexts)), dtype=int), index=empty_index, columns=contexts)
        else:
            tw_rows_arr = np.asarray(tw_rows, dtype=np.int32)
            tw_cols_arr = np.asarray(tw_cols, dtype=np.int32)
            tw_data_arr = np.asarray(tw_data, dtype=dtype)
            tw_coo = coo_matrix(
                (tw_data_arr, (tw_rows_arr, tw_cols_arr)),
                shape=(len(target_word_types), len(contexts)), dtype=dtype,
            )
            tw_index = pd.MultiIndex.from_tuples(target_word_types, names=["filler", "category"])
            df_target_words = pd.DataFrame.sparse.from_spmatrix(tw_coo, index=tw_index, columns=contexts)
        return df, df_target_words

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


def resolve_noun_verb_seed_overlap(noun_seeds, verb_seeds, word_col="Word",
                                    include_col="Include", prop_col="PROPORTION"):
    """
    A lemma can legitimately be tagged as both a noun and a verb somewhere in
    the corpus (e.g. "walk"), so it can end up with Include==1 in BOTH
    noun_seeds and verb_seeds independently. Left alone, that word would be
    used as a seed for both categories at once - is_noun("walk") and
    is_verb("walk") would both be True in extract_context_patterns_fast/
    categorize_with_contexts_fast - which isn't a meaningful category
    assignment for the pattern-learning step.

    Resolve every such conflict by comparing the word's PROPORTION (its
    share of the FULL noun vocabulary vs. the full verb vocabulary - see
    add_proportion; must already be present on both dataframes, computed
    BEFORE any Include filtering) and keeping Include==1 only in whichever
    category the word accounts for the larger share of - e.g. "walk" at 2%
    of all noun-tagged tokens vs. 1% of all verb-tagged tokens is kept as a
    noun seed only, with Include forced to 0 in verb_seeds. Ties (proportions
    exactly equal) default to keeping the noun assignment - two independently
    -computed proportions landing on the same float is not expected to occur
    with real corpus counts.

    Only words that are Include==1 in BOTH dataframes going in count as a
    conflict - a word that's already excluded from one list isn't actually
    contending to be a seed there, so there's nothing to resolve.

    Call this once, right after add_proportion has been applied to both
    dataframes and before any Include-based filtering/seed selection (e.g.
    compute_seed_steps) - modifies both dataframes' Include column in place
    (via .loc) and also returns them for convenience.
    """
    for label, df in (("noun_seeds", noun_seeds), ("verb_seeds", verb_seeds)):
        for col in (word_col, include_col, prop_col):
            if col not in df.columns:
                raise KeyError(f"{label} is missing required column '{col}'")

    noun_active = noun_seeds[noun_seeds[include_col] == 1]
    verb_active = verb_seeds[verb_seeds[include_col] == 1]

    overlap = set(noun_active[word_col]) & set(verb_active[word_col])
    if not overlap:
        return noun_seeds, verb_seeds

    noun_prop = dict(zip(noun_active[word_col], noun_active[prop_col]))
    verb_prop = dict(zip(verb_active[word_col], verb_active[prop_col]))

    demote_from_noun = [w for w in overlap if noun_prop[w] < verb_prop[w]]
    demote_from_verb = [w for w in overlap if noun_prop[w] >= verb_prop[w]]

    if demote_from_verb:
        verb_seeds.loc[verb_seeds[word_col].isin(demote_from_verb), include_col] = 0
    if demote_from_noun:
        noun_seeds.loc[noun_seeds[word_col].isin(demote_from_noun), include_col] = 0

    # stderr, not stdout: --print-num-seed-steps below relies on stdout
    # containing ONLY the final step count (a dispatcher like run_cluster.sh
    # captures it via command substitution) - diagnostics must not share it.
    print(
        f"Resolved {len(overlap)} noun/verb seed overlap(s): kept as noun seed "
        f"({len(demote_from_verb)}) or kept as verb seed ({len(demote_from_noun)}), "
        f"picked by larger share of that category's total token count.",
        file=sys.stderr,
    )

    return noun_seeds, verb_seeds


def df_contexts_to_long(df_contexts):
    """
    Convert the (rows=fillers, columns=patterns) sparse count matrix
    returned by extract_context_patterns_fast into a tidy long-format
    DataFrame - one row per (pattern, filler, count), count > 0 only - i.e.
    every pattern LEARNED FROM TRAINING and every filler (a specific corpus
    word, or "NOUN"/"VERB" if abstracted) it was ever seen with, and how many
    times. This is the full trained model, not just the subset of patterns
    actually used to classify a test-set word (that's pattern_usage, in
    strict_precision_recall).

    Uses the DataFrame's sparse .to_coo() representation directly rather
    than densifying - df_contexts can have hundreds of thousands of pattern
    columns for a full-size corpus (see extract_context_patterns_fast), so
    iterating cell-by-cell or calling .values would be far too slow/memory-
    heavy. Returns an empty (0-row) DataFrame with the right columns if
    df_contexts has no columns at all (e.g. an empty seed list produced no
    patterns).

    Rows are sorted by pattern, then by count descending within each
    pattern, so the most frequent filler for a pattern appears first -
    convenient for skimming in a spreadsheet.
    """
    columns = ['pattern', 'filler', 'count']
    if df_contexts.shape[1] == 0:
        return pd.DataFrame(columns=columns)

    coo = df_contexts.sparse.to_coo()
    fillers = df_contexts.index.to_numpy()
    patterns = df_contexts.columns.to_numpy()
    long_df = pd.DataFrame({
        'pattern': patterns[coo.col],
        'filler': fillers[coo.row],
        'count': coo.data,
    })
    long_df = long_df.sort_values(['pattern', 'count'], ascending=[True, False]).reset_index(drop=True)
    return long_df[columns]


def df_target_words_to_long(df_target_words):
    """
    Convert the (rows=(filler, category) MultiIndex, columns=patterns)
    sparse count matrix returned by extract_context_patterns_fast when
    track_target_words=True into a tidy long-format DataFrame - one row per
    (pattern, filler, category, count), count > 0 only.

    Unlike df_contexts_to_long's 'filler' column - which collapses a target
    word to the literal string "NOUN"/"VERB" whenever it qualifies as a
    noun/verb - this keeps the actual surface word in 'filler' always, and
    records its NOUN/VERB/OTHER status in a separate 'category' column
    instead. Covers every word that occurred as a pattern's target during
    training, not just the non-noun/non-verb "literal filler" words - i.e.
    every filler in df_contexts_to_long's output, whether it was abstracted
    to NOUN/VERB there or not, appears here under its actual word with its
    category.

    Rows are sorted by pattern, then by count descending within each
    pattern, matching df_contexts_to_long. Returns an empty (0-row)
    DataFrame with the right columns if df_target_words has no columns at
    all.
    """
    columns = ['pattern', 'filler', 'category', 'count']
    if df_target_words.shape[1] == 0:
        return pd.DataFrame(columns=columns)

    coo = df_target_words.sparse.to_coo()
    fillers = df_target_words.index.get_level_values('filler').to_numpy()
    categories = df_target_words.index.get_level_values('category').to_numpy()
    patterns = df_target_words.columns.to_numpy()
    long_df = pd.DataFrame({
        'pattern': patterns[coo.col],
        'filler': fillers[coo.row],
        'category': categories[coo.row],
        'count': coo.data,
    })
    long_df = long_df.sort_values(['pattern', 'count'], ascending=[True, False]).reset_index(drop=True)
    return long_df[columns]


def run_extract_and_evaluate(
    train_tokens,
    test_tokens,
    test_tags,
    selected_noun_seeds,
    selected_verb_seeds,
    token_counts,
    sorted_noun_tokens,
    sorted_verb_tokens,
    train_words=None,
    test_words=None,
    word_primary_tag=None,
    target_prob_cutoff=0.0005,
    window_size=2, pattern_type=1,
    train_tags=None, require_tag_match=False,
    all_tagged_nouns_verbs=False, abstract_context=True,
    track_target_words=False,
):
    """
    Run extraction on train_tokens, categorize test_tokens (with test_tags),
    and compute strict precision/recall.
    Returns: (metrics, df_contexts, df_target_words)
    - metrics: output of strict_precision_recall(results)
    - df_contexts: the (rows=fillers, columns=patterns) sparse count matrix
      from extract_context_patterns_fast - the full trained model, i.e.
      every pattern learned from training and every filler count for it, not
      just the subset actually used to classify a test-set word (see
      pattern_usage inside metrics for that). Callers that want this as a
      spreadsheet-friendly table should pass it to df_contexts_to_long.
    - df_target_words: None unless track_target_words=True, in which case
      it's the (rows=(filler, category), columns=patterns) sparse matrix
      from extract_context_patterns_fast that keeps target words literal
      instead of collapsed to "NOUN"/"VERB" - see
      extract_context_patterns_fast and df_target_words_to_long. Does not
      affect metrics or df_contexts.

    track_target_words: passed straight through to
        extract_context_patterns_fast - see there.

    train_tokens/test_tokens: the LEMMA sequences - used only for seed
    matching (is_noun/is_verb membership checks). train_words/test_words:
    the aligned surface WORD FORM sequences - used for pattern text and for
    the lexical filler recorded for a target/context word. Required.

    word_primary_tag: {surface word form -> its most frequent corpus tag},
    from load_corpus_and_split - passed straight through to
    strict_precision_recall for confusion_words' item-<pos> breakdown.

    require_tag_match: if True, a training-corpus word only counts as a noun/verb
        when it is also tagged as that category in train_tags (see
        extract_context_patterns_fast). Requires train_tags to be provided.

    all_tagged_nouns_verbs: if True, seeds are ignored entirely and every
        tagged noun/verb in the training corpus (per train_tags) is used to
        extract patterns, rather than only seed words. Requires train_tags
        and test_tags. Takes precedence over require_tag_match.

    abstract_context: if False, context words are left as literal surface
        forms instead of being abstracted to "noun"/"verb". Must be applied
        consistently between pattern extraction (train) and categorization
        (test) - see extract_context_patterns_fast and
        categorize_with_contexts_fast.
    """
    if train_words is None or test_words is None:
        raise ValueError("train_words and test_words (surface word forms) must be provided")

    seeds = {'nouns': selected_noun_seeds, 'verbs': selected_verb_seeds}

    extraction_result = extract_context_patterns_fast(
        train_tokens, seeds, corpus_words=train_words, window_size=window_size, pattern_type=pattern_type,
        corpus_tags=train_tags, require_tag_match=require_tag_match,
        all_tagged_nouns_verbs=all_tagged_nouns_verbs,
        abstract_context=abstract_context,
        track_target_words=track_target_words,
    )
    if track_target_words:
        df_contexts, df_target_words = extraction_result
    else:
        df_contexts = extraction_result
        df_target_words = None

    # Baseline guess probabilities: the proportion of TRAINING-corpus word
    # occurrences that are in the noun seed list, the verb seed list, or
    # neither (all_tagged_nouns_verbs mode has no seed list, so it uses each
    # occurrence's own corpus tag instead). require_tag_match mirrors this
    # run's own criterion - require_tag_match_true also requires the tag to
    # agree, require_tag_match_false is seed-list membership alone. See
    # compute_seed_tag_guess_probs. Falls back to None (-> baseline_random_scores
    # defaults to the test set's own frequency) if there were no eligible
    # occurrences at all.
    guess_probs = compute_seed_tag_guess_probs(
        train_tokens, seeds, corpus_words=train_words,
        corpus_tags=train_tags, all_tagged_nouns_verbs=all_tagged_nouns_verbs,
        require_tag_match=require_tag_match,
    )

    # token_counts is keyed by surface word form (see load_corpus_and_split),
    # so targets below - the words rare enough to attempt classifying at all -
    # are surface forms too. token_counts includes punctuation marks (their
    # own literal surface form, e.g. "." or ",", not yet normalized to
    # "PUNCT" - that normalization only happens inside pattern/context
    # building) alongside real words, so exclude anything that isn't a word
    # (_is_word_token) here - predictions should only ever be attempted for
    # actual words, never punctuation (or the sentence-boundary braces,
    # which token_counts doesn't contain anyway).
    corpus_total = sum(token_counts.values())
    token_probs = {k: (v / corpus_total) for k, v in token_counts.items()}
    targets = [k for k, p in token_probs.items() if p < target_prob_cutoff and _is_word_token(k)]

    results = categorize_with_contexts_fast(
        df_contexts,
        test_tokens,
        test_words,
        targets,
        selected_noun_seeds,
        selected_verb_seeds,
        sorted_noun_tokens,
        sorted_verb_tokens,
        window_size=window_size,
        pattern_type=pattern_type,
        tags=test_tags,
        all_tagged_nouns_verbs=all_tagged_nouns_verbs,
        abstract_context=abstract_context,
    )

    metrics = strict_precision_recall(
        results, guess_probs=guess_probs,
        sorted_noun_tokens=sorted_noun_tokens, sorted_verb_tokens=sorted_verb_tokens,
        word_primary_tag=word_primary_tag,
    )
    return metrics, df_contexts, df_target_words


def get_max_count_item(this_pattern,df):
    if this_pattern not in df.columns:
        return None
    counts = df[this_pattern]
    max_count = counts.max()
    if max_count <= 0:
        # No occurrences of any type for this pattern - nothing to report.
        return "OTHER"
    # Find all categories with the max count
    max_labels = [label for label, val in counts.items() if val == max_count]
    if len(max_labels) == 1:
        return max_labels[0]
    else:
        # Tie for the top (nonzero) count - no single winner, so report the
        # identities of the tied types (e.g. "NOUN|VERB", or "cat|dog") instead
        # of collapsing them into the uninformative literal string "OTHER".
        # Callers that need a strict NOUN/VERB/OTHER split (scoring in
        # categorize_with_contexts_fast) already treat anything other than
        # the exact strings "NOUN"/"VERB" as OTHER,
        # so this is transparent to them - it only changes what shows up in
        # human-facing output like pattern_usage's "predicted" column.
        return "|".join(sorted(str(label) for label in max_labels))

def categorize_with_contexts_fast(df, tokens, word_forms, targets,
                                  selected_noun_seeds, selected_verb_seeds,
                                  sorted_noun_tokens, sorted_verb_tokens,
                                  window_size=2, tags=None, pattern_type=1,
                                  all_tagged_nouns_verbs=False, abstract_context=True):
    """
    tokens: the LEMMA sequence for the corpus being categorized - used only
        to match context words against selected_noun_seeds/selected_verb_seeds
        (the same lemma-based seed-matching extract_context_patterns_fast
        uses at train time).

    word_forms: the SURFACE WORD FORM sequence, aligned index-for-index with
        tokens. This drives target-word iteration/eligibility (targets_set
        membership) and is what's recorded as the classified word's identity
        in the returned results, and it's what fills in a context word's
        literal (non-abstracted) pattern text - i.e. everywhere this
        function used to use the lemma for a literal identity/pattern text,
        it now uses the surface form instead. Required.

    A target is only ever an actual word - punctuation is never classified,
    enforced here directly via _is_word_token (not just by relying on the
    caller's `targets` list already excluding it - see run_extract_and_evaluate),
    mirroring the same target-eligibility check extract_context_patterns_fast
    applies when learning patterns.

    all_tagged_nouns_verbs: when True, context words are marked "noun"/"verb"
    based on their own corpus tag (tags[idx] starting "N"/"V") instead of
    seed-list membership - mirrors extract_context_patterns_fast's
    all_tagged_nouns_verbs mode, so patterns built that way at train time
    actually line up with contexts built at test/categorization time.
    Requires tags to be provided and aligned with tokens/word_forms.

    abstract_context: if False, context words are left as literal surface
        forms (no noun/verb abstraction), matching
        extract_context_patterns_fast's abstract_context=False mode. Must
        match the setting used when the patterns in df were built.
    """
    if word_forms is None:
        raise ValueError("word_forms (surface word forms) must be provided")
    if len(word_forms) != len(tokens):
        raise ValueError("word_forms must be the same length as tokens")
    if all_tagged_nouns_verbs and tags is None:
        raise ValueError("tags must be provided when all_tagged_nouns_verbs=True")
    if all_tagged_nouns_verbs and len(tags) != len(tokens):
        raise ValueError("tags must be the same length as tokens")

    token_count = len(word_forms)
    #print("CALLED!")
    #print(tags)
    selected_noun_set = set(selected_noun_seeds)
    selected_verb_set = set(selected_verb_seeds)
    targets_set = set(targets)
    df_loc = df

    results = []
    trim = _trim_braces_fast
    for i, word in enumerate(word_forms):
        # A target must be an actual word: not a sentence-boundary brace,
        # and not punctuation (_is_word_token requires at least one letter -
        # punctuation marks like "." or "," have none). Checked here
        # directly, not just via targets_set membership, so this holds even
        # if a caller's targets list wasn't already filtered this way.
        if word in ("{", "}") or not _is_word_token(word) or word.lower() not in targets_set:
            continue

        begin = max(i - window_size, 0)
        end = min(i + window_size, token_count - 1)
        context_indices = list(range(begin, end + 1))
        del context_indices[i-begin]

        context = []
        for idx in context_indices:
            w_lemma = tokens[idx]
            w_form = word_forms[idx]
            # sentence-boundary braces stay literal (needed by the trimming
            # regexes below); any other non-word token is punctuation and is
            # normalized to "PUNCT" - must match extract_context_patterns_fast's
            # normalization exactly, or patterns built at train time with
            # "PUNCT" won't be found here at test time.
            fallback = w_form if (w_form in ("{", "}") or _is_word_token(w_form)) else "PUNCT"
            if not abstract_context:
                context.append(fallback)
            elif all_tagged_nouns_verbs:
                # verb-first, matching extract_context_patterns_fast's priority
                if re.match(r"^V", tags[idx]):
                    context.append("verb")
                elif re.match(r"^N", tags[idx]):
                    context.append("noun")
                else:
                    context.append(fallback)
            else:
                if w_lemma in selected_noun_set:
                    context.append("noun")
                elif w_lemma in selected_verb_set:
                    context.append("verb")
                else:
                    context.append(fallback)

        if len(context) != 4:
            continue
        p1 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X_" + context[2]))
        p1a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X"))
        p2 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", "X_" + context[2] + "_" + context[3]))
        p2a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", "X_" + context[2]))
        p3 = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[0] + "_" + context[1] + "_X"))
        p3a = re.sub(r"(.+\}).+", r"\1", re.sub(r".+(\{.+)", r"\1", context[1] + "_X"))

        if pattern_type == 1:
            primary, fallback = p1, p1a
        elif pattern_type == 2:
            primary, fallback = p2, p2a
        else:
            primary, fallback = p3, p3a

        # raw_pred is the un-collapsed winning label from get_max_count_item -
        # could be "NOUN"/"VERB", a specific corpus word (get_max_count_item
        # can return any row label, not just NOUN/VERB), a "|"-joined list of
        # tied labels (e.g. "NOUN|VERB", "cat|dog") when there's no single
        # training-time winner for this pattern, or the literal string
        # "OTHER" itself when the pattern column had no occurrences at all.
        # Whatever raw_pred is, `cat` below still collapses anything that
        # isn't exactly "NOUN"/"VERB" to "OTHER" for scoring purposes - only
        # human-facing output (e.g. pattern_usage's "predicted" column) shows
        # the tied identities. used_pattern records which of the two patterns
        # (primary or its short fallback) actually matched a column in the
        # trained df - None if neither did, i.e. no trained pattern was
        # actually used for this word (it fell through to OTHER for lack of
        # any match at all).
        raw_pred = get_max_count_item(primary, df)
        used_pattern = primary
        if raw_pred is None:
            raw_pred = get_max_count_item(fallback, df)
            used_pattern = fallback
        if raw_pred is None:
            used_pattern = None

        cat = raw_pred if raw_pred in ("NOUN", "VERB") else "OTHER"

        # produce triple if tags provided, else pair - both now carry the
        # pattern-usage bookkeeping too.
        if tags is not None:
            results.append((word, cat, tags[i], used_pattern, raw_pred))
        else:
            results.append((word, cat, used_pattern, raw_pred))

    return results


def compute_seed_tag_guess_probs(corpus, seeds, corpus_words=None, corpus_tags=None,
                                  all_tagged_nouns_verbs=False, require_tag_match=False):
    """
    Guess-probability source for the baseline: the proportion of TRAINING-
    corpus word occurrences that are in the noun seed list, the verb seed
    list, or neither - NOT a self-classification pass over the run's own
    learned patterns.

    corpus/corpus_words: as in extract_context_patterns_fast - corpus is the
    LEMMA sequence (used for noun/verb seed-list membership, same as
    everywhere else in the pipeline), corpus_words is the aligned surface
    WORD FORM sequence (used only to exclude punctuation via
    _is_word_token). Both required and must be the same length.

    corpus_tags: the corpus's own per-occurrence POS tags (spaCy-derived -
    see ChildesDataPrep_Eng.ipynb/ChildesDataPrep_JP.ipynb), required.

    require_tag_match mirrors the run's own noun/verb criterion (same flag
    passed to extract_context_patterns_fast/categorize_with_contexts_fast
    for this run), so the baseline reflects exactly as much information as
    that run mode itself uses - ignored when all_tagged_nouns_verbs=True.

    For every real-word occurrence in `corpus` (punctuation and the "{"/"}"
    sentence-boundary markers excluded via _is_word_token):
      - all_tagged_nouns_verbs=True: there is no curated seed list in this
        mode (see compute_all_tagged_counts/sweep_and_save_runs - it treats
        every corpus-tagged noun/verb as a candidate), so the occurrence's
        own tag is the only signal available: NOUN if the tag starts "N",
        VERB if it starts "V", matching how this mode already decides
        noun/verb status everywhere else in the pipeline.
      - Otherwise, require_tag_match=True (require_tag_match_true mode):
        NOUN if the word's lemma is in the noun seed list AND this
        occurrence's own tag starts "N", VERB if it's in the verb seed list
        AND the tag starts "V". Both conditions are required - a word on
        the noun seed list whose tag doesn't say "noun" in this particular
        occurrence is OTHER, not NOUN.
      - Otherwise, require_tag_match=False (require_tag_match_false mode):
        NOUN if the word's lemma is in the noun seed list, VERB if it's in
        the verb seed list - seed-list membership alone decides this,
        regardless of what this occurrence's own tag says.
      - In both non-all_tagged cases, noun takes priority over verb on a
        tie (a lemma in both seed lists, with both conditions satisfied),
        matching the tie-break convention used for target words elsewhere
        in this pipeline (see extract_context_patterns_fast). Anything not
        classified NOUN or VERB is OTHER.

    Returns a dict {'NOUN': p_noun, 'VERB': p_verb, 'OTHER': p_other}
    (always sums to 1), or None if there were no eligible occurrences at
    all (e.g. an empty corpus) - callers should fall back to some other
    guess distribution in that case.
    """
    if corpus_words is None:
        raise ValueError("corpus_words (surface word forms) must be provided")
    if len(corpus_words) != len(corpus):
        raise ValueError("corpus_words must be the same length as corpus")
    if corpus_tags is None:
        raise ValueError("corpus_tags must be provided")
    if len(corpus_tags) != len(corpus):
        raise ValueError("corpus_tags must be the same length as corpus")

    seeds = seeds or {}
    noun_set = set(seeds.get('nouns', []))
    verb_set = set(seeds.get('verbs', []))

    counts = {'NOUN': 0, 'VERB': 0, 'OTHER': 0}
    total = 0

    for i, word in enumerate(corpus):
        if word in ("{", "}"):
            continue
        if not _is_word_token(corpus_words[i]):
            continue

        if all_tagged_nouns_verbs:
            is_noun = bool(re.match(r"^N", corpus_tags[i]))
            is_verb = bool(re.match(r"^V", corpus_tags[i]))
        elif require_tag_match:
            is_noun = (word in noun_set) and bool(re.match(r"^N", corpus_tags[i]))
            is_verb = (word in verb_set) and bool(re.match(r"^V", corpus_tags[i]))
        else:
            is_noun = word in noun_set
            is_verb = word in verb_set

        if is_noun:
            cat = 'NOUN'
        elif is_verb:
            cat = 'VERB'
        else:
            cat = 'OTHER'

        counts[cat] += 1
        total += 1

    if total == 0:
        return None

    return {c: counts[c] / total for c in ('NOUN', 'VERB', 'OTHER')}



# Metric columns that get averaged (and, for CV mean rows, given a matching
# "<col>_std" column) across folds - see evaluate_kfold_and_aggregate. Every
# column here is a plain float computed per fold; runtime_s is included so
# the mean row also reports average (and std of) per-fold wall time.
METRIC_COLS = [
    "runtime_s",
    "NOUN_precision", "NOUN_recall",
    "VERB_precision", "VERB_recall",
    "macro_precision", "macro_recall",
    "micro_precision", "micro_recall",
    "baseline_NOUN_precision", "baseline_NOUN_recall",
    "baseline_VERB_precision", "baseline_VERB_recall",
    "baseline_macro_precision", "baseline_macro_recall",
    "baseline_micro_precision", "baseline_micro_recall",
]
STD_COLS = [f"{c}_std" for c in METRIC_COLS]

# "fold": None for a run made with n_folds=1 (pre-cross-validation single
# split) - matches folds' own 'fold'=None convention (see
# load_corpus_and_split). Otherwise 0..n_folds-1 for an individual fold's
# own row, or the string "mean" for the row averaging across that job's
# folds (see evaluate_kfold_and_aggregate) - the STD_COLS are only populated
# (non-blank) on "mean" rows; individual fold rows and n_folds=1 rows leave
# them blank (NaN), since a single value has no std to report.
# "n_folds": how many folds this job's cross-validation used (1 for the
# pre-cross-validation single-split behavior).
SUMMARY_COLS = [
    "time", "mode", "pattern_type", "num_noun_seeds", "num_verb_seeds",
    "fold", "n_folds",
] + METRIC_COLS + STD_COLS


def compute_all_tagged_counts(train_words, train_tags):
    """
    Returns (num_nouns, num_verbs): the count of distinct noun-/verb-tagged
    surface WORD FORMS in the training corpus (per train_tags) - what
    all_tagged_nouns_verbs=True actually uses instead of a seed list. Used
    both by sweep_and_save_runs and the standalone single-run CLI mode, so
    that mode logs a meaningful "how many nouns/verbs" number. Takes
    train_words (surface form), not the lemma array, so this lines up with
    the rest of the pipeline's word-form-based reporting.
    """
    if train_tags is None:
        raise ValueError("train_tags must be provided for all_tagged_nouns_verbs mode")
    actual_nouns = {w for w, t in zip(train_words, train_tags) if re.match(r"^N", t)}
    actual_verbs = {w for w, t in zip(train_words, train_tags) if re.match(r"^V", t)}
    return len(actual_nouns), len(actual_verbs)


def compute_seed_steps(noun_seeds_df, verb_seeds_df,
                        max_cum_prop_threshold=0.239, max_sweep_steps=None):
    """
    Filters both seed dataframes down to Include==1 (only words eligible to
    be seeds), recomputes cumulative proportion on that filtered,
    frequency-ordered subset (see add_cumulative_proportion), then returns a
    MATCHED SEQUENCE of (num_nouns, num_verbs) pairs - NOT a cross product/
    grid - one entry per noun count from 1 up to whatever count first
    reaches max_cum_prop_threshold's share of noun tokens (or the full
    Include==1 noun list, if smaller).

    The matching walks the noun and verb cumulative-proportion staircases
    together. Each noun count n is paired with every verb count v (0..the
    full verb list) whose OWN cumulative-proportion "breakpoint" falls
    strictly after noun count (n-1)'s cumulative proportion and up to and
    including noun count n's - i.e. whichever verb breakpoints appear while
    the noun staircase is sitting on step n get attached to step n. v=0 (no
    verbs at all) counts as a breakpoint at proportion 0, so noun counts
    whose own proportion is still below the smallest real verb count's
    proportion get paired with num_verbs=0 rather than being skipped or
    forced onto verb count 1.

    If a noun step's window happens to contain NO verb breakpoint at all
    (possible where verbs are coarser than nouns in some stretch), that noun
    count is still paired with the single best-matching verb count (the
    largest verb count whose own cumulative proportion doesn't exceed this
    noun count's), so every noun count in range gets at least one pairing.

    This guarantees every noun count 1..N and every verb count 0..M appears
    at least once across the returned sequence (N/M being however many of
    each max_cum_prop_threshold allows) - unlike a "closest single verb
    count per noun count" scheme, which can skip some verb counts entirely
    when verb granularity is locally finer than noun granularity. That's
    exactly why some noun counts end up paired with MORE THAN ONE verb count
    (two or more verb breakpoints landing in the same noun step's window)
    rather than every noun mapping to exactly one verb - e.g. with the
    current seed files/default threshold, noun counts 28-36 each pair with
    two verb counts while 1-27 each pair with one, for 45 total pairings
    (36 noun counts, 9 of which contribute 2 pairings each).

    max_sweep_steps (None by default): an optional EXTRA cap on how many
    noun counts are considered - only noun counts 1..min(N, max_sweep_steps)
    are used, instead of every noun count max_cum_prop_threshold allows.

    max_cum_prop_threshold exists because the noun and verb seed lists are
    typically very different sizes (thousands of nouns vs. a few dozen
    human-curated verbs) - it caps the noun side directly (fewest
    highest-frequency nouns whose cumulative token share reaches this
    proportion), and indirectly caps the verb side too, since verbs are only
    matched up to whatever proportion the noun side reaches. Note a small,
    curated seed list (e.g. this pipeline's 33 human-approved verbs) can max
    out well before reaching max_cum_prop_threshold - e.g. all 33 verbs
    together cover only 23.9% of verb tokens, so raising
    max_cum_prop_threshold past that point only grows the noun side further,
    with every extra noun count then pairing with verb count 33 (the full
    verb list) since there are no higher verb breakpoints left to match.

    Returns (steps, noun_seeds_df, verb_seeds_df):
      - steps: a list of (num_nouns, num_verbs) tuples, ordered by
        increasing num_nouns (and, within a noun count that matches more
        than one verb count, by increasing num_verbs).
      - noun_seeds_df, verb_seeds_df: the filtered/annotated dataframes, so
        callers can slice them directly, e.g. noun_seeds_df.iloc[:num_nouns]['Word'].
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
    n_cum = noun_seeds_df[m_col].to_numpy()
    v_cum = verb_seeds_df[n_col].to_numpy()
    total_noun = len(n_cum)
    total_verb = len(v_cum)

    n_max = max(1, int((n_cum < max_cum_prop_threshold).sum()) + 1)
    n_max = min(n_max, total_noun)
    if max_sweep_steps is not None:
        n_max = min(n_max, max_sweep_steps)

    def _best_verb_match(p):
        """Largest verb count whose own cumulative proportion doesn't
        exceed p, else 0 (no real verb count reaches this low yet)."""
        best = 0
        for v in range(1, total_verb + 1):
            if v_cum[v - 1] <= p:
                best = v
            else:
                break  # v_cum is non-decreasing - nothing later will match either
        return best

    steps = []
    prev_noun_p = 0.0
    for n in range(1, n_max + 1):
        p = n_cum[n - 1]
        matched = [
            v for v in range(0, total_verb + 1)
            if prev_noun_p < (v_cum[v - 1] if v > 0 else 0.0) <= p
        ]
        if not matched:
            matched = [_best_verb_match(p)]
        steps.extend((n, v) for v in matched)
        prev_noun_p = p

    return steps, noun_seeds_df, verb_seeds_df


def evaluate_single_run(
    run_fn, train_tokens, test_tokens, test_tags,
    selected_nouns, selected_verbs, num_nouns, num_verbs,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    train_words=None, test_words=None, word_primary_tag=None,
    target_prob_cutoff=0.0005, window_size=2, pattern_type=1,
    train_tags=None, require_tag_match=False, all_tagged_nouns_verbs=False,
    abstract_context=True,
    track_target_words=False,
    run_mode="run",
):
    """
    Runs a single (mode, pattern_type, seed-set) configuration and returns
    everything needed to log it - (row, confusion_text, confusion_words,
    pattern_usage, learned_patterns, learned_patterns_words) - WITHOUT
    writing to any file. This is the atomic unit of work shared by:
      - sweep_and_save_runs, which appends the result to a shared
        summary.csv/confusion_matrices.txt (safe since it runs sequentially
        in a single process), and
      - the standalone single-job CLI mode (--mode ...), which writes the
        result to its own uniquely-named per-run files instead, so many
        instances of this script can run in parallel - e.g. one per core on
        a cluster (see run_cluster.sh) - without corrupting a shared file
        through concurrent writes.

    learned_patterns: the full trained model for this run as a tidy
    (pattern, filler, count) DataFrame - see df_contexts_to_long - not just
    the patterns actually used to classify a test-set word (that's
    pattern_usage).

    learned_patterns_words: None unless track_target_words=True, in which
    case it's the same trained model as learned_patterns but with the
    target's literal surface word always in 'filler' (never collapsed to
    "NOUN"/"VERB") and its status in a separate 'category' column - see
    df_target_words_to_long.
    """
    print(f"\nRunning [{run_mode}] pattern_type={pattern_type} with num_noun_seeds={num_nouns}, num_verb_seeds={num_verbs}...")
    t0 = time.time()
    metrics, df_contexts, df_target_words = run_fn(
        train_tokens, test_tokens, test_tags,
        selected_nouns, selected_verbs,
        token_counts, sorted_noun_tokens, sorted_verb_tokens,
        train_words=train_words, test_words=test_words, word_primary_tag=word_primary_tag,
        target_prob_cutoff=target_prob_cutoff, window_size=window_size,
        pattern_type=pattern_type,
        train_tags=train_tags, require_tag_match=require_tag_match,
        all_tagged_nouns_verbs=all_tagged_nouns_verbs,
        abstract_context=abstract_context,
        track_target_words=track_target_words,
    )
    t1 = time.time()
    learned_patterns = df_contexts_to_long(df_contexts)
    learned_patterns_words = df_target_words_to_long(df_target_words) if df_target_words is not None else None

    per_class = metrics['per_class']
    macro = metrics['macro']
    micro = metrics['micro']
    confusion = metrics['confusion']
    confusion_words = metrics.get('confusion_words')
    baseline = metrics.get('baseline', {})
    pattern_usage = metrics.get('pattern_usage')

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

    return row, confusion_text, confusion_words, pattern_usage, learned_patterns, learned_patterns_words


def evaluate_kfold_and_aggregate(
    run_fn, folds,
    selected_nouns, selected_verbs, num_nouns, num_verbs,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    word_primary_tag=None,
    target_prob_cutoff=0.0005, window_size=2, pattern_type=1,
    require_tag_match=False, all_tagged_nouns_verbs=False,
    abstract_context=True,
    track_target_words=False,
    run_mode="run",
):
    """
    Runs evaluate_single_run once per fold in `folds` (see
    load_corpus_and_split/make_kfold_sentence_splits) and aggregates the
    results. Returns (rows, confusion_texts, confusion_words_list,
    pattern_usage_list, learned_patterns_list, learned_patterns_words_list):

    - rows: one row dict per fold (each with 'fold' set to that fold's own
      label - None if folds is the single n_folds=1 split, else 0..n_folds-1
      - see load_corpus_and_split), PLUS, only when there's more than one
      fold, one additional row with 'fold'="mean" giving the across-fold
      mean of every column in METRIC_COLS and its standard deviation in the
      matching "<col>_std" column (left as None on the individual per-fold
      rows - a single value has no std to report). Every row also gets
      'n_folds' set to len(folds).

    - confusion_texts: list of (fold_label, confusion_text) tuples, one per
      fold - no aggregate text.

    - confusion_words_list/pattern_usage_list/learned_patterns_list/
      learned_patterns_words_list: list of (fold_label, num_nouns, num_verbs,
      value) tuples, one per fold - num_nouns/num_verbs are that SPECIFIC
      fold's own seed-count display values (see all_tagged_nouns_verbs
      below - matters because these can differ fold-to-fold), included here
      so callers building per-fold filenames (see sweep_and_save_runs) don't
      need to separately re-derive or look them up. No aggregate/merged
      version of any of these dataframes, since each fold trains on
      different data, so there's no single "the model" to combine them into
      (this mirrors why the mean row above only covers scalar metrics, not
      these per-fold artifacts).

    all_tagged_nouns_verbs: unlike selected_nouns/selected_verbs/num_nouns/
        num_verbs (which, same as evaluate_single_run, are used as-is for
        every fold), when this is True, each fold's OWN row instead reports
        num_noun_seeds/num_verb_seeds recomputed from that fold's own
        training data via compute_all_tagged_counts (mirroring what the
        pre-cross-validation single-job path used to compute for this mode),
        so it varies fold-to-fold with exactly how many distinct noun/verb
        word types that fold's training split contains. The passed-in
        num_nouns/num_verbs is then used only as a display fallback (e.g.
        the mean row's num_noun_seeds/num_verb_seeds, which - being a
        fixed-across-folds identity/config value everywhere else in this
        pipeline, not a per-fold metric - is taken from fold 0 rather than
        averaged).
    """
    rows = []
    confusion_texts = []
    confusion_words_list = []
    pattern_usage_list = []
    learned_patterns_list = []
    learned_patterns_words_list = []

    n_folds = len(folds)
    for fold_info in folds:
        fold_label = fold_info["fold"]
        if fold_label is not None:
            print(f"\n=== {run_mode} p{pattern_type} fold {fold_label + 1}/{n_folds} ===")

        fold_num_nouns, fold_num_verbs = num_nouns, num_verbs
        if all_tagged_nouns_verbs:
            fold_num_nouns, fold_num_verbs = compute_all_tagged_counts(
                fold_info["train_words"], fold_info["train_tags"],
            )

        row, confusion_text, confusion_words, pattern_usage, learned_patterns, learned_patterns_words = evaluate_single_run(
            run_fn, fold_info["train"], fold_info["test"], fold_info["test_tags"],
            selected_nouns, selected_verbs, fold_num_nouns, fold_num_verbs,
            token_counts, sorted_noun_tokens, sorted_verb_tokens,
            train_words=fold_info["train_words"], test_words=fold_info["test_words"],
            word_primary_tag=word_primary_tag,
            target_prob_cutoff=target_prob_cutoff, window_size=window_size, pattern_type=pattern_type,
            train_tags=fold_info["train_tags"], require_tag_match=require_tag_match,
            all_tagged_nouns_verbs=all_tagged_nouns_verbs, abstract_context=abstract_context,
            track_target_words=track_target_words,
            run_mode=run_mode,
        )
        row["fold"] = fold_label
        row["n_folds"] = n_folds
        for c in STD_COLS:
            row[c] = None
        rows.append(row)
        confusion_texts.append((fold_label, confusion_text))
        confusion_words_list.append((fold_label, fold_num_nouns, fold_num_verbs, confusion_words))
        pattern_usage_list.append((fold_label, fold_num_nouns, fold_num_verbs, pattern_usage))
        learned_patterns_list.append((fold_label, fold_num_nouns, fold_num_verbs, learned_patterns))
        learned_patterns_words_list.append((fold_label, fold_num_nouns, fold_num_verbs, learned_patterns_words))

    if n_folds > 1:
        mean_row = dict(rows[0])
        mean_row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        mean_row["fold"] = "mean"
        mean_row["n_folds"] = n_folds
        mean_row["num_noun_seeds"] = rows[0]["num_noun_seeds"]
        mean_row["num_verb_seeds"] = rows[0]["num_verb_seeds"]
        for c in METRIC_COLS:
            values = [r[c] for r in rows]
            mean_row[c] = float(np.mean(values))
            mean_row[f"{c}_std"] = float(np.std(values))
        rows.append(mean_row)

    return rows, confusion_texts, confusion_words_list, pattern_usage_list, learned_patterns_list, learned_patterns_words_list


def sweep_and_save_runs(
    run_fn, folds,
    noun_seeds_df, verb_seeds_df,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    word_primary_tag=None,
    out_dir="sweep_runs",
    max_cum_prop_threshold=0.239,
    target_prob_cutoff=0.0005,
    window_size=2, pattern_type=1,
    require_tag_match=False,
    all_tagged_nouns_verbs=False,
    abstract_context=True,
    track_target_words=False,
    force_full_seeds=False,
    max_sweep_steps=None,
    run_mode=None,
):
    """
    folds: as returned by load_corpus_and_split - a list of per-split dicts
        (a single element with 'fold'=None for the pre-cross-validation
        single 80/20-style split; n_folds elements with 'fold'=0..n_folds-1
        under k-fold cross-validation). Each (mode, pattern_type, seed-step)
        configuration this function tries is run once per fold and
        aggregated by evaluate_kfold_and_aggregate - see there for exactly
        what gets logged/written per fold vs. only once (as a "mean" row).

    track_target_words: passed straight through to evaluate_single_run/
        extract_context_patterns_fast - when True, an additional
        learned_patterns_words_*.xlsx (literal target word + NOUN/VERB/OTHER
        category columns, instead of the usual collapsed filler) is written
        alongside each learned_patterns_*.xlsx. See
        df_target_words_to_long.

    force_full_seeds: run a single pass using the entire (Include==1) seed
        list, rather than sweeping over increasing seed-list sizes. Ignored
        (implied True) when all_tagged_nouns_verbs=True.

    max_sweep_steps: optional EXTRA cap on how many noun counts are
        considered (verb counts are derived from the matched noun counts,
        not swept independently - see compute_seed_steps) - e.g. 20 means
        only noun counts 1..20 are used even if max_cum_prop_threshold would
        otherwise allow more. None (default) means no extra cap. Ignored
        when all_tagged_nouns_verbs=True or force_full_seeds=True (both are
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
    # compute_seed_steps for the filtering/cumulative-proportion/matching
    # logic - it's shared with the standalone single-job CLI mode so the
    # exact same seed-set sequence is available to both.
    steps, noun_seeds_df, verb_seeds_df = compute_seed_steps(
        noun_seeds_df, verb_seeds_df,
        max_cum_prop_threshold=max_cum_prop_threshold,
        max_sweep_steps=max_sweep_steps,
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
                f"schema now includes 'fold', 'n_folds' and '<metric>_std' "
                f"columns for k-fold cross-validation). Move, rename, or "
                f"delete the old file, or point out_dir somewhere new, "
                f"before re-running."
            )

    def _fold_suffix(fold_label):
        return "" if fold_label is None else f"_fold{fold_label}"

    def _log(rows, confusion_texts, confusion_words_list, pattern_usage_list,
              learned_patterns_list, learned_patterns_words_list):
        pd.DataFrame(rows)[SUMMARY_COLS].to_csv(summary_path, mode="a", header=False, index=False)

        with open(confusion_path, "a", encoding="utf-8") as f:
            for _fold_label, confusion_text in confusion_texts:
                f.write(confusion_text)

        for fold_label, num_nouns, num_verbs, confusion_words in confusion_words_list:
            if confusion_words is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                words_csv_path = os.path.join(
                    out_dir,
                    f"confusion_words_{run_mode_safe}_p{pattern_type}_n{num_nouns}_v{num_verbs}"
                    f"{_fold_suffix(fold_label)}_{ts}.csv",
                )
                confusion_words.to_csv(words_csv_path)
                print(f"Word-level confusion breakdown written to {words_csv_path}")

        for fold_label, num_nouns, num_verbs, pattern_usage in pattern_usage_list:
            if pattern_usage is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                pattern_usage_csv_path = os.path.join(
                    out_dir,
                    f"pattern_usage_{run_mode_safe}_p{pattern_type}_n{num_nouns}_v{num_verbs}"
                    f"{_fold_suffix(fold_label)}_{ts}.csv",
                )
                pattern_usage.to_csv(pattern_usage_csv_path)
                print(f"Pattern usage breakdown written to {pattern_usage_csv_path}")

        for fold_label, num_nouns, num_verbs, learned_patterns in learned_patterns_list:
            if learned_patterns is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                learned_patterns_xlsx_path = os.path.join(
                    out_dir,
                    f"learned_patterns_{run_mode_safe}_p{pattern_type}_n{num_nouns}_v{num_verbs}"
                    f"{_fold_suffix(fold_label)}_{ts}.xlsx",
                )
                learned_patterns.to_excel(learned_patterns_xlsx_path, index=False)
                print(f"Learned patterns/fillers written to {learned_patterns_xlsx_path}")

        for fold_label, num_nouns, num_verbs, learned_patterns_words in learned_patterns_words_list:
            if learned_patterns_words is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                learned_patterns_words_xlsx_path = os.path.join(
                    out_dir,
                    f"learned_patterns_words_{run_mode_safe}_p{pattern_type}_n{num_nouns}_v{num_verbs}"
                    f"{_fold_suffix(fold_label)}_{ts}.xlsx",
                )
                learned_patterns_words.to_excel(learned_patterns_words_xlsx_path, index=False)
                print(f"Learned patterns/target-words (literal + category) written to {learned_patterns_words_xlsx_path}")

    def _run_and_log(selected_nouns, selected_verbs, num_nouns, num_verbs):
        (rows, confusion_texts, confusion_words_list, pattern_usage_list,
         learned_patterns_list, learned_patterns_words_list) = evaluate_kfold_and_aggregate(
            run_fn, folds,
            selected_nouns, selected_verbs, num_nouns, num_verbs,
            token_counts, sorted_noun_tokens, sorted_verb_tokens,
            word_primary_tag=word_primary_tag,
            target_prob_cutoff=target_prob_cutoff, window_size=window_size, pattern_type=pattern_type,
            require_tag_match=require_tag_match,
            all_tagged_nouns_verbs=all_tagged_nouns_verbs, abstract_context=abstract_context,
            track_target_words=track_target_words,
            run_mode=run_mode,
        )
        _log(rows, confusion_texts, confusion_words_list, pattern_usage_list,
             learned_patterns_list, learned_patterns_words_list)

    if all_tagged_nouns_verbs:
        # Single full pass, no sweep over increasing seed-list sizes.
        # Noun/verb status in this mode is decided purely from each
        # occurrence's own corpus tag (see extract_context_patterns_fast/
        # categorize_with_contexts_fast's all_tagged_nouns_verbs branch,
        # which bypasses seed-set membership entirely) - so the word list
        # fed in here should be every word actually tagged noun/verb in the
        # postprocessed training corpus, not the curated (Include==1) seed
        # list. sorted_noun_tokens/sorted_verb_tokens are already exactly
        # that (computed by load_corpus_and_split straight from the
        # corpus's own tags, and now surface-form-based).
        selected_nouns = list(sorted_noun_tokens)
        selected_verbs = list(sorted_verb_tokens)
        # Seed-list sizes (total_noun/total_verb) would be a misleading
        # thing to log here too - each fold's own row (and filenames) report
        # the actual count of distinct noun-/verb-tagged word types found in
        # THAT fold's training data instead, computed inside
        # evaluate_kfold_and_aggregate. The 0, 0 here is just an unused
        # placeholder, immediately overridden per fold - see there.
        _run_and_log(selected_nouns, selected_verbs, 0, 0)
        return summary_path, confusion_path

    if force_full_seeds:
        # Single full pass, no sweep over increasing seed-list sizes. Seeds
        # ARE what's driving noun/verb status here, so use and log the
        # actual full (Include==1) seed list/sizes.
        selected_nouns = noun_seeds_df['Word'].tolist()
        selected_verbs = verb_seeds_df['Word'].tolist()
        num_nouns_display, num_verbs_display = total_noun, total_verb
        _run_and_log(selected_nouns, selected_verbs, num_nouns_display, num_verbs_display)
        return summary_path, confusion_path

    for num_nouns, num_verbs in steps:
        selected_nouns = noun_seeds_df.iloc[:num_nouns]['Word'].tolist()
        selected_verbs = verb_seeds_df.iloc[:num_verbs]['Word'].tolist()
        _run_and_log(selected_nouns, selected_verbs, num_nouns, num_verbs)

    return summary_path, confusion_path


def run_mode_comparison(
    run_fn, folds,
    noun_seeds_df, verb_seeds_df,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    word_primary_tag=None,
    out_dir="sweep_runs",
    max_cum_prop_threshold=0.239,
    target_prob_cutoff=0.0005,
    window_size=2,
    pattern_types=(1, 2, 3),
    num_sweep_steps=None,
    abstract_context=True,
    track_target_words=False,
):
    """
    folds: as returned by load_corpus_and_split - passed straight through to
        sweep_and_save_runs (run once per fold, aggregated - see
        evaluate_kfold_and_aggregate).

    For EACH pattern_type in pattern_types (all three by default), runs
    three passes in sequence, sharing the same summary.csv/
    confusion_matrices.txt (distinguished by "mode" and "pattern_type"
    columns/labels) and out_dir for the confusion-words CSVs:

      1. all_tagged_nouns_verbs=True - one full run using every tagged
         noun/verb in the training corpus (mode="all_tagged_nouns_verbs").
      2. require_tag_match=True, swept across the MATCHED SEQUENCE of
         (num_nouns, num_verbs) pairs (mode="require_tag_match_true") - see
         compute_seed_steps for exactly how that sequence is built from
         max_cum_prop_threshold and num_sweep_steps (every noun count is
         paired with whichever verb count(s) cover the matching share of
         verb tokens - NOT a cross product of every noun size with every
         verb size).
      3. require_tag_match=False, swept across the SAME sequence of
         seed-set sizes (mode="require_tag_match_false"). "Same" holds
         because the sequence is derived purely from noun_seeds_df/
         verb_seeds_df + max_cum_prop_threshold/num_sweep_steps, which are
         identical across passes 2 and 3 (and across pattern types, since
         pattern_type doesn't affect seed selection).

    So with the default pattern_types=(1, 2, 3), this runs 3 * (1 + 2 * G)
    passes total, where G is the number of (num_nouns, num_verbs) pairings
    from compute_seed_steps (e.g. G=45 with the current seed files/default
    threshold). pattern_type is recorded in its own summary.csv column, in
    the confusion_matrices.txt entry header, and in the confusion-words CSV
    filename, so every row/entry is traceable to exactly which (pattern_type,
    mode, seed-set) combination produced it.

    Returns the (summary_path, confusion_path) from the very last pass.
    """
    common = dict(
        run_fn=run_fn, folds=folds,
        noun_seeds_df=noun_seeds_df, verb_seeds_df=verb_seeds_df,
        token_counts=token_counts, sorted_noun_tokens=sorted_noun_tokens, sorted_verb_tokens=sorted_verb_tokens,
        word_primary_tag=word_primary_tag,
        out_dir=out_dir,
        max_cum_prop_threshold=max_cum_prop_threshold,
        target_prob_cutoff=target_prob_cutoff, window_size=window_size,
        abstract_context=abstract_context,
        track_target_words=track_target_words,
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



# Friendly names for the Universal-Dependencies-style POS tags this pipeline's
# corpus uses (after from_tagged_corpus_to_seeds.py's cleanup - PROPN folded
# into NOUN, AUX kept distinct from VERB, etc.), used only to build readable
# 'item-<pos>' column names in confusion_words below. Any tag not listed here
# (e.g. a tagset this pipeline hasn't seen yet) still works - it just falls
# back to the tag lowercased as-is, so this table never needs to be
# exhaustive to avoid an error, only to make the common cases read nicely.
# PUNCT/BOS/EOS are deliberately NOT here - see _NON_WORD_ITEM_TAGS below,
# punctuation/sentence-boundary markers never get their own item-<pos>
# column since predictions are never made for them in the first place (see
# the _is_word_token filter on `targets` in run_extract_and_evaluate).
POS_ITEM_COLUMN_NAMES = {
    'NOUN': 'noun', 'PROPN': 'proper-noun', 'VERB': 'verb', 'AUX': 'auxiliary',
    'ADJ': 'adjective', 'ADV': 'adverb', 'DET': 'determiner', 'PRON': 'pronoun',
    'ADP': 'adposition', 'CCONJ': 'conjunction', 'CONJ': 'conjunction',
    'SCONJ': 'subordinating-conjunction', 'NUM': 'numeral', 'PART': 'particle',
    'INTJ': 'interjection', 'SYM': 'symbol', 'X': 'other',
}

# Tags that should never produce their own item-<pos> column, even if a word
# with this as its primary tag somehow ends up in confusion_words - real
# words are never classified as PUNCT/BOS/EOS overall (predictions are only
# ever attempted for actual words - see _is_word_token), so a word landing
# here would indicate something upstream let punctuation/a boundary marker
# through as a target; fall back to 'item-neither' rather than normalizing
# that with a dedicated column.
_NON_WORD_ITEM_TAGS = {'PUNCT', 'BOS', 'EOS'}


def _pos_item_col(tag):
    """'item-<pos>' column name for a given corpus tag, e.g. 'ADJ' -> 'item-adjective'."""
    return f"item-{POS_ITEM_COLUMN_NAMES.get(tag, tag.lower())}"


def strict_precision_recall(results, guess_probs=None, sorted_noun_tokens=None,
                             sorted_verb_tokens=None, word_primary_tag=None):
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
    'confusion_words': same rows, plus 'NOUN'/'VERB' prediction columns as
    before, but instead of one column per individual word the categorizer
    didn't put in NOUN/VERB, there are 'item-<pos>' summary columns - always
    including 'item-noun', 'item-verb', plus one additional column per other
    part of speech actually present (e.g. 'item-adjective', 'item-determiner',
    'item-auxiliary') - each counting the total number of TOKEN occurrences
    (not distinct word types) in that row that weren't predicted NOUN/VERB,
    bucketed by that word's own primary tag in the training corpus:
      - 'item-noun' if the word is in sorted_noun_tokens, 'item-verb' if in
        sorted_verb_tokens (the same majority-tag - N-count vs V-count -
        classification load_corpus_and_split uses elsewhere for these two
        categories specifically);
      - otherwise, 'item-<pos>' for whatever that word's single most
        frequent corpus tag is (from word_primary_tag, a full argmax over
        every tag the word was ever seen with - see load_corpus_and_split),
        via the POS_ITEM_COLUMN_NAMES friendly-name table above.
    There's no catch-all 'item-neither' column - every real word that can
    reach this point has a primary tag (word_primary_tag is built from the
    same corpus scan as everything else), so it always lands in one of the
    columns above. sorted_noun_tokens/sorted_verb_tokens/word_primary_tag
    are all optional (default None, i.e. treated as empty) so this remains
    callable without them, but callers should pass the same data used to
    build the corresponding df_contexts/categorization, so the bucketing
    means what it says.
    Also returns 'baseline': the scores a random guesser would get if it
    guessed NOUN/VERB/OTHER with probabilities guess_probs, scored against
    this run's actual test-set labels (see baseline_random_scores) - no
    confusion matrix is built for it. guess_probs is expected to come from
    compute_seed_tag_guess_probs (the proportion of training occurrences on
    the noun seed list, the verb seed list, or neither - for
    require_tag_match_true runs, seed-list membership must also agree with
    the occurrence's own tag; for require_tag_match_false runs, seed-list
    membership alone is enough), passed in by the caller. If guess_probs
    isn't provided, falls back to guessing in proportion to the test set's
    own tag frequencies instead.
    Also returns 'pattern_usage': a table of only the patterns actually used
    to classify a test-set word (not the full set of patterns extracted from
    training) - one row per pattern, with the number of times it was used,
    how many of those words were actually nouns/verbs in the test set (per
    true_mapped, as token occurrences: num_true_noun/num_true_verb) and how
    many DISTINCT noun/verb word types those occurrences represent
    (num_true_noun_types/num_true_verb_types), and the pattern's predicted
    output (a category or a specific word, whichever get_max_count_item
    picked for that pattern).
    Returns dict {per_class, micro, macro, confusion, confusion_words,
    baseline, pattern_usage}.
    """
    df = pd.DataFrame(results, columns=['token', 'pred', 'true', 'used_pattern', 'raw_pred'])
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
    confusion_words_raw = pd.crosstab(df['true'], df['pred_expanded'])
    noun_verb_cols = [c for c in ('NOUN', 'VERB') if c in confusion_words_raw.columns]
    word_cols = [c for c in confusion_words_raw.columns if c not in ('NOUN', 'VERB')]

    noun_set = set(sorted_noun_tokens or [])
    verb_set = set(sorted_verb_tokens or [])
    word_primary_tag = word_primary_tag or {}

    def _item_bucket(word):
        if word in noun_set:
            return 'item-noun'
        elif word in verb_set:
            return 'item-verb'
        tag = word_primary_tag.get(word)
        if tag and tag not in _NON_WORD_ITEM_TAGS:
            # Whatever this word's actual majority tag is - if that happens
            # to be NOUN/VERB (a word sorted_noun_tokens/sorted_verb_tokens'
            # N-count-vs-V-count comparison didn't catch, e.g. a near-tie),
            # _pos_item_col naturally produces 'item-noun'/'item-verb' too,
            # merging into the same column rather than needing a special case.
            return _pos_item_col(tag)
        # Every real word should be covered by one of the branches above -
        # word_primary_tag is built from the same corpus scan as everything
        # else, and punctuation/boundary markers (_NON_WORD_ITEM_TAGS) are
        # already excluded from ever being a classification target upstream
        # (see the _is_word_token guards in extract_context_patterns_fast/
        # categorize_with_contexts_fast), so this should never actually be
        # reached. Kept only as a safety-net label - deliberately NOT added
        # to base_item_cols below, so a plain 'item-noun'/'item-verb'/
        # 'item-<pos>' schema is all that ever shows up in practice; if this
        # ever DID get hit, it would still surface as its own column (via
        # the same dynamic-extra-column mechanism as item-<pos>) rather than
        # silently disappearing.
        return 'item-neither'

    # Base columns are always present for a stable schema across runs/files;
    # any other part of speech actually seen among this run's OTHER-predicted
    # words gets its own additional column, appended in alphabetical order.
    # 'item-neither' is deliberately NOT a base column - see _item_bucket -
    # it's redundant now that every real word gets its own item-noun/
    # item-verb/item-<pos> column, so it only reappears (dynamically) if
    # something unexpected actually needs it.
    base_item_cols = ['item-noun', 'item-verb']
    confusion_words = confusion_words_raw[noun_verb_cols].copy()
    if word_cols:
        # Sum raw token-occurrence counts (not distinct word types) per
        # bucket - confusion_words_raw[w] for a given row is how many
        # occurrences of word w had that true tag.
        buckets_by_word = {w: _item_bucket(w) for w in word_cols}
        extra_item_cols = sorted(set(buckets_by_word.values()) - set(base_item_cols))
        item_cols = base_item_cols + extra_item_cols
        for bucket in item_cols:
            cols_in_bucket = [w for w in word_cols if buckets_by_word[w] == bucket]
            confusion_words[bucket] = confusion_words_raw[cols_in_bucket].sum(axis=1) if cols_in_bucket else 0
    else:
        item_cols = base_item_cols
        for bucket in item_cols:
            confusion_words[bucket] = 0

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

    baseline = baseline_random_scores(df['true_mapped'], guess_probs=guess_probs)

    # Patterns actually used to classify a test-set word - not the full set
    # of patterns extract_context_patterns_fast produced from training, only
    # the ones get_max_count_item actually matched at test time (see
    # categorize_with_contexts_fast). used_pattern is None for words where
    # neither the primary nor fallback pattern matched any trained column at
    # all (nothing to report there, so those rows are excluded).
    used_df = df[df['used_pattern'].notna()]
    pattern_usage_cols = [
        'uses', 'num_true_noun', 'num_true_verb',
        'num_true_noun_types', 'num_true_verb_types', 'predicted',
    ]
    if len(used_df) > 0:
        pattern_usage = used_df.groupby('used_pattern').agg(
            uses=('used_pattern', 'size'),
            num_true_noun=('true_mapped', lambda s: int((s == 'NOUN').sum())),
            num_true_verb=('true_mapped', lambda s: int((s == 'VERB').sum())),
            predicted=('raw_pred', 'first'),
        )
        # num_true_noun/num_true_verb above count TOKEN occurrences; these
        # count DISTINCT word types among them instead - e.g. if a pattern
        # was used for "cat" three times and "dog" once, all as true nouns,
        # num_true_noun=4 but num_true_noun_types=2.
        noun_type_counts = (
            used_df[used_df['true_mapped'] == 'NOUN']
            .groupby('used_pattern')['token'].nunique()
        )
        verb_type_counts = (
            used_df[used_df['true_mapped'] == 'VERB']
            .groupby('used_pattern')['token'].nunique()
        )
        pattern_usage['num_true_noun_types'] = noun_type_counts.reindex(pattern_usage.index, fill_value=0).astype(int)
        pattern_usage['num_true_verb_types'] = verb_type_counts.reindex(pattern_usage.index, fill_value=0).astype(int)
        pattern_usage = pattern_usage[pattern_usage_cols]
        pattern_usage.index.name = 'pattern'
        pattern_usage = pattern_usage.sort_values('uses', ascending=False)
    else:
        pattern_usage = pd.DataFrame(columns=pattern_usage_cols).rename_axis('pattern')

    print(detailed_confusion)
    return {
        'per_class': per_class,
        'micro': {'precision': micro_p, 'recall': micro_r, 'f1': micro_f},
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f},
        'confusion': detailed_confusion,
        'confusion_words': confusion_words,
        'baseline': baseline,
        'pattern_usage': pattern_usage,
    }

### RUNTIME CODE STARTS HERE ###

def make_kfold_sentence_splits(n_sentences, n_folds, split_seed):
    """
    Partition range(n_sentences) into n_folds folds of nearly-equal size,
    shuffled deterministically by split_seed, and return a list of
    (train_idx_set, test_idx_set) pairs - one per fold, fold i's test set
    being fold i's chunk and its train set being every other chunk.

    Folds are sized as evenly as possible: n_sentences % n_folds of the
    folds get one extra sentence, so no fold differs from another by more
    than one sentence. The shuffle + chunking is deterministic given
    split_seed, so every independent process (see run_cluster.sh)
    reproduces the exact same folds.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_folds > n_sentences:
        raise ValueError(
            f"n_folds ({n_folds}) exceeds the corpus's {n_sentences} sentences"
        )

    rng = random.Random(split_seed)
    order = list(range(n_sentences))
    rng.shuffle(order)

    base, remainder = divmod(n_sentences, n_folds)
    chunks = []
    start = 0
    for fold in range(n_folds):
        size = base + (1 if fold < remainder else 0)
        chunks.append(order[start:start + size])
        start += size

    splits = []
    for fold in range(n_folds):
        test_idx_set = set(chunks[fold])
        train_idx_set = set().union(*(chunks[j] for j in range(n_folds) if j != fold))
        splits.append((train_idx_set, test_idx_set))
    return splits


def load_corpus_and_split(corpus_file, split_seed=42, test_fraction=0.2,
                           corpus_size=None, subsample_scope="train_only",
                           n_folds=5):
    """
    Reads the WORD_LEMMA_TAG corpus file, builds the flat token/tag lists
    (lemma-lowercased, sentence-bounded by "{"/"}"), then splits it by
    sentence, deterministically given split_seed (so every independent
    process that calls this with the same arguments reproduces the exact
    same split(s) - this is what lets run_cluster.sh dispatch each (mode,
    pattern_type, seed-set) combination to its own independent process/core
    without sharing any state: each process just redoes this same
    deterministic corpus load+split itself).

    n_folds: how to split.
        n_folds >= 2 (default 5) - k-FOLD CROSS-VALIDATION: the corpus (or
            its corpus_size subsample - see below) is partitioned into
            n_folds folds (see make_kfold_sentence_splits); every sentence
            is used as test exactly once, across the n_folds folds
            train/test pairs returned. test_fraction is ignored (each fold's
            test set is ~1/n_folds of the pool by construction).
            subsample_scope is also ignored when corpus_size is given - a
            fixed held-out test set independent of corpus_size doesn't apply
            under cross-validation, since every sentence rotates through the
            test role - so corpus_size always subsamples the whole pool
            first (like subsample_scope="whole_corpus"), then folds it.
        n_folds == 1 - the ORIGINAL single train/test split behavior:
            holds out test_fraction of sentences as a single fixed test set,
            honoring subsample_scope exactly as before. Use this to
            reproduce pre-cross-validation runs.

    corpus_size: if given, randomly subsample down to this many sentences
        (utterances) instead of using the full corpus. None (default) means
        no subsampling - use every sentence, exactly as before. Which
        sentences this affects depends on subsample_scope (n_folds==1 only -
        see above):
          "train_only" (default) - the held-out test set is always the same
              test_fraction of sentences drawn from the FULL corpus, i.e.
              unaffected by corpus_size, so results across different corpus
              sizes stay comparable against one fixed test set. Only the
              training pool is subsampled down to corpus_size sentences (if
              corpus_size is at least as large as the full training pool,
              the full pool is used - no error).
          "whole_corpus" - the full corpus is subsampled down to
              corpus_size sentences FIRST, then split test_fraction/
              (1 - test_fraction) as usual, so both train and test shrink
              and the test set itself changes between corpus sizes.
        Sampling is deterministic given split_seed (same seed used for the
        train/test split itself), so every independent process reproduces
        the exact same subsample.

    Returns (folds, token_counts, sorted_noun_tokens, sorted_verb_tokens,
    word_primary_tag).

    folds: a list of per-split dicts, each with keys 'fold', 'train',
    'test', 'train_tags', 'test_tags', 'train_words', 'test_words' (the last
    six exactly as train/test/train_tags/test_tags/train_words/test_words
    used to be returned directly - see below). When n_folds==1, folds is a
    single-element list with 'fold'=None (not under cross-validation - keeps
    single-split callers/output filenames free of any fold labeling). When
    n_folds>=2, folds has n_folds elements with 'fold'=0..n_folds-1.

    token_counts, sorted_noun_tokens, sorted_verb_tokens, word_primary_tag
    are computed over the WHOLE corpus (see below) and so are shared/
    identical across every fold - returned once, not per-fold.

    train_words/test_words: the SURFACE WORD FORM aligned index-for-index
    with train/test (which hold the LEMMA, as before, lowercased and
    sentence-bounded the same way). Lemma is used only for seed selection
    (from_tagged_corpus_to_seeds.py, unaffected by this) and for matching
    seeds to tokens when learning patterns (the is_noun/is_verb checks in
    extract_context_patterns_fast/compute_seed_tag_guess_probs/
    categorize_with_contexts_fast, which all take the lemma arrays) -
    everything else, including the pattern text itself, the lexical filler
    recorded for a target/context word, token_counts (test-time target
    eligibility), and sorted_noun_tokens/sorted_verb_tokens (the
    all_tagged_nouns_verbs word pool and confusion_words' item-noun/
    item-verb/item-neither bucketing) is now based on the surface word form.

    word_primary_tag: {surface word form -> its single most frequent corpus
    tag (e.g. "DET", "ADJ", "AUX", ...), computed over the WHOLE corpus (same
    scope as noun_tokens/verb_tokens above), ties broken by whichever tag
    Python's max() sees first for that word. Used by strict_precision_recall
    to break confusion_words' 'item-neither' bucket down into one column per
    part of speech (e.g. 'item-adjective', 'item-determiner') for words that
    aren't primarily nouns or verbs, instead of lumping them all together.
    """
    noun_tokens = defaultdict(int)
    verb_tokens = defaultdict(int)
    token_counts = defaultdict(int)
    # Per-word tag frequency, over EVERY tag (not just N/V) - used to derive
    # word_primary_tag below, which drives confusion_words' per-part-of-speech
    # 'item-<pos>' bucketing for words that aren't primarily nouns or verbs.
    word_tag_counts = defaultdict(lambda: defaultdict(int))
    tokens = []
    words = []
    tags = []
    with open(corpus_file) as file:
        for line in file:
            tokens.append("{")
            words.append("{")
            tags.append("BOS")
            line_array = line.split()
            for element in line_array:
                # File format is WORD_LEMMA_TAG (e.g. "thought_think_VERB"),
                # with word/lemma each potentially themselves containing
                # underscores for multi-word compounds - see
                # _split_word_lemma_tag for how those are disambiguated.
                surface, w, t = _split_word_lemma_tag(element)
                tokens.append(str.lower(w))
                words.append(str.lower(surface))
                tags.append(t)
                token_counts[str.lower(surface)] += 1
                word_tag_counts[str.lower(surface)][t] += 1
                if re.match(r"^N", t):
                    noun_tokens[str.lower(surface)] += 1
                if re.match(r"^V", t):
                    verb_tokens[str.lower(surface)] += 1
            tokens.append("}")
            words.append("}")
            tags.append("EOS")

    sorted_noun_counts = sorted(noun_tokens.items(), key=lambda item: item[1], reverse=True)
    sorted_verb_counts = sorted(verb_tokens.items(), key=lambda item: item[1], reverse=True)
    sorted_noun_tokens = [x for x, _ in sorted_noun_counts]
    sorted_verb_tokens = [x for x, _ in sorted_verb_counts]
    excluded_nouns = ["mummy", "daddy", "john", "carl", "dominic"]
    excluded_verbs = [""]
    sorted_noun_tokens = [x for x in sorted_noun_tokens if x not in excluded_nouns and noun_tokens[x] > verb_tokens[x]]
    sorted_verb_tokens = [x for x in sorted_verb_tokens if x not in excluded_verbs and verb_tokens[x] > noun_tokens[x]]

    # word_primary_tag: each word's single most-frequent tag (e.g. "DET",
    # "ADJ", "AUX") over the whole corpus - ties broken arbitrarily but
    # deterministically by dict iteration order (insertion order, i.e. the
    # tag first seen for that word). Deliberately independent of
    # sorted_noun_tokens/sorted_verb_tokens above (which only compare N vs V
    # counts) - this is a full argmax over every tag a word was ever seen
    # with, used purely for confusion_words' finer-grained item-<pos>
    # breakdown, not for seed selection or all_tagged_nouns_verbs.
    word_primary_tag = {
        w: max(tag_counts.items(), key=lambda kv: kv[1])[0]
        for w, tag_counts in word_tag_counts.items()
    }

    # Split by utterance (sentence). Find all indices where a sentence ends
    # ("}"), then derive (start, end) bounds for each sentence (a sentence
    # runs from just after the previous "}" through its own "}", inclusive).
    sentence_end_indices = [i for i, tok in enumerate(tokens) if tok == "}"]
    sentence_bounds = []
    start = 0
    for end in sentence_end_indices:
        sentence_bounds.append((start, end))
        start = end + 1

    n_sentences = len(sentence_bounds)
    if corpus_size is not None:
        if corpus_size < 1:
            raise ValueError(f"--corpus-size must be at least 1, got {corpus_size}")
        if corpus_size > n_sentences:
            raise ValueError(
                f"--corpus-size {corpus_size} exceeds the corpus's {n_sentences} sentences"
            )
        if subsample_scope not in ("train_only", "whole_corpus"):
            raise ValueError(f"Unknown subsample_scope {subsample_scope!r}")

    # Single RNG instance, seeded once, shared by every sampling step below -
    # deterministic given split_seed, so every independent process (see
    # run_cluster.sh) reproduces the exact same draws in the exact same
    # order.
    rng = random.Random(split_seed)

    def _extract(train_idx_set, test_idx_set):
        train, test, train_tags, test_tags, train_words, test_words = [], [], [], [], [], []
        for i, (s, e) in enumerate(sentence_bounds):
            if i in test_idx_set:
                test.extend(tokens[s:e + 1])
                test_tags.extend(tags[s:e + 1])
                test_words.extend(words[s:e + 1])
            elif i in train_idx_set:
                train.extend(tokens[s:e + 1])
                train_tags.extend(tags[s:e + 1])
                train_words.extend(words[s:e + 1])
            # else: excluded by subsampling - neither train nor test.
        return train, test, train_tags, test_tags, train_words, test_words

    if n_folds == 1:
        # Original single train/test split behavior: a single fixed
        # test_fraction test set, honoring subsample_scope.
        if corpus_size is not None and subsample_scope == "whole_corpus":
            # Subsample the whole corpus down to corpus_size sentences
            # first, so both train and test shrink together (test set
            # changes between corpus sizes).
            sentence_pool = rng.sample(range(n_sentences), corpus_size)
        else:
            sentence_pool = list(range(n_sentences))

        # Test set: test_fraction of sentence_pool. When subsample_scope
        # isn't "whole_corpus" (including corpus_size=None), sentence_pool
        # is always the full range(n_sentences) here, so this draw - and
        # therefore the resulting test set - is identical regardless of
        # corpus_size, exactly as the "train_only" scope requires.
        n_test = int(len(sentence_pool) * test_fraction)
        test_sentence_idx = set(rng.sample(sentence_pool, n_test))
        train_pool = [i for i in sentence_pool if i not in test_sentence_idx]

        if corpus_size is not None and subsample_scope == "train_only" and corpus_size < len(train_pool):
            train_pool = rng.sample(train_pool, corpus_size)
        train_idx_set = set(train_pool)

        train, test, train_tags, test_tags, train_words, test_words = _extract(train_idx_set, test_sentence_idx)

        if corpus_size is not None:
            print(
                f"Corpus subsampled (scope={subsample_scope}, corpus_size={corpus_size}): "
                f"{len(train_idx_set)} train sentence(s), {len(test_sentence_idx)} test sentence(s) "
                f"(of {n_sentences} total)."
            )

        folds = [{
            "fold": None,
            "train": train, "test": test,
            "train_tags": train_tags, "test_tags": test_tags,
            "train_words": train_words, "test_words": test_words,
        }]
    else:
        # k-fold cross-validation: every sentence is used as test exactly
        # once, across n_folds train/test pairs. corpus_size (if given)
        # always subsamples the whole pool first (subsample_scope is
        # ignored - see docstring), THEN that pool is folded.
        if corpus_size is not None:
            sentence_pool = rng.sample(range(n_sentences), corpus_size)
        else:
            sentence_pool = list(range(n_sentences))

        # make_kfold_sentence_splits partitions range(len(sentence_pool)) -
        # i.e. positions WITHIN sentence_pool, not raw corpus sentence
        # indices - map each fold's local indices back through
        # sentence_pool to get real sentence_bounds indices.
        fold_splits = make_kfold_sentence_splits(len(sentence_pool), n_folds, split_seed)
        folds = []
        for fold_i, (train_local, test_local) in enumerate(fold_splits):
            train_idx_set = {sentence_pool[j] for j in train_local}
            test_idx_set = {sentence_pool[j] for j in test_local}
            train, test, train_tags, test_tags, train_words, test_words = _extract(train_idx_set, test_idx_set)
            folds.append({
                "fold": fold_i,
                "train": train, "test": test,
                "train_tags": train_tags, "test_tags": test_tags,
                "train_words": train_words, "test_words": test_words,
            })

        if corpus_size is not None:
            print(
                f"Corpus subsampled to corpus_size={corpus_size} sentence(s) "
                f"(of {n_sentences} total) before {n_folds}-fold split."
            )
        print(f"{n_folds}-fold cross-validation: {len(sentence_pool)} sentence(s) split into {n_folds} folds.")

    return (
        folds,
        token_counts, sorted_noun_tokens, sorted_verb_tokens, word_primary_tag,
    )


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
    parser.add_argument(
        "--max-cum-prop-threshold", type=float, default=0.239,
        help="Cap on the noun seed-list size tested, expressed as cumulative "
             "proportion of noun tokens (e.g. 0.239 = include "
             "highest-frequency nouns up to the point where they account for "
             "23.9%% of noun tokens, then stop - or the full Include==1 noun "
             "list if it's smaller). The seed-set sweep is a MATCHED "
             "SEQUENCE, NOT a cross product: for each noun count 1..N below "
             "this cap, it's paired with whichever verb count(s) cover the "
             "matching share of verb tokens (see compute_seed_steps) - e.g. "
             "N=36 gives 45 total (num_nouns, num_verbs) pairings with the "
             "current seed files, not 36*33=1188. Default 0.239 (chosen so "
             "verbs - only 33 curated words, covering 23.9%% of verb tokens "
             "at most - are always fully matched, while nouns are capped to "
             "a comparable ~36-word list with the current seed files).",
    )
    parser.add_argument(
        "--num-sweep-steps", type=int, default=None,
        help="Optional EXTRA cap on how many noun counts are considered "
             "(verb counts are derived from the matched noun counts, not "
             "swept independently) - e.g. 20 means only noun counts 1..20 "
             "are used even if --max-cum-prop-threshold would otherwise "
             "allow more. Default: unset - --max-cum-prop-threshold alone "
             "determines the sequence.",
    )
    parser.add_argument(
        "--print-num-seed-steps", action="store_true",
        help="Print the number of (num_nouns, num_verbs) pairings that "
             "--mode require_tag_match_true/require_tag_match_false would "
             "sweep over with the given --noun-seeds-file/--verb-seeds-file/"
             "--max-cum-prop-threshold/--num-sweep-steps, then exit "
             "immediately (skips loading/splitting the corpus - seed-set "
             "sizes don't depend on it). Useful for a dispatcher (e.g. "
             "run_cluster.sh) that needs to know how many --seed-step "
             "values 0..N-1 are valid before generating that many jobs.",
    )
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument(
        "--no-abstract-context", dest="abstract_context", action="store_false",
        default=True,
        help="Disable noun/verb abstraction of CONTEXT words - context tokens "
             "are left as literal surface forms instead of being collapsed to "
             "\"noun\"/\"verb\". Default is abstraction enabled (original "
             "behavior). Target-word classification (NOUN/VERB row labels) is "
             "unaffected.",
    )
    parser.add_argument(
        "--emit-target-words", action="store_true", default=False,
        help="Also write a learned_patterns_words_*.xlsx alongside each "
             "learned_patterns_*.xlsx, with the target's literal surface word "
             "always in a 'filler' column (never collapsed to \"NOUN\"/\"VERB\") "
             "and its status in a separate 'category' column (\"NOUN\"/\"VERB\"/"
             "\"OTHER\"). Purely an additional reporting output - does not change "
             "df_contexts, learned_patterns_*.xlsx, summary.csv, or any "
             "evaluation metric. Default off (original behavior/file set).",
    )
    parser.add_argument("--out-dir", default="sweep_out")
    parser.add_argument("--corpus-file", default="manchester_input_tagged_trf_word_and_lemma_postprocessed.txt")
    parser.add_argument("--noun-seeds-file", default="noun_selection.xlsx")
    parser.add_argument("--verb-seeds-file", default="verb_selection.xlsx")
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of folds for k-fold cross-validation (default: 5). Every "
             "sentence is used as test exactly once, across --n-folds train/test "
             "splits, each run independently - see evaluate_kfold_and_aggregate. "
             "summary.csv gets one row per fold plus a 'mean' row (mean + std "
             "across folds); learned_patterns/confusion_words/pattern_usage are "
             "written once per fold (filenames get a _foldN suffix). Pass "
             "--n-folds 1 to instead reproduce the original single 80/20-style "
             "split (see --test-fraction/--split-seed/--subsample-scope, only "
             "relevant when --n-folds 1).",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of sentences held out as the test set. Only used when "
             "--n-folds 1 (ignored under cross-validation, where each fold's test "
             "set is ~1/--n-folds by construction).",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--corpus-size", type=int, default=None,
        help="Randomly subsample the corpus down to this many sentences instead of "
             "using the full corpus. Default (unset) uses the full corpus. See "
             "--subsample-scope for what exactly gets subsampled (--n-folds 1 only "
             "- under cross-validation this always subsamples the whole pool "
             "before folding, regardless of --subsample-scope).",
    )
    parser.add_argument(
        "--subsample-scope", choices=["train_only", "whole_corpus"], default="train_only",
        help="Only relevant when --corpus-size is given AND --n-folds 1. "
             "'train_only' (default) keeps the held-out test set fixed (always "
             "the same test_fraction of the FULL corpus) and only subsamples the "
             "training pool down to --corpus-size, so results across different "
             "corpus sizes stay comparable against one fixed test set. "
             "'whole_corpus' subsamples the full corpus down to --corpus-size "
             "sentences first, then splits as usual, so the test set also shrinks "
             "and changes between corpus sizes.",
    )
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

    if args.print_num_seed_steps:
        # Seed-set sequence length doesn't depend on the corpus split at all
        # - only on the seed files + max-cum-prop-threshold/num-sweep-steps -
        # so skip load_corpus_and_split entirely here (it can be slow) and just
        # replicate the seed-loading steps below.
        noun_seeds = pd.read_excel(args.noun_seeds_file)
        verb_seeds = pd.read_excel(args.verb_seeds_file)
        noun_seeds = add_proportion(noun_seeds)
        verb_seeds = add_proportion(verb_seeds)
        noun_seeds, verb_seeds = resolve_noun_verb_seed_overlap(noun_seeds, verb_seeds)
        steps, _, _ = compute_seed_steps(
            noun_seeds, verb_seeds,
            max_cum_prop_threshold=args.max_cum_prop_threshold,
            max_sweep_steps=args.num_sweep_steps,
        )
        print(len(steps))
        return

    (folds, token_counts,
     sorted_noun_tokens, sorted_verb_tokens, word_primary_tag) = load_corpus_and_split(
        args.corpus_file, split_seed=args.split_seed, test_fraction=args.test_fraction,
        corpus_size=args.corpus_size, subsample_scope=args.subsample_scope,
        n_folds=args.n_folds,
    )

    noun_seeds = pd.read_excel(args.noun_seeds_file)
    verb_seeds = pd.read_excel(args.verb_seeds_file)
    # Precompute each word's proportion of the FULL vocabulary once, up
    # front. The cumulative proportion used to pick seeds is computed later,
    # on the fly (see compute_seed_steps), after filtering down to
    # Include==1 words.
    noun_seeds = add_proportion(noun_seeds)
    verb_seeds = add_proportion(verb_seeds)
    # A word can be Include==1 in both seed lists at once (e.g. "walk" as
    # both noun and verb) - resolve that by keeping it a seed only in
    # whichever category it accounts for the larger share of (see
    # resolve_noun_verb_seed_overlap). Must run after add_proportion (needs
    # PROPORTION) and before any Include-based filtering/seed selection
    # below (compute_seed_steps, all_tagged_nouns_verbs path, etc.).
    noun_seeds, verb_seeds = resolve_noun_verb_seed_overlap(noun_seeds, verb_seeds)

    if args.mode is None:
        # Original single-machine behavior: run the full in-process
        # comparison across all three modes and all three pattern types,
        # sequentially, writing straight to the shared summary.csv/
        # confusion_matrices.txt.
        summary_csv = run_mode_comparison(
            run_extract_and_evaluate,
            folds,
            noun_seeds, verb_seeds,
            token_counts, sorted_noun_tokens, sorted_verb_tokens,
            word_primary_tag=word_primary_tag,
            out_dir=args.out_dir,
            pattern_types=(1, 2, 3),
            num_sweep_steps=args.num_sweep_steps,
            max_cum_prop_threshold=args.max_cum_prop_threshold,
            window_size=args.window_size,
            abstract_context=args.abstract_context,
            track_target_words=args.emit_target_words,
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
        # Use every word tagged noun/verb in the postprocessed training
        # corpus (already computed by load_corpus_and_split from the
        # corpus's own tags), not the curated (Include==1) seed list - see
        # the matching comment in run_extract_and_evaluate_sweep.
        selected_nouns = list(sorted_noun_tokens)
        selected_verbs = list(sorted_verb_tokens)
        # Each fold's OWN row (and artifact filenames) report the actual
        # noun/verb counts from THAT fold's own training data, recomputed
        # inside evaluate_kfold_and_aggregate. This fold-0-based count is
        # only used below to build a stable job_id - the summary_parts/
        # confusion_parts file pair covers every fold, not just one, so it
        # can't embed a single fully-accurate-for-every-fold n/v anyway.
        num_nouns, num_verbs = compute_all_tagged_counts(folds[0]["train_words"], folds[0]["train_tags"])
        require_tag_match = False
        all_tagged = True
        step_label = "full"
    else:
        require_tag_match = (args.mode == "require_tag_match_true")
        all_tagged = False
        if args.seed_step is None:
            raise SystemExit(f"--seed-step is required for --mode {args.mode}")
        steps, noun_seeds_f, verb_seeds_f = compute_seed_steps(
            noun_seeds, verb_seeds,
            max_cum_prop_threshold=args.max_cum_prop_threshold,
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

    (rows, confusion_texts, confusion_words_list, pattern_usage_list,
     learned_patterns_list, learned_patterns_words_list) = evaluate_kfold_and_aggregate(
        run_extract_and_evaluate, folds,
        selected_nouns, selected_verbs, num_nouns, num_verbs,
        token_counts, sorted_noun_tokens, sorted_verb_tokens,
        word_primary_tag=word_primary_tag,
        window_size=args.window_size, pattern_type=args.pattern_type,
        require_tag_match=require_tag_match,
        all_tagged_nouns_verbs=all_tagged, abstract_context=args.abstract_context,
        track_target_words=args.emit_target_words,
        run_mode=args.mode,
    )

    run_mode_safe = re.sub(r"[^A-Za-z0-9_-]+", "_", args.mode)
    job_id = f"{run_mode_safe}_p{args.pattern_type}_{step_label}_n{num_nouns}_v{num_verbs}"

    pd.DataFrame(rows)[SUMMARY_COLS].to_csv(os.path.join(parts_dir, f"{job_id}.csv"), index=False)
    with open(os.path.join(conf_parts_dir, f"{job_id}.txt"), "w", encoding="utf-8") as f:
        for _fold_label, confusion_text in confusion_texts:
            f.write(confusion_text)

    def _fold_suffix(fold_label):
        return "" if fold_label is None else f"_fold{fold_label}"

    for fold_label, _n, _v, confusion_words in confusion_words_list:
        if confusion_words is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            words_csv_path = os.path.join(args.out_dir, f"confusion_words_{job_id}{_fold_suffix(fold_label)}_{ts}.csv")
            confusion_words.to_csv(words_csv_path)
            print(f"Word-level confusion breakdown written to {words_csv_path}")

    for fold_label, _n, _v, pattern_usage in pattern_usage_list:
        if pattern_usage is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            pattern_usage_csv_path = os.path.join(args.out_dir, f"pattern_usage_{job_id}{_fold_suffix(fold_label)}_{ts}.csv")
            pattern_usage.to_csv(pattern_usage_csv_path)
            print(f"Pattern usage breakdown written to {pattern_usage_csv_path}")

    for fold_label, _n, _v, learned_patterns in learned_patterns_list:
        if learned_patterns is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            learned_patterns_xlsx_path = os.path.join(args.out_dir, f"learned_patterns_{job_id}{_fold_suffix(fold_label)}_{ts}.xlsx")
            learned_patterns.to_excel(learned_patterns_xlsx_path, index=False)
            print(f"Learned patterns/fillers written to {learned_patterns_xlsx_path}")

    for fold_label, _n, _v, learned_patterns_words in learned_patterns_words_list:
        if learned_patterns_words is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            learned_patterns_words_xlsx_path = os.path.join(args.out_dir, f"learned_patterns_words_{job_id}{_fold_suffix(fold_label)}_{ts}.xlsx")
            learned_patterns_words.to_excel(learned_patterns_words_xlsx_path, index=False)
            print(f"Learned patterns/target-words (literal + category) written to {learned_patterns_words_xlsx_path}")

    print(f"Single-run result written to {parts_dir}/{job_id}.csv and {conf_parts_dir}/{job_id}.txt")


if __name__ == "__main__":
    main()

