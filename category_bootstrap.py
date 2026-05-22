import re
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


def extract_context_patterns_fast(corpus, seeds, window_size=2, dtype=np.int32, pattern_type=1):
    types = sorted(set(corpus))
    types.extend(["NOUN", "VERB"])
    types_to_idx = {t: i for i, t in enumerate(types)}

    noun_set = set(seeds.get('nouns', []))
    verb_set = set(seeds.get('verbs', []))

    contexts = []
    context_to_idx = {}
    rows = []
    cols = []
    data = []

    for i, word in enumerate(corpus):
        if word not in ["{" , "}"]:
            begin = max(i - window_size, 0)
            end = min(i + window_size, len(corpus))
            context = corpus[begin: end + 1]
            del context[i-begin]

            context = ["verb" if w in verb_set else w for w in context]
            context = ["noun" if w in noun_set else w for w in context]
            if len(context) == 4:
                if word in noun_set:
                    seed_id = types_to_idx["NOUN"]
                elif word in verb_set:
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
    #df = pd.DataFrame.sparse.from_spmatrix(coo, index=types, columns=contexts)
    dense = coo.tocsr().toarray().astype(int)
    df = pd.DataFrame(dense, index=types, columns=contexts)
   
    return df


def _trim_braces_fast(s: str) -> str:
    i = s.find('{')
    if i != -1:
        j = s.rfind('}')
        if j >= i:
            return s[i:j+1]
    return s


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
):
    """
    Run extraction on train_tokens, categorize test_tokens (with test_tags),
    and compute strict precision/recall.
    Returns: (results, metrics, df_contexts)
    - results: list of (token, pred, true_tag) tuples from categorize_with_contexts_fast
    - metrics: output of strict_precision_recall(results)
    - df_contexts: DataFrame from extract_context_patterns_fast
    """
    seeds = {'nouns': selected_noun_seeds, 'verbs': selected_verb_seeds}

    df_contexts = extract_context_patterns_fast(train_tokens, seeds, window_size=window_size, pattern_type=pattern_type)

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
        tags=test_tags
    )

    metrics = strict_precision_recall(results)
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
                                  window_size=2, tags=None, pattern_type=1):
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
        end_excl = min(i + window_size, token_count) + 1
        context = tokens[begin:end_excl]
        del context[i-begin]
        for j in range(len(context)):
            w = context[j]
            if w in selected_noun_set:
                context[j] = "noun"
            elif w in selected_verb_set:
                context[j] = "verb"

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


def sweep_and_save_runs(
    run_fn, train_tokens, test_tokens, test_tags,
    noun_seeds_df, verb_seeds_df,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    out_dir="sweep_runs",
    cum_prop_threshold=0.1,
    target_prob_cutoff=0.0005,
    window_size=2, pattern_type=1,
):
    import os
    import time
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    confusion_path = os.path.join(out_dir, "confusion_matrices.txt")

    def _find_cumcol(df):
        for c in df.columns:
            if "CUMULATIVE" in str(c).upper():
                return c
        raise KeyError("No cumulative-proportion column found")

    m_col = _find_cumcol(noun_seeds_df)
    n_col = _find_cumcol(verb_seeds_df)
    base_m = int((noun_seeds_df[m_col] < cum_prop_threshold).sum())
    base_n = int((verb_seeds_df[n_col] < cum_prop_threshold).sum())
    base_m = max(1, base_m)
    base_n = max(1, base_n)

    total_noun = len(noun_seeds_df)
    total_verb = len(verb_seeds_df)

    summary_cols = [
        "time", "num_noun_seeds", "num_verb_seeds", "runtime_s",
        "NOUN_precision", "NOUN_recall",
        "VERB_precision", "VERB_recall",
        "macro_precision", "macro_recall",
        "micro_precision", "micro_recall"
    ]
    if not os.path.exists(summary_path):
        pd.DataFrame(columns=summary_cols).to_csv(summary_path, index=False)

    multiplier = 1
    seen = set()
    while True:
        num_nouns = min(total_noun, base_m * multiplier)
        num_verbs = min(total_verb, base_n * multiplier)
        if (num_nouns, num_verbs) in seen:
            break
        seen.add((num_nouns, num_verbs))
        print(f"\nRunning with num_noun_seeds={num_nouns}, num_verb_seeds={num_verbs}...")
        selected_nouns = noun_seeds_df.iloc[:num_nouns]['Word'].tolist()
        selected_verbs = verb_seeds_df.iloc[:num_verbs]['Word'].tolist()
       
        t0 = time.time()
        metrics = run_fn(
            train_tokens, test_tokens, test_tags,
            selected_nouns, selected_verbs,
            token_counts, sorted_noun_tokens, sorted_verb_tokens,
            target_prob_cutoff=target_prob_cutoff, window_size=window_size,
            pattern_type=pattern_type, 
        )
        t1 = time.time()

        per_class = metrics['per_class']
        macro = metrics['macro']
        micro = metrics['micro']
        confusion = metrics['confusion']

        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
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
        }
        pd.DataFrame([row]).to_csv(summary_path, mode="a", header=False, index=False)

        with open(confusion_path, "a", encoding="utf-8") as f:
            f.write(f"num_noun_seeds={num_nouns}, num_verb_seeds={num_verbs}, time={row['time']}\n")
            f.write(confusion.to_string())
            f.write("\n\n")

        if num_nouns >= total_noun and num_verbs >= total_verb:
            break
        multiplier *= 2

    return summary_path, confusion_path

def strict_precision_recall(results):
    """
    Scoring per your rules:
      - map preds -> 'NOUN'/'VERB' else 'OTHER'
      - map trues -> 'NOUN'/'VERB' else 'OTHER'
      - For class L in {NOUN, VERB}:
          TP = pred==L and true==L
          FP = pred==L and true!=L
          FN = true==L and pred!=L
      - Predictions == OTHER are never counted as TP.
    Returns dict {per_class, micro, macro, confusion}.
    """
    df = pd.DataFrame(results, columns=['token','pred','true'])
    df['pred_mapped'] = df['pred'].where(df['pred'].isin(['NOUN','VERB']), 'OTHER')
    df['true_mapped'] = df['true'].where(df['true'].isin(['NOUN','VERB']), 'OTHER')

    labels = ['NOUN', 'VERB', 'OTHER']
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
    print(conf)
    return {
        'per_class': per_class,
        'micro': {'precision': micro_p, 'recall': micro_r, 'f1': micro_f},
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f},
        'confusion': conf
    }

### RUNTIME CODE STARTS HERE ###
token_counts=defaultdict(int)
tokens=[]
tags=[]
#df.index = seeds

# a) Import and preprocess data, decide on seeds
filename = "manchester_input_tagged_trf_postprocessed.txt"
noun_tokens = defaultdict(int)
verb_tokens = defaultdict(int)
token_counts = defaultdict(int)
tokens = []
tags = []
with open(filename) as file:
    for line in file:
        #print(line)
        tokens.append("{")
        tags.append("BOS")
        line_array = line.split()
        for element in line_array:
            la = re.match(r"([^ ]+)_([^ ]+)", element)
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

# Seed selection from Excel
noun_seeds = pd.read_excel("noun_selection.xlsx")
#selected_noun_seeds = noun_seeds[noun_seeds['CUMULATIVE PROPORTION '] < 0.1]['Word'].tolist()
verb_seeds = pd.read_excel("verb_selection.xlsx")
#selected_verb_seeds = verb_seeds[verb_seeds['CUMULATIVE_PROPORTION'] < 0.1]['Word'].tolist()
#selected_noun_seeds = [w.lower() for w in selected_noun_seeds]
#selected_verb_seeds = [w.lower() for w in selected_verb_seeds]

# Sort and filter tokens for seed refinement
sorted_noun_counts = sorted(noun_tokens.items(), key=lambda item: item[1], reverse=True)
sorted_verb_counts = sorted(verb_tokens.items(), key=lambda item: item[1], reverse=True)
sorted_noun_tokens = [x for x, _ in sorted_noun_counts]
sorted_verb_tokens = [x for x, _ in sorted_verb_counts]
excluded_nouns = ["mummy", "daddy", "john", "carl", "dominic"]
excluded_verbs = [""]
sorted_noun_tokens = [x for x in sorted_noun_tokens if x not in excluded_nouns and noun_tokens[x] > verb_tokens[x]]
sorted_verb_tokens = [x for x in sorted_verb_tokens if x not in excluded_verbs and verb_tokens[x] > noun_tokens[x]]

# b) Split the data into a training corpus and a test corpus
# Find all indices where a sentence ends ("}")
sentence_end_indices = [i for i, tok in enumerate(tokens) if tok == "}"]

# Compute the split point as close as possible to 80% of the data, but at a sentence boundary
target_cutoff = int(len(tokens) * 0.8)
split_idx = max([idx for idx in sentence_end_indices if idx <= target_cutoff])

# Now split at this sentence boundary
train = tokens[:split_idx + 1]
test = tokens[split_idx + 1:]
train_tags = tags[:split_idx + 1]
test_tags = tags[split_idx + 1:]

corpus_total = sum(token_counts.values())
token_probs = {k: (v / corpus_total) for k, v in token_counts.items()}
targets = list({k: v for k, v in token_probs.items() if v < 0.0005}.keys())


summary_csv = sweep_and_save_runs(
    run_extract_and_evaluate,
    train, test, test_tags,
    noun_seeds, verb_seeds,
    token_counts, sorted_noun_tokens, sorted_verb_tokens,
    out_dir="sweep_out", pattern_type=1,
)
print("summary written to", summary_csv)

