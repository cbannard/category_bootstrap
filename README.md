# Category Bootstrap

Code for bootstrapping words into grammatical categories (NOUN/VERB) from a tagged child-language corpus, based on Brusini et al. (2021). It extracts context patterns around seed nouns/verbs, uses those patterns to categorize other words in the corpus, and scores the result against the corpus's own tags.

## Data

- `manchester_input_tagged_trf_word_and_lemma.txt` — raw tagged Manchester corpus, one utterance per line, tokens in `WORD_LEMMA_TAG` format.
- `manchester_input_tagged_trf_word_and_lemma_postprocessed.txt` — cleaned-up version produced by `from_tagged_corpus_to_seeds.py` (tag/lemma fixups, filler-word normalization, punctuation stripped). This is the file `category_bootstrap.py` actually reads.
- `noun_selection.xlsx` / `verb_selection.xlsx` — candidate seed words with columns `Word`, `Count`, `Include` (1 = eligible to be used as a seed, 0 = excluded), also produced by `from_tagged_corpus_to_seeds.py`. Noun inclusion is decided via WordNet (`physical entity` hypernym check); verb inclusion is decided by human judgment, read from `verb_inclusion.xlsx`.
- `verb_inclusion.xlsx` — human-curated verb inclusion judgments, columns `lemma`, `root`, `INCLUDE_wordnet`, `INCLUDE_human`. Every lemma here must occur in the corpus's verb list, or `from_tagged_corpus_to_seeds.py` raises an error.

## Pipeline

### 1. `from_tagged_corpus_to_seeds.py`

Reads the raw corpus, applies tag/token cleanup rules, writes the postprocessed corpus file, then builds the noun and verb seed selection files (`noun_selection.xlsx`/`.csv`, `verb_selection.xlsx`/`.csv`). Run this first, and whenever the raw corpus or `verb_inclusion.xlsx` changes.

```
python3 from_tagged_corpus_to_seeds.py
```

### 2. `category_bootstrap.py`

The main pipeline: loads the postprocessed corpus, splits it into train/test (random 20% of sentences by default, seeded for reproducibility), extracts context patterns around nouns/verbs in the training set, categorizes words in the test set from those patterns, and scores precision/recall against the corpus's own tags plus a frequency-matched random baseline.

It can be run two ways:

- **In-process** (no `--mode`): runs the full mode × pattern-type comparison sequentially in a single process, writing straight to `<out-dir>/summary.csv` and `confusion_matrices.txt`.
- **Single job** (`--mode ...`): runs exactly one `(pattern_type, mode, seed-step)` configuration and writes it to its own uniquely-named file under `<out-dir>/summary_parts/` and `<out-dir>/confusion_parts/`. This is the unit of work the cluster scripts dispatch in parallel — many single-job processes never write to the same file, so there's no risk of concurrent-write corruption. Run with `--merge` afterwards to combine the parts into the final `summary.csv`/`confusion_matrices.txt`.

Key concepts:

- **Modes**: `all_tagged_nouns_verbs` (extract patterns from every corpus-tagged noun/verb, not just seeds), `require_tag_match_true` (a word only counts as a noun/verb seed if it's also tagged that way in the corpus), `require_tag_match_false` (seed list alone decides). The `require_tag_match_*` modes sweep across increasing seed-set sizes (smallest allowed by `--cum-prop-threshold`, doubling `--num-sweep-steps` times).
- **Pattern types**: `--pattern-type 1/2/3`, three different ways of defining the context window pattern around a target word.
- **Baseline**: every run also reports the score of a frequency-matched random-guess classifier, computed analytically (no simulation needed), for comparison.

Useful flags (see `python3 category_bootstrap.py --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | (none = full comparison) | `all_tagged_nouns_verbs` / `require_tag_match_true` / `require_tag_match_false` |
| `--pattern-type` | 1 | 1, 2, or 3 |
| `--seed-step` | (none) | 0-indexed seed-set size step; required for `require_tag_match_*` single-job mode |
| `--num-sweep-steps` | 6 | number of doubling steps in the seed-set sweep |
| `--cum-prop-threshold` | 0.1 | cumulative-frequency threshold used to pick the smallest seed-set size |
| `--window-size` | 2 | context window size (tokens either side of the target) |
| `--out-dir` | `sweep_out` | output directory |
| `--corpus-file` | `manchester_input_tagged_trf_word_and_lemma_postprocessed.txt` | corpus to read |
| `--noun-seeds-file` / `--verb-seeds-file` | `noun_selection.xlsx` / `verb_selection.xlsx` | seed files |
| `--test-fraction` | 0.2 | fraction of sentences held out for testing |
| `--split-seed` | 42 | RNG seed for the train/test split (keeps the split identical across independent processes) |
| `--merge` | off | merge `summary_parts`/`confusion_parts` into the final output files, then exit |

Example — run everything in one process:

```
python3 category_bootstrap.py --out-dir sweep_out
```

Example — run a single job (as the cluster scripts do), then merge:

```
python3 category_bootstrap.py --mode require_tag_match_true --pattern-type 2 --seed-step 3 --out-dir sweep_out
python3 category_bootstrap.py --merge --out-dir sweep_out
```

### Outputs

- `<out-dir>/summary.csv` — one row per run, columns defined in `SUMMARY_COLS`: `time`, `mode`, `pattern_type`, `num_noun_seeds`, `num_verb_seeds`, `runtime_s`, per-class and macro/micro precision/recall, plus the same set prefixed `baseline_` for the random-guess comparison.
- `<out-dir>/confusion_matrices.txt` — one confusion matrix per run (true axis = actual corpus tags, predicted axis = collapsed NOUN/VERB/OTHER), labeled with mode and pattern type.
- `<out-dir>/confusion_words_*.csv` — word-level breakdown of everything predicted OTHER, one file per run.

## Running the full comparison on a cluster

The full mode × pattern-type comparison (3 pattern types × (1 `all_tagged_nouns_verbs` + `--num-sweep-steps` `require_tag_match_true` + `--num-sweep-steps` `require_tag_match_false`) runs — 39 jobs with the defaults) can be dispatched as many independent single-job processes instead of running sequentially in one process.

### `run_cluster.sh` — local / any-scheduler parallel dispatch

Runs all jobs via `xargs -P`, using as many workers as requested (default: all cores), then merges.

```
./run_cluster.sh [OUT_DIR] [NUM_SWEEP_STEPS] [JOBS]
```

- `OUT_DIR` — default `sweep_out`
- `NUM_SWEEP_STEPS` — default 6
- `JOBS` — default: number of cores (`nproc`)

Extra `category_bootstrap.py` flags can be forwarded to every job via the `EXTRA_ARGS` environment variable:

```
EXTRA_ARGS="--window-size 3" ./run_cluster.sh sweep_out 6 8
```

### `run_cluster_slurm.sh` — SLURM job array

Generates the same job list, then submits it as a SLURM array job (one array task per job) on the `serial` partition with a 1-day time limit, followed by a dependent merge job that only runs once the whole array has completed successfully.

```
./run_cluster_slurm.sh [OUT_DIR] [NUM_SWEEP_STEPS] [MAX_CONCURRENT_TASKS]
```

- `OUT_DIR` — default `sweep_out`
- `NUM_SWEEP_STEPS` — default 6
- `MAX_CONCURRENT_TASKS` — optional throttle on simultaneously running array tasks (`--array=1-N%K`); default unthrottled

Override the partition/time limit via the `PARTITION` / `TIME_LIMIT` environment variables (defaults: `serial` / `1-00:00:00`). `EXTRA_ARGS` works the same as in `run_cluster.sh`. The script only submits jobs and returns immediately — track progress with `squeue -u $USER`; results land in `<out-dir>/summary.csv` and `confusion_matrices.txt` once the merge job finishes. Per-task logs go to `<out-dir>/logs/`.

## Notebooks

- `ChildesDataPrep_Eng.ipynb` / `ChildesDataPrep_JP.ipynb` — corpus preparation for the English/Japanese CHILDES data.
- `postTaggingProcessing.ipynb` — exploratory post-tagging processing.
- `GlobalWordnet.ipynb` — WordNet exploration used in seed selection.

## Requirements

Python 3 with `pandas`, `numpy`, `scipy`, `openpyxl`, `nltk`, `wn` (WordNet). Install with:

```
pip install pandas numpy scipy openpyxl nltk wn
```
