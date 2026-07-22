# Category Bootstrap

Code for bootstrapping words into grammatical categories (NOUN/VERB) from a tagged child-language corpus, based on Brusini et al. (2021). It extracts context patterns around seed nouns/verbs, uses those patterns to categorize other words in the corpus, and scores the result against the corpus's own tags.

## Data

- `manchester_input_tagged_trf_word_and_lemma.txt` — raw tagged Manchester corpus, one utterance per line, tokens in `WORD_LEMMA_TAG` format.
- `manchester_input_tagged_trf_word_and_lemma_postprocessed.txt` — cleaned-up version produced by `from_tagged_corpus_to_seeds.py` (tag/lemma fixups, filler-word normalization, punctuation stripped). This is the file `category_bootstrap.py` actually reads. `be`, `do`, and `have` are retagged away from `VERB` entirely here (blanket exclusion, covering both auxiliary and lexical/main-verb uses), so they never count as verbs anywhere downstream, including `all_tagged_nouns_verbs` mode (which reads tags directly).
- `noun_selection.xlsx` / `verb_selection.xlsx` — candidate seed words with columns `Word`, `Count`, `Include` (1 = eligible to be used as a seed, 0 = excluded), also produced by `from_tagged_corpus_to_seeds.py`. Noun inclusion is decided via WordNet (`physical entity` hypernym check); verb inclusion is decided by human judgment, read from `verb_inclusion.xlsx`. Only consulted by the `require_tag_match_*` modes - `all_tagged_nouns_verbs` mode ignores these files entirely (see Modes below).
- `verb_inclusion.xlsx` — human-curated verb inclusion judgments, columns `lemma`, `root`, `INCLUDE_wordnet`, `INCLUDE_human`. Every lemma here must occur in the corpus's verb list, or `from_tagged_corpus_to_seeds.py` raises an error. `be`/`do`/`have` are deliberately absent - since they're retagged away from `VERB` before this list is built, they'd otherwise trip that same-error check.

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

- **Modes**: `all_tagged_nouns_verbs` (extract patterns from every word tagged noun/verb in the postprocessed training corpus itself - noun/verb status is decided purely from each occurrence's own corpus tag, and `noun_selection.xlsx`/`verb_selection.xlsx` are ignored entirely, not just for the noun/verb decision but for which words get used at all), `require_tag_match_true` (a word only counts as a noun/verb seed if it's also tagged that way in the corpus), `require_tag_match_false` (seed list alone decides). The `require_tag_match_*` modes sweep across increasing seed-set sizes (smallest allowed by `--cum-prop-threshold`, doubling `--num-sweep-steps` times).
- **Pattern types**: `--pattern-type 1/2/3`, three different ways of defining the context window pattern around a target word.
- **Punctuation**: punctuation tokens (anything with no letters, aside from the `{`/`}` sentence-boundary markers) are normalized to a single `PUNCT` placeholder before patterns are built, so e.g. `,` and `.` aren't treated as different context words. `PUNCT` can appear as a context slot in a learned pattern (e.g. `PUNCT_X_noun`), but a punctuation token is never used as the target/filler word (the `X` itself) - only real words (and the `NOUN`/`VERB` abstractions) can fill that position.
- **Context abstraction**: `--no-abstract-context` (default is abstraction on) controls whether CONTEXT words (not the target word - that's always abstracted) get collapsed to `"noun"`/`"verb"` when they qualify, or are left as their literal surface form. Must be set consistently between the run that built a pattern set and any later categorization pass reusing it, since a pattern like `the_X_noun` won't match anything if context words weren't abstracted when it was built.
- **Baseline**: every run also reports the score of a random-guess classifier, computed analytically (no simulation needed), for comparison. The guess probabilities come from self-classifying the training occurrences that built the run's own patterns (see `compute_pattern_guess_probs`) — e.g. if seed words occur 100 times in training and the model's patterns classify 10 of those as NOUN, 10 as VERB, and 80 as OTHER, the baseline guesses with probabilities 0.1/0.1/0.8, reflecting how decisive this particular pattern set is rather than the corpus's raw tag proportions. Those guesses are then scored against the actual test set.
- **Corpus size**: `--corpus-size` randomly subsamples the corpus down to that many sentences instead of using the full corpus (deterministic given `--split-seed`). By default (`--subsample-scope train_only`) only the training pool is subsampled — the held-out test set is always the same fixed sentences regardless of corpus size, so results across different sizes are comparable against one fixed test set. `--subsample-scope whole_corpus` instead subsamples the full corpus before splitting, so the test set shrinks and changes between sizes too.

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
| `--corpus-size` | (none = full corpus) | randomly subsample the corpus down to this many sentences |
| `--subsample-scope` | `train_only` | `train_only` (fixed test set across corpus sizes) or `whole_corpus` (test set also shrinks) — only matters when `--corpus-size` is given |
| `--no-abstract-context` | off (abstraction on) | disable noun/verb abstraction of CONTEXT words - see Context abstraction above |
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
- `<out-dir>/confusion_words_*.csv` — same true-tag rows as `confusion_matrices.txt`, but with the `NOUN`/`VERB` prediction columns kept and the `OTHER` column broken out into three summary columns instead of one column per individual word: `item-noun`, `item-verb`, `item-neither`. Each counts the total number of TOKEN occurrences (not distinct word types) in that row that weren't predicted `NOUN`/`VERB`, bucketed by that word's own primary tag in the training corpus (`item-noun` if the word is one of `sorted_noun_tokens`, `item-verb` if one of `sorted_verb_tokens`, `item-neither` otherwise) - so you can see, per true tag, how many missed tokens were canonically nouns vs. verbs vs. neither, without a sparse column per word.
- `<out-dir>/pattern_usage_*.csv` — one file per run, one row per pattern actually used to classify a test-set word (not the full set of patterns extracted from training). Columns: `pattern`, `uses`, `num_true_noun`/`num_true_verb` (token-occurrence counts of how many times this pattern was used on a true noun/verb), `num_true_noun_types`/`num_true_verb_types` (the same, but counting DISTINCT word types rather than occurrences - e.g. a pattern used on "cat" three times and "dog" once, all true nouns, gets `num_true_noun=4` but `num_true_noun_types=2`), and `predicted` (the pattern's winning label at training time: `NOUN`, `VERB`, or a specific corpus word - whichever type had the single highest training-time count for that pattern; if two or more types tie for the top count, `predicted` instead shows the tied identities joined with `|`, e.g. `NOUN|VERB` or `cat|dog`, rather than collapsing to an uninformative `OTHER`. `OTHER` on its own only appears when the pattern had no recorded occurrences at all).

## Running the full comparison on a cluster

The full mode × pattern-type comparison (3 pattern types × (1 `all_tagged_nouns_verbs` + `--num-sweep-steps` `require_tag_match_true` + `--num-sweep-steps` `require_tag_match_false`) runs — 39 jobs with the defaults) can be dispatched as many independent single-job processes instead of running sequentially in one process.

Both cluster scripts run a preflight step before dispatching any jobs: by default (`REGENERATE_SEEDS=1`, the default) they rerun `from_tagged_corpus_to_seeds.py` to regenerate the postprocessed corpus and `noun_selection.csv`/`verb_selection.csv`, then refresh `noun_selection.xlsx`/`verb_selection.xlsx` from those `.csv` files - so a sweep never silently runs against a stale postprocessed corpus or seed list (e.g. after editing `from_tagged_corpus_to_seeds.py`'s tag cleanup rules or `verb_inclusion.xlsx`'s `Include` judgments). Set `REGENERATE_SEEDS=0` to skip this and dispatch against whatever's already on disk instead (useful if you've already regenerated things yourself, or want to avoid the network-dependent wordnet/`wn` lexicon download on every run). In `run_cluster_slurm.sh` this runs once on the submission host, before any `sbatch` call, since compute nodes may lack the network access the one-time wordnet/`wn` download needs.

```
REGENERATE_SEEDS=0 ./run_cluster.sh sweep_out 6 8
```

### `run_cluster.sh` — local / any-scheduler parallel dispatch

Runs all jobs via `xargs -P`, using as many workers as requested (default: all cores), then merges.

```
./run_cluster.sh [OUT_DIR] [NUM_SWEEP_STEPS] [JOBS] [CORPUS_SIZE]
```

- `OUT_DIR` — default `sweep_out`
- `NUM_SWEEP_STEPS` — default 6
- `JOBS` — default: number of cores (`nproc`)
- `CORPUS_SIZE` — optional; forwarded as `--corpus-size` to every job. Default: unset, i.e. full corpus.

Extra `category_bootstrap.py` flags (including `--subsample-scope`) can be forwarded to every job via the `EXTRA_ARGS` environment variable:

```
EXTRA_ARGS="--window-size 3" ./run_cluster.sh sweep_out 6 8
EXTRA_ARGS="--subsample-scope whole_corpus" ./run_cluster.sh sweep_out 6 8 5000
```

### `run_cluster_slurm.sh` — SLURM job array

Generates the same job list, then submits it as a SLURM array job (one array task per job) on the `serial` partition with a 1-day time limit, followed by a dependent merge job that only runs once the whole array has completed successfully.

```
./run_cluster_slurm.sh [OUT_DIR] [NUM_SWEEP_STEPS] [MAX_CONCURRENT_TASKS] [CORPUS_SIZE]
```

- `OUT_DIR` — default `sweep_out`
- `NUM_SWEEP_STEPS` — default 6
- `MAX_CONCURRENT_TASKS` — optional throttle on simultaneously running array tasks (`--array=1-N%K`); default unthrottled
- `CORPUS_SIZE` — optional; forwarded as `--corpus-size` to every job. Default: unset, i.e. full corpus.

Override the partition/time limit via the `PARTITION` / `TIME_LIMIT` environment variables (defaults: `serial` / `1-00:00:00`). `EXTRA_ARGS` works the same as in `run_cluster.sh` (e.g. `EXTRA_ARGS="--subsample-scope whole_corpus"` to also shrink the test set as corpus size shrinks). The script only submits jobs and returns immediately — track progress with `squeue -u $USER`; results land in `<out-dir>/summary.csv` and `confusion_matrices.txt` once the merge job finishes. Per-task logs go to `<out-dir>/logs/`.

## Notebooks

- `ChildesDataPrep_Eng.ipynb` / `ChildesDataPrep_JP.ipynb` — corpus preparation for the English/Japanese CHILDES data.
- `postTaggingProcessing.ipynb` — exploratory post-tagging processing.
- `GlobalWordnet.ipynb` — WordNet exploration used in seed selection.

## Requirements

Python 3 with `pandas`, `numpy`, `scipy`, `openpyxl`, `nltk`, `wn` (WordNet). Install with:

```
pip install pandas numpy scipy openpyxl nltk wn
```
