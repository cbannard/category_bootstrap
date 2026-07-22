#!/usr/bin/env bash
#
# Dispatches the full category_bootstrap mode/pattern-type comparison as
# many independent, parallel single-job invocations of category_bootstrap.py
# (one per core, or one per cluster task), then merges the results.
#
# What gets run (39 jobs with the defaults: 3 pattern types x (1 + 6 + 6)):
#   For pattern_type in 1 2 3:
#     - all_tagged_nouns_verbs           (1 job - uses every word tagged
#                                          noun/verb in the postprocessed
#                                          corpus itself, not the curated
#                                          noun_selection.xlsx/verb_selection.xlsx
#                                          seed lists)
#     - require_tag_match_true,  steps 0..NUM_SWEEP_STEPS-1   (NUM_SWEEP_STEPS jobs - seed-list-based)
#     - require_tag_match_false, steps 0..NUM_SWEEP_STEPS-1   (NUM_SWEEP_STEPS jobs - seed-list-based)
#
# Each job writes its own uniquely-named file under
#   $OUT_DIR/summary_parts/*.csv
#   $OUT_DIR/confusion_parts/*.txt
#   $OUT_DIR/confusion_words_*.csv
#   $OUT_DIR/pattern_usage_*.csv
# instead of appending to a shared file, so running many jobs in parallel
# never risks corrupting a file through concurrent writes. Once all jobs
# finish, this script calls `category_bootstrap.py --merge` once to combine
# the parts into the final $OUT_DIR/summary.csv and
# $OUT_DIR/confusion_matrices.txt.
#
# Usage:
#   ./run_cluster.sh [OUT_DIR] [NUM_SWEEP_STEPS] [JOBS] [CORPUS_SIZE]
#
#   OUT_DIR          Where results go. Default: sweep_out
#   NUM_SWEEP_STEPS  Seed-list-size steps per require_tag_match_true/false
#                     sweep. Default: 6
#   JOBS             How many jobs to run at once (e.g. number of cores).
#                     Default: number of cores on this machine (nproc), or
#                     4 if nproc isn't available.
#   CORPUS_SIZE      Optional: randomly subsample the corpus down to this
#                     many sentences instead of using the full corpus
#                     (forwarded as category_bootstrap.py's --corpus-size to
#                     every job). Default: unset, i.e. use the full corpus.
#                     By default this only subsamples the training pool,
#                     keeping the same held-out test set across every corpus
#                     size - set EXTRA_ARGS="--subsample-scope whole_corpus"
#                     to subsample the test set too instead.
#
# Any extra category_bootstrap.py options (--corpus-file, --noun-seeds-file,
# --verb-seeds-file, --cum-prop-threshold, --window-size, --test-fraction,
# --split-seed, --subsample-scope) can be set via the EXTRA_ARGS environment
# variable, e.g.:
#   EXTRA_ARGS="--window-size 3" ./run_cluster.sh sweep_out 6 8
#
# On a SLURM (or similar) cluster, replace the `xargs -P "$JOBS"` line below
# with your job submission command (e.g. `srun`, `sbatch --wait`, or a job
# array) reading the same one-command-per-line job list - everything else
# (unique per-job output files + the final --merge step) stays the same.
#
# REGENERATE_SEEDS  Set to 0 to skip the from_tagged_corpus_to_seeds.py
#                     preflight step below and dispatch jobs against whatever
#                     manchester_input_tagged_trf_word_and_lemma_postprocessed.txt
#                     / noun_selection.xlsx / verb_selection.xlsx already
#                     exist on disk. Default: 1 (regenerate every time), so a
#                     sweep never silently runs against a stale postprocessed
#                     corpus or seed list (e.g. after editing
#                     from_tagged_corpus_to_seeds.py's tag cleanup rules or
#                     verb_inclusion.xlsx's Include judgments). Requires
#                     network access the first time, to fetch nltk's wordnet
#                     data and the `wn` package's omw-en:1.4 lexicon.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/category_bootstrap.py"
SEEDS_SCRIPT="$SCRIPT_DIR/from_tagged_corpus_to_seeds.py"

OUT_DIR="${1:-sweep_out}"
NUM_SWEEP_STEPS="${2:-6}"
if command -v nproc >/dev/null 2>&1; then
    DEFAULT_JOBS="$(nproc)"
else
    DEFAULT_JOBS=4
fi
JOBS="${3:-$DEFAULT_JOBS}"
CORPUS_SIZE="${4:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REGENERATE_SEEDS="${REGENERATE_SEEDS:-1}"

CORPUS_SIZE_ARGS=""
if [[ -n "$CORPUS_SIZE" ]]; then
    CORPUS_SIZE_ARGS="--corpus-size $CORPUS_SIZE"
fi

if [[ "$REGENERATE_SEEDS" == "1" ]]; then
    echo "Regenerating postprocessed corpus and seed files (from_tagged_corpus_to_seeds.py)..."
    (cd "$SCRIPT_DIR" && python3 "$SEEDS_SCRIPT")
    echo "Refreshing noun_selection.xlsx/verb_selection.xlsx from the regenerated .csv files..."
    python3 -c "
import pandas as pd
pd.read_csv('$SCRIPT_DIR/noun_selection.csv', index_col=0).to_excel('$SCRIPT_DIR/noun_selection.xlsx')
pd.read_csv('$SCRIPT_DIR/verb_selection.csv', index_col=0).to_excel('$SCRIPT_DIR/verb_selection.xlsx')
"
else
    echo "REGENERATE_SEEDS=0: skipping from_tagged_corpus_to_seeds.py, using existing corpus/seed files as-is."
fi

mkdir -p "$OUT_DIR/summary_parts" "$OUT_DIR/confusion_parts"

JOBS_FILE="$(mktemp)"
trap 'rm -f "$JOBS_FILE"' EXIT

for pattern_type in 1 2 3; do
    echo "python3 \"$PYTHON_SCRIPT\" --mode all_tagged_nouns_verbs --pattern-type $pattern_type --out-dir \"$OUT_DIR\" --num-sweep-steps $NUM_SWEEP_STEPS $CORPUS_SIZE_ARGS $EXTRA_ARGS" >> "$JOBS_FILE"
    for mode in require_tag_match_true require_tag_match_false; do
        for ((step = 0; step < NUM_SWEEP_STEPS; step++)); do
            echo "python3 \"$PYTHON_SCRIPT\" --mode $mode --pattern-type $pattern_type --seed-step $step --out-dir \"$OUT_DIR\" --num-sweep-steps $NUM_SWEEP_STEPS $CORPUS_SIZE_ARGS $EXTRA_ARGS" >> "$JOBS_FILE"
        done
    done
done

NUM_JOBS="$(wc -l < "$JOBS_FILE" | tr -d ' ')"
echo "Dispatching $NUM_JOBS jobs across $JOBS parallel worker(s)..."
xargs -P "$JOBS" -I{} bash -c '{}' < "$JOBS_FILE"

echo "All jobs finished. Merging results..."
python3 "$PYTHON_SCRIPT" --merge --out-dir "$OUT_DIR"

echo "Done. See $OUT_DIR/summary.csv and $OUT_DIR/confusion_matrices.txt"
