#!/usr/bin/env bash
#
# SLURM version of run_cluster.sh: submits the full category_bootstrap
# mode/pattern-type comparison as a SLURM job array (one array task per
# job), then submits a merge job that only runs once every array task has
# finished successfully. See run_cluster.sh's header for the exact set of
# jobs this generates (3 pattern types x (1 + NUM_STEPS + NUM_STEPS), where
# NUM_STEPS is the MATCHED SEQUENCE of (num_nouns, num_verbs) seed-set
# pairings - NOT a cross product/grid - queried fresh each run via
# --print-num-seed-steps - see run_cluster.sh's header for details), and for
# how --n-folds (default 5, k-fold cross-validation, run internally by each
# job) affects runtime and per-job output file counts.
#
# Every task and the merge job run on:
#   --partition=serial
#   --time=1-00:00:00     (1 day)
# Override either via the PARTITION / TIME_LIMIT environment variables. Set
# MEM_PER_TASK (e.g. "16G") to add an explicit --mem= request - required on
# clusters where only certain partitions (e.g. himem) accept a memory
# specification at all; leave it unset on partitions where --mem is rejected
# or unnecessary. Applies to both the array tasks and the merge job.
#
# Each job writes its own uniquely-named file under
#   $OUT_DIR/summary_parts/*.csv
#   $OUT_DIR/confusion_parts/*.txt
#   $OUT_DIR/confusion_words_*.csv
#   $OUT_DIR/pattern_usage_*.csv
# so concurrent array tasks never write to the same file. The merge job
# combines them into $OUT_DIR/summary.csv and $OUT_DIR/confusion_matrices.txt.
#
# Usage:
#   ./run_cluster_slurm.sh [OUT_DIR] [MAX_CUM_PROP_THRESHOLD] [MAX_CONCURRENT_TASKS] [CORPUS_SIZE]
#
#   OUT_DIR                 Where results/logs go. Default: sweep_out
#   MAX_CUM_PROP_THRESHOLD  Cap on the noun seed-list size tested (every
#                            noun count 1..N is tested, matched against verb
#                            counts - not a doubling sequence, not a cross
#                            product) - see compute_seed_steps. Default:
#                            0.239 (all 33 curated verbs are always matched -
#                            they cover 23.9% of verb tokens at most - while
#                            nouns are capped to a comparable ~36-word list
#                            with the current seed files). Check the
#                            resulting job count first with
#                            --print-num-seed-steps before raising this.
#   MAX_CONCURRENT_TASKS     Optional throttle on simultaneously running
#                            array tasks (SLURM's --array=1-N%K). Default:
#                            unthrottled (let the scheduler decide).
#   CORPUS_SIZE              Optional: randomly subsample the corpus down to
#                            this many sentences instead of using the full
#                            corpus (forwarded as category_bootstrap.py's
#                            --corpus-size to every job). Default: unset,
#                            i.e. use the full corpus. By default this only
#                            subsamples the training pool, keeping the same
#                            held-out test set across every corpus size -
#                            set EXTRA_ARGS="--subsample-scope whole_corpus"
#                            to subsample the test set too instead.
#
# Any extra category_bootstrap.py options (--corpus-file, --noun-seeds-file,
# --verb-seeds-file, --num-sweep-steps, --window-size, --test-fraction,
# --split-seed, --subsample-scope) can be set via the EXTRA_ARGS environment
# variable, e.g.:
#   EXTRA_ARGS="--window-size 3" ./run_cluster_slurm.sh sweep_out 0.239
#
# REGENERATE_SEEDS  Set to 0 to skip the from_tagged_corpus_to_seeds.py
#                     preflight step below and submit jobs against whatever
#                     manchester_input_tagged_trf_word_and_lemma_postprocessed.txt
#                     / noun_selection.xlsx / verb_selection.xlsx already
#                     exist on disk. Default: 1 (regenerate every time), so a
#                     sweep never silently runs against a stale postprocessed
#                     corpus or seed list. This runs once, here on the
#                     submission host, before any `sbatch` call - not per
#                     array task - since it needs network access (nltk's
#                     wordnet data, the `wn` package's omw-en:1.4 lexicon)
#                     that compute nodes may not have.
#
# This script only calls `sbatch` twice (array job + dependent merge job)
# and returns immediately - it does not wait for the run to finish. Track
# progress with `squeue -u $USER`. Once the merge job completes, results
# are in $OUT_DIR/summary.csv and $OUT_DIR/confusion_matrices.txt.
#
# NOTE: this script was written and syntax-checked without access to a real
# SLURM scheduler (no sbatch/srun available in the dev sandbox), so the
# job-list generation and array-index-to-job mapping were verified by
# simulating `sed -n "${i}p"` locally, but the actual `sbatch` submission
# and --dependency=afterok behavior could not be run end-to-end. Worth a
# dry run on your cluster before trusting it for a long sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/category_bootstrap.py"
SEEDS_SCRIPT="$SCRIPT_DIR/from_tagged_corpus_to_seeds.py"

OUT_DIR="${1:-sweep_out}"
MAX_CUM_PROP_THRESHOLD="${2:-0.239}"
MAX_CONCURRENT_TASKS="${3:-}"
CORPUS_SIZE="${4:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REGENERATE_SEEDS="${REGENERATE_SEEDS:-1}"

PARTITION="${PARTITION:-serial}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
MEM_PER_TASK="${MEM_PER_TASK:-}"

MEM_SBATCH_LINE=""
if [[ -n "$MEM_PER_TASK" ]]; then
    MEM_SBATCH_LINE="#SBATCH --mem=$MEM_PER_TASK"
fi

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

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"   # make absolute: array tasks run later/elsewhere
mkdir -p "$OUT_DIR/summary_parts" "$OUT_DIR/confusion_parts" "$OUT_DIR/logs"

JOBS_FILE="$OUT_DIR/jobs.txt"
> "$JOBS_FILE"

# The seed-set sweep is a MATCHED SEQUENCE of (num_nouns, num_verbs) pairs
# (see compute_seed_steps/--max-cum-prop-threshold in category_bootstrap.py),
# NOT a cross product, so the number of valid --seed-step values has to be
# queried rather than assumed. Independent of pattern_type/mode
# (require_tag_match_true and _false sweep the identical sequence), so this
# only needs to run once.
NUM_STEPS="$(python3 "$PYTHON_SCRIPT" --print-num-seed-steps --max-cum-prop-threshold "$MAX_CUM_PROP_THRESHOLD" $EXTRA_ARGS)"
echo "Seed-set sweep size: $NUM_STEPS pairings (--max-cum-prop-threshold $MAX_CUM_PROP_THRESHOLD, plus any --noun-seeds-file/--verb-seeds-file/--num-sweep-steps overrides in EXTRA_ARGS)."

for pattern_type in 1 2 3; do
    echo "python3 \"$PYTHON_SCRIPT\" --mode all_tagged_nouns_verbs --pattern-type $pattern_type --out-dir \"$OUT_DIR\" $CORPUS_SIZE_ARGS $EXTRA_ARGS" >> "$JOBS_FILE"
    for mode in require_tag_match_true require_tag_match_false; do
        for ((step = 0; step < NUM_STEPS; step++)); do
            echo "python3 \"$PYTHON_SCRIPT\" --mode $mode --pattern-type $pattern_type --seed-step $step --out-dir \"$OUT_DIR\" --max-cum-prop-threshold $MAX_CUM_PROP_THRESHOLD $CORPUS_SIZE_ARGS $EXTRA_ARGS" >> "$JOBS_FILE"
        done
    done
done

NUM_JOBS="$(wc -l < "$JOBS_FILE" | tr -d ' ')"
echo "Wrote $NUM_JOBS job(s) to $JOBS_FILE"

# Per-array-task runner: SLURM_ARRAY_TASK_ID selects a line (1-indexed) from
# jobs.txt and runs it. Kept as its own file (rather than inline -wrap) so
# each task's stdout/stderr can be captured per-task via #SBATCH --output.
TASK_SCRIPT="$OUT_DIR/_slurm_task.sh"
cat > "$TASK_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME_LIMIT
$MEM_SBATCH_LINE
#SBATCH --job-name=category_bootstrap
#SBATCH --output=$OUT_DIR/logs/task_%A_%a.out
#SBATCH --error=$OUT_DIR/logs/task_%A_%a.err

set -euo pipefail
JOBS_FILE="$JOBS_FILE"
CMD="\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "\$JOBS_FILE")"
echo "Task \$SLURM_ARRAY_TASK_ID: \$CMD"
eval "\$CMD"
EOF
chmod +x "$TASK_SCRIPT"

# Merge job: runs once, after the whole array succeeds (afterok on an array
# job ID waits for every task in the array, not just the first).
MERGE_SCRIPT="$OUT_DIR/_slurm_merge.sh"
cat > "$MERGE_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME_LIMIT
$MEM_SBATCH_LINE
#SBATCH --job-name=category_bootstrap_merge
#SBATCH --output=$OUT_DIR/logs/merge_%j.out
#SBATCH --error=$OUT_DIR/logs/merge_%j.err

set -euo pipefail
python3 "$PYTHON_SCRIPT" --merge --out-dir "$OUT_DIR"
EOF
chmod +x "$MERGE_SCRIPT"

ARRAY_SPEC="1-$NUM_JOBS"
if [[ -n "$MAX_CONCURRENT_TASKS" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT_TASKS}"
fi

echo "Submitting array job ($NUM_JOBS tasks, partition=$PARTITION, time=$TIME_LIMIT)..."
ARRAY_JOB_ID="$(sbatch --array="$ARRAY_SPEC" --parsable "$TASK_SCRIPT")"
echo "Array job ID: $ARRAY_JOB_ID"

echo "Submitting merge job (runs after the array job succeeds)..."
MERGE_JOB_ID="$(sbatch --dependency=afterok:"$ARRAY_JOB_ID" --parsable "$MERGE_SCRIPT")"
echo "Merge job ID: $MERGE_JOB_ID"

echo "Done submitting. Track progress with: squeue -u \$USER"
echo "Once job $MERGE_JOB_ID finishes, see $OUT_DIR/summary.csv and $OUT_DIR/confusion_matrices.txt"
