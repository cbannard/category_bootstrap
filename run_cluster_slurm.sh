#!/usr/bin/env bash
#
# SLURM version of run_cluster.sh: submits the full category_bootstrap
# mode/pattern-type comparison as a SLURM job array (one array task per
# job), then submits a merge job that only runs once every array task has
# finished successfully. See run_cluster.sh's header for the exact set of
# jobs this generates (39 with the defaults: 3 pattern types x (1 + 6 + 6)).
#
# Every task and the merge job run on:
#   --partition=serial
#   --time=1-00:00:00     (1 day)
# Override either via the PARTITION / TIME_LIMIT environment variables.
#
# Each job writes its own uniquely-named file under
#   $OUT_DIR/summary_parts/*.csv
#   $OUT_DIR/confusion_parts/*.txt
#   $OUT_DIR/confusion_words_*.csv
# so concurrent array tasks never write to the same file. The merge job
# combines them into $OUT_DIR/summary.csv and $OUT_DIR/confusion_matrices.txt.
#
# Usage:
#   ./run_cluster_slurm.sh [OUT_DIR] [NUM_SWEEP_STEPS] [MAX_CONCURRENT_TASKS]
#
#   OUT_DIR                 Where results/logs go. Default: sweep_out
#   NUM_SWEEP_STEPS          Seed-list-size steps per require_tag_match_true/
#                            false sweep. Default: 6
#   MAX_CONCURRENT_TASKS     Optional throttle on simultaneously running
#                            array tasks (SLURM's --array=1-N%K). Default:
#                            unthrottled (let the scheduler decide).
#
# Any extra category_bootstrap.py options (--corpus-file, --noun-seeds-file,
# --verb-seeds-file, --cum-prop-threshold, --window-size, --test-fraction,
# --split-seed) can be set via the EXTRA_ARGS environment variable, e.g.:
#   EXTRA_ARGS="--window-size 3" ./run_cluster_slurm.sh sweep_out 6
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

OUT_DIR="${1:-sweep_out}"
NUM_SWEEP_STEPS="${2:-6}"
MAX_CONCURRENT_TASKS="${3:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

PARTITION="${PARTITION:-serial}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"   # make absolute: array tasks run later/elsewhere
mkdir -p "$OUT_DIR/summary_parts" "$OUT_DIR/confusion_parts" "$OUT_DIR/logs"

JOBS_FILE="$OUT_DIR/jobs.txt"
> "$JOBS_FILE"

for pattern_type in 1 2 3; do
    echo "python3 \"$PYTHON_SCRIPT\" --mode all_tagged_nouns_verbs --pattern-type $pattern_type --out-dir \"$OUT_DIR\" --num-sweep-steps $NUM_SWEEP_STEPS $EXTRA_ARGS" >> "$JOBS_FILE"
    for mode in require_tag_match_true require_tag_match_false; do
        for ((step = 0; step < NUM_SWEEP_STEPS; step++)); do
            echo "python3 \"$PYTHON_SCRIPT\" --mode $mode --pattern-type $pattern_type --seed-step $step --out-dir \"$OUT_DIR\" --num-sweep-steps $NUM_SWEEP_STEPS $EXTRA_ARGS" >> "$JOBS_FILE"
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
