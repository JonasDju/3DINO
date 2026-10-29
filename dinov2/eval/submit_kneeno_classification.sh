#!/bin/bash
# Standalone KneeNo classification evaluation of a 3DINO checkpoint (dinov2/eval/kneeno_classification.py). Submit
# from the repo root:
#     sbatch dinov2/eval/submit_kneeno_classification.sh
# CHECKPOINT defaults to the official ViT-L weights, which the default config describes.
# Pass env vars (CHECKPOINT, CONFIG or CONFIG_BASE_DIR + CONFIG_FILE_NAME, REPO_DIR, KNEENO_DIR, SPLIT_LEVEL,
# SPLIT_REP) with --export: RWTH's sbatch submits with --export=NONE, so a `VAR=... sbatch ...` prefix never reaches
# the job, e.g. `sbatch --export=ALL,CHECKPOINT=... dinov2/eval/submit_kneeno_classification.sh`. One job per config
# variant of a directory:
#     for name in plain query2label; do
#         sbatch --export=ALL,CHECKPOINT=...,CONFIG_BASE_DIR=/path/to/configs,CONFIG_FILE_NAME=$name \
#             dinov2/eval/submit_kneeno_classification.sh
#     done
# SPLIT_LEVEL / SPLIT_REP (default 1.0 / 0) pick the split manifest's train/val split, one job per (level, rep):
#     for level in 1.0 0.1 0.01; do for rep in 0 1 2; do
#         sbatch --export=ALL,CHECKPOINT=...,SPLIT_LEVEL=$level,SPLIT_REP=$rep dinov2/eval/submit_kneeno_classification.sh
#     done; done
#
# The job logs to slurm_logs/type-<CONFIG_FILE_NAME>.level-<SPLIT_LEVEL>.rep-<SPLIT_REP>.<job id>.log in the submit
# directory (slurm_logs/ is in the repo: SLURM does not create it). Until the script has resolved the config and the
# split it is slurm_logs/<job id>.log, where an early error (e.g. CONFIG and CONFIG_FILE_NAME both set) stays.
# The evaluation runs in a single process on one GPU (KneeNo evaluates on rank 0 only, more GPUs would not help).
# --cpus-per-task feeds the eval loader's eval.data.num_workers (31) workers.
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=3dino-eval
#SBATCH --output=slurm_logs/%j.log

set -euo pipefail

# --- Locations (override through --export) ---------------------------------------------------------------
# Output directories and everything else about the evaluation are in the eval config.
# SLURM runs a copy of this script, so the repo cannot be located via $0.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KNEENO_DIR="${KNEENO_DIR:-$REPO_DIR/../KneeNo}"
# The eval config: either CONFIG (a path to a .yaml file) or CONFIG_BASE_DIR (a directory of configs) +
# CONFIG_FILE_NAME (a config in it without the .yaml, e.g. query2label for query2label.yaml), not both; relative
# paths are relative to REPO_DIR. CONFIG_FILE_NAME is then exported as the config's file name without the .yaml
# either way, which the eval config uses to name its output folders.
if [[ -n "${CONFIG:-}" ]]; then
    if [[ -n "${CONFIG_BASE_DIR:-}" || -n "${CONFIG_FILE_NAME:-}" ]]; then
        echo "Set either CONFIG or CONFIG_BASE_DIR + CONFIG_FILE_NAME, not both" >&2
        exit 1
    fi
elif [[ -n "${CONFIG_BASE_DIR:-}" || -n "${CONFIG_FILE_NAME:-}" ]]; then
    if [[ -z "${CONFIG_BASE_DIR:-}" || -z "${CONFIG_FILE_NAME:-}" ]]; then
        echo "CONFIG_BASE_DIR and CONFIG_FILE_NAME must be set together" >&2
        exit 1
    fi
    if [[ "$CONFIG_FILE_NAME" == */* || "$CONFIG_FILE_NAME" == *.yaml ]]; then
        echo "CONFIG_FILE_NAME must be a config's name in CONFIG_BASE_DIR without the .yaml (got $CONFIG_FILE_NAME)" >&2
        exit 1
    fi
    CONFIG="$CONFIG_BASE_DIR/$CONFIG_FILE_NAME.yaml"
else
    CONFIG="dinov2/configs/eval/kneeno-vitl-official.yaml"
fi
if [[ "$CONFIG" != *.yaml ]]; then
    echo "The eval config must be a .yaml file (got $CONFIG)" >&2
    exit 1
fi
export CONFIG_FILE_NAME="$(basename "$CONFIG" .yaml)"
# The split of KneeNo's split manifest the heads train and validate on (the eval config's eval.split.level / rep).
# Defaults: the whole internal "train" split, rep 0. Each (level, rep) writes to its own output folder.
export SPLIT_LEVEL="${SPLIT_LEVEL:-1.0}"
export SPLIT_REP="${SPLIT_REP:-0}"
# SLURM opens the log before the script runs, so --output cannot name it after the config and split: rename it now.
# SLURM keeps writing to the renamed file. Left alone if the log is elsewhere (sbatch --output=...), and never moved
# over an earlier log (a requeued job keeps its id).
LOG_DIR="${SLURM_SUBMIT_DIR:-}/slurm_logs"
if [[ -n "${SLURM_JOB_ID:-}" && -f "$LOG_DIR/$SLURM_JOB_ID.log" ]]; then
    mv -n "$LOG_DIR/$SLURM_JOB_ID.log" \
        "$LOG_DIR/type-$CONFIG_FILE_NAME.level-$SPLIT_LEVEL.rep-$SPLIT_REP.$SLURM_JOB_ID.log"
fi
# The config's model: block must describe the checkpoint, or the strict load fails -- after the data extraction.
CHECKPOINT="${CHECKPOINT:-$HPCWORK/public_checkpoints/3dino/3dino_vit_weights.pth}"
# The job's node-local scratch: the data is extracted there and the eval config's data_root points there.
: "${TMP:?TMP (the node-local scratch directory of the job) is not set}"

# Fail here rather than after the data extraction
[[ "$CONFIG" == /* ]] || CONFIG="$REPO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Eval config not found at $CONFIG (set CONFIG, or CONFIG_BASE_DIR + CONFIG_FILE_NAME)" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found at $CHECKPOINT (set CHECKPOINT)" >&2; exit 1; }
[[ -f "$KNEENO_DIR/data/prepare_data.py" ]] || { echo "KneeNo not found at $KNEENO_DIR (set KNEENO_DIR)" >&2; exit 1; }
if [[ "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "The evaluation runs in a single process; submit with one task (got ${SLURM_NTASKS})" >&2
    exit 1
fi

cd "$REPO_DIR"
PYTHON="$REPO_DIR/.venv/bin/python"

# --- Data -> node-local scratch ($TMP/kneeno_data/{internal,external}, prepare_data.py's default): the labeled
# eval datasets (internal: train part + "internal" test set; external: "external" test set); each skipped if
# already populated
srun "$PYTHON" "$KNEENO_DIR/data/prepare_data.py" --internal-tar-dir /hpcwork/p0021834/workspace_roman/jonas/BigKneeTar \
     --external-tar /hpcwork/p0021834/workspace_roman/jonas/uka_gold_nifti.tar --pool-size 16

# --- Evaluation -------------------------------------------------------------------------------------------
srun "$PYTHON" -m dinov2.eval.kneeno_classification --config "$CONFIG" --checkpoint "$CHECKPOINT" --device cuda:0

# Cleanup
rm -rf "$TMP/kneeno_data"
echo "Deleted KneeNo data"
