#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run_train_latent_channels.sh <128|512|1024> [GPUs-per-node] [options]

Train the hourly ERA/observation Encoder pipeline with torchrun.
Stages: ae -> dump -> stats -> diffusion -> satellite -> convi

  --skip-to STAGE       First stage (default: ae)
  --stop-after STAGE    Last stage (default: ae for multi-node AE starts; else convi)
  --output DIR         Output root (default: outputs/latent_channels/c<channels>)
  --ae-id ID           ERA AE run id (default: era_c<channels>)
  --instrument NAME    satellite stage: all|atms|amsua|mhs|mwts|mwhs
  --nodes N            Node count (NNODES, or platform WORLD_SIZE; default: 1)
  --node-rank R        Node rank (NODE_RANK, or platform RANK; default: 0)
  --master-addr HOST   Address of node 0 (MASTER_ADDR; required for multiple nodes)
  --master-port PORT   Shared rendezvous port (MASTER_PORT; default: 29500)
  --dry-run            Print exact commands without training or writing outputs
  --help               Show this help

Activate the appa environment first. GPUs/node defaults to PET_NPROC_PER_NODE or 8.
Run the same command on every node with different node ranks. Data, code,
output path, model id, and training parameters must agree across nodes.
For multiple nodes, starting at ae defaults to stopping after ae. Then run
dump/stats once on one node, followed by --skip-to diffusion --stop-after diffusion
on all nodes. Dump/stats cannot run on multiple nodes.
AE_MICROBATCH=2, DIFF_MICROBATCH=4, DUMP_BATCH_SIZE=2 may be lowered.
Effective batch / optimizer updates can be set independently by stage:
  AE_GLOBAL_BATCH=64     AE_STEPS=100000
  DIFF_GLOBAL_BATCH=256  DIFF_STEPS=125000
  OBS_GLOBAL_BATCH=8     OBS_STEPS=100000
WANDB_MODE defaults to disabled; set it to online explicitly if desired.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
CHANNELS=$1
shift
case "$CHANNELS" in 128|512|1024) ;; *) usage >&2; exit 2 ;; esac
GPUS=${PET_NPROC_PER_NODE:-8}
if [[ ${1:-} =~ ^[0-9]+$ ]]; then GPUS=$1; shift; fi
case "$GPUS" in 1|2|4|8) ;; *) echo 'GPUs must be 1, 2, 4, or 8' >&2; exit 2 ;; esac

TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$TRAIN_ROOT/outputs/latent_channels/c$CHANNELS"
AE_ID="era_c$CHANNELS"
START_STAGE=ae
STOP_STAGE=
INSTRUMENT=all
DRY_RUN=0
NODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_ID=${NODE_RANK:-${RANK:-0}}
RENDEZVOUS_HOST=${MASTER_ADDR:-}
RENDEZVOUS_PORT=${MASTER_PORT:-29500}
while (( $# )); do
    case "$1" in
        --skip-to) START_STAGE=${2:?Missing stage}; shift 2 ;;
        --stop-after) STOP_STAGE=${2:?Missing stage}; shift 2 ;;
        --output) OUTPUT_DIR=${2:?Missing output}; shift 2 ;;
        --ae-id) AE_ID=${2:?Missing AE id}; shift 2 ;;
        --instrument) INSTRUMENT=${2:?Missing instrument}; shift 2 ;;
        --nodes) NODES=${2:?Missing node count}; shift 2 ;;
        --node-rank) NODE_ID=${2:?Missing node rank}; shift 2 ;;
        --master-addr) RENDEZVOUS_HOST=${2:?Missing master address}; shift 2 ;;
        --master-port) RENDEZVOUS_PORT=${2:?Missing master port}; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$PWD/$OUTPUT_DIR"
if [[ ! "$AE_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'AE id must contain only letters, digits, underscores, and hyphens' >&2
    exit 2
fi
case "$INSTRUMENT" in all|atms|amsua|mhs|mwts|mwhs) ;; *) echo 'Invalid instrument' >&2; exit 2 ;; esac
if [[ ! "$NODES" =~ ^[1-9][0-9]*$ || ! "$NODE_ID" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo 'Node count must be positive and node rank must be a nonnegative integer' >&2
    exit 2
fi
if (( NODE_ID >= NODES )); then
    echo "Node rank $NODE_ID must be smaller than node count $NODES" >&2; exit 2
fi
if [[ ! "$RENDEZVOUS_PORT" =~ ^[1-9][0-9]*$ ]] || (( RENDEZVOUS_PORT > 65535 )); then
    echo 'Master port must be an integer in 1..65535' >&2; exit 2
fi
if (( NODES > 1 )) && [[ -z "$RENDEZVOUS_HOST" ]]; then
    echo 'Set MASTER_ADDR or --master-addr to the address of node 0 for multiple nodes' >&2
    exit 2
fi
if (( NODES > 1 )); then
    case "$RENDEZVOUS_HOST" in
        localhost|127.*|::1|'[::1]')
            echo 'For multiple nodes, MASTER_ADDR/--master-addr must be a reachable node-0 address, not loopback' >&2
            exit 2 ;;
    esac
fi
TOTAL_PROCESSES=$((NODES * GPUS))

if [[ -z "$STOP_STAGE" ]]; then
    STOP_STAGE=convi
    if (( NODES > 1 )) && [[ "$START_STAGE" == ae ]]; then
        STOP_STAGE=ae
        echo '[launcher] Multiple nodes and no --stop-after: defaulting to AE only.'
        echo '[launcher] After AE finishes on all nodes, run dump/stats once on one node.'
    fi
fi

STAGES=(ae dump stats diffusion satellite convi)
START_INDEX=-1
STOP_INDEX=-1
for i in "${!STAGES[@]}"; do
    [[ ${STAGES[$i]} != "$START_STAGE" ]] || START_INDEX=$i
    [[ ${STAGES[$i]} != "$STOP_STAGE" ]] || STOP_INDEX=$i
done
if (( START_INDEX < 0 || STOP_INDEX < START_INDEX )); then
    echo 'Invalid stage range; use ae, dump, stats, diffusion, satellite, convi' >&2
    exit 2
fi
if (( NODES > 1 && START_INDEX <= 2 && STOP_INDEX >= 1 )); then
    echo 'dump/stats are single-node stages. Train with --stop-after ae on all nodes,' >&2
    echo 'run --nodes 1 --node-rank 0 --skip-to dump --stop-after stats once,' >&2
    echo 'then launch --skip-to diffusion --stop-after diffusion on all nodes.' >&2
    exit 2
fi

export RISDA_ROOT="$TRAIN_ROOT"
export PYTHONPATH="$TRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RISDA_OUTPUT_ROOT="$OUTPUT_DIR"
export RISDA_ERA5_1H="${ERA_TRAIN_DATA:-${RISDA_ERA5_1H:-$TRAIN_ROOT/data/era5_hourly}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

: "${RISDA_MEAN_STD:?Set RISDA_MEAN_STD (see configs/paths.env.example)}"
SOURCE_MEAN="$RISDA_ERA5_1H/mean_std/era5_1h_mean.npy"
SOURCE_STD="$RISDA_ERA5_1H/mean_std/era5_1h_std.npy"
TARGET_MEAN="$RISDA_MEAN_STD/mean_era5.npy"
TARGET_STD="$RISDA_MEAN_STD/std_era5.npy"
OBS_ERA_DATA="${OBS_ERA_DATA:-$RISDA_ERA5_1H}"
AE_PATH="$OUTPUT_DIR/ae/$AE_ID"
LATENT_DIR="$AE_PATH/latents/full69"
AE_MICROBATCH=${AE_MICROBATCH:-2}
DIFF_MICROBATCH=${DIFF_MICROBATCH:-4}
DUMP_BATCH_SIZE=${DUMP_BATCH_SIZE:-2}
AE_GLOBAL_BATCH=${AE_GLOBAL_BATCH:-64}
AE_STEPS=${AE_STEPS:-100000}
DIFF_GLOBAL_BATCH=${DIFF_GLOBAL_BATCH:-256}
DIFF_STEPS=${DIFF_STEPS:-125000}
OBS_GLOBAL_BATCH=${OBS_GLOBAL_BATCH:-8}
OBS_STEPS=${OBS_STEPS:-100000}
for value in "$AE_MICROBATCH" "$DIFF_MICROBATCH" "$DUMP_BATCH_SIZE" \
    "$AE_GLOBAL_BATCH" "$AE_STEPS" "$DIFF_GLOBAL_BATCH" "$DIFF_STEPS" \
    "$OBS_GLOBAL_BATCH" "$OBS_STEPS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo 'Batch sizes and update counts must be positive integers' >&2; exit 2
    fi
done
if (( START_INDEX == 0 && AE_GLOBAL_BATCH % (AE_MICROBATCH * TOTAL_PROCESSES) )); then
    echo "AE_GLOBAL_BATCH must be divisible by AE_MICROBATCH times total GPUs ($TOTAL_PROCESSES)" >&2; exit 2
fi
if (( START_INDEX <= 3 && STOP_INDEX >= 3 && DIFF_GLOBAL_BATCH % (DIFF_MICROBATCH * TOTAL_PROCESSES) )); then
    echo "DIFF_GLOBAL_BATCH must be divisible by DIFF_MICROBATCH times total GPUs ($TOTAL_PROCESSES)" >&2; exit 2
fi
if (( STOP_INDEX >= 4 && OBS_GLOBAL_BATCH % TOTAL_PROCESSES )); then
    echo "OBS_GLOBAL_BATCH must be divisible by total GPUs ($TOTAL_PROCESSES), with microbatch 1" >&2; exit 2
fi

run_entry() {
    local label=$1 nproc=$2 entry=$3
    shift 3
    local logfile="$OUTPUT_DIR/logs/$label.log"
    if (( NODES > 1 )); then logfile="$OUTPUT_DIR/logs/$label.node$NODE_ID.log"; fi
    local -a command
    if (( nproc == 0 )); then
        command=(python "$TRAIN_ROOT/$entry" "$@")
    elif (( NODES > 1 )); then
        command=(torchrun "--nnodes=$NODES" "--node_rank=$NODE_ID"
            "--nproc_per_node=$nproc" "--master_addr=$RENDEZVOUS_HOST"
            "--master_port=$RENDEZVOUS_PORT" "$TRAIN_ROOT/$entry" "$@")
    else
        command=(torchrun --standalone "--nproc_per_node=$nproc" "$TRAIN_ROOT/$entry" "$@")
    fi
    printf '[%s] ' "$label"
    printf '%q ' "${command[@]}"
    printf '\n'
    if (( DRY_RUN )); then return; fi
    mkdir -p "$OUTPUT_DIR/logs"
    {
        printf '\n[%s] ' "$label"
        printf '%q ' "${command[@]}"
        printf '\n'
    } >> "$logfile"
    (
        cd "$(dirname "$TRAIN_ROOT/$entry")"
        "${command[@]}"
    ) 2>&1 | tee -a "$logfile"
}

require_file() {
    if (( ! DRY_RUN )) && [[ ! -f "$1" ]]; then
        echo "Required file missing: $1" >&2; exit 1
    fi
}

printf 'latent_channels=%s, nodes=%s, node_rank=%s, GPUs/node=%s, total_GPUs=%s, stages=%s..%s\nOutput: %s\n' \
    "$CHANNELS" "$NODES" "$NODE_ID" "$GPUS" "$TOTAL_PROCESSES" "$START_STAGE" "$STOP_STAGE" "$OUTPUT_DIR"
printf 'Hourly ERA data: %s\nTarget normalization: %s\n' "$RISDA_ERA5_1H" "$TARGET_MEAN"
printf 'Training: 2017-2024 ERA; validation: Q1 2025; diffusion: 6 hourly frames, anchor=4\n'
printf 'Global batch / updates: AE=%s/%s, diffusion=%s/%s, observations=%s/%s\n' \
    "$AE_GLOBAL_BATCH" "$AE_STEPS" "$DIFF_GLOBAL_BATCH" "$DIFF_STEPS" "$OBS_GLOBAL_BATCH" "$OBS_STEPS"

if (( STOP_INDEX >= 4 )); then
    : "${RISDA_OBS_ROOT:?Set RISDA_OBS_ROOT (see configs/paths.env.example)}"
    for path in "$OBS_ERA_DATA/mean_std/era5_1h_mean.npy" "$OBS_ERA_DATA/mean_std/era5_1h_std.npy" "$TARGET_MEAN" "$TARGET_STD"; do
        require_file "$path"
    done
fi
if (( ! DRY_RUN )) && [[ -f "$AE_PATH/config.yaml" ]]; then
    python - "$AE_PATH/config.yaml" "$CHANNELS" "$TARGET_MEAN" "$TARGET_STD" <<'PY'
import sys
from omegaconf import OmegaConf
c = OmegaConf.load(sys.argv[1])
if int(c.ae.lat_channels) != int(sys.argv[2]):
    raise SystemExit('Existing AE has different latent channels; choose a new output/id')
if (str(c.normalization.target_mean_path), str(c.normalization.target_std_path)) != tuple(sys.argv[3:]):
    raise SystemExit('Existing AE has different normalization; choose a new output/id')
PY
fi
if (( START_INDEX == 0 )); then
    for path in "$SOURCE_MEAN" "$SOURCE_STD" "$TARGET_MEAN" "$TARGET_STD"; do require_file "$path"; done
    run_entry ae "$GPUS" experiments/autoencoder/train_1h.py \
        --gpus "$TOTAL_PROCESSES" --nodes "$NODES" --output-dir "$OUTPUT_DIR" --runid "$AE_ID" \
        "ae.lat_channels=$CHANNELS" \
        "normalization.source_mean_path=$SOURCE_MEAN" "normalization.source_std_path=$SOURCE_STD" \
        "normalization.target_mean_path=$TARGET_MEAN" "normalization.target_std_path=$TARGET_STD" \
        "train.batch_element_per_gpu=$AE_MICROBATCH" "train.batch_size_per_step=$AE_GLOBAL_BATCH" \
        "train.update_steps=$AE_STEPS"
fi

if (( START_INDEX <= 1 && STOP_INDEX >= 1 )); then
    require_file "$AE_PATH/model_best.pth"
    if (( ! DRY_RUN )) && [[ -e "$LATENT_DIR/era5.h5" ]]; then
        echo "Latent dump already exists: $LATENT_DIR/era5.h5; use --skip-to stats (or a new --output)" >&2
        exit 1
    fi
    # Single-process dump. Limit chunk/buffer sizes for 512/1024-channel files.
    run_entry dump 0 experiments/autoencoder/dump_1h.py \
        "model_path=$AE_PATH" "+id=full69" "data_path=$RISDA_ERA5_1H" \
        "start_date=2017-01-01" "end_date=2025-12-31" "start_hour=0" "end_hour=23" \
        "batch_size=$DUMP_BATCH_SIZE" "num_chunks=64" "save_every=16"
fi

if (( START_INDEX <= 2 && STOP_INDEX >= 2 )); then
    require_file "$LATENT_DIR/era5.h5"
    run_entry stats 0 scripts/data/latent_stats_v1.py \
        "latent_path=$LATENT_DIR/era5.h5" "start_date=2017-01-01" "end_date=2024-12-31" \
        "++start_hour=0" "++end_hour=23" "subchunk_size=64"
fi

if (( STOP_INDEX >= 3 )); then
    require_file "$LATENT_DIR/ae/model.pth"
    require_file "$LATENT_DIR/stats.pth"
    if (( ! DRY_RUN )); then
        python - "$LATENT_DIR" "$CHANNELS" <<'PY'
import sys
from pathlib import Path
import h5py
import torch
from omegaconf import OmegaConf
p = Path(sys.argv[1])
channels = int(sys.argv[2])
c = OmegaConf.load(p / 'ae/config.yaml')
if tuple(c.ae.latent_shape) != (23, 47, channels):
    raise SystemExit('AE latent shape does not match the requested channel count')
with h5py.File(p / 'era5.h5') as h:
    if tuple(h['latents'].shape[1:]) != (1081, channels):
        raise SystemExit('era5.h5 shape does not match the requested channel count')
s = torch.load(p / 'stats.pth', map_location='cpu', weights_only=False)
if any(s[k].numel() != channels for k in ('mean', 'std')):
    raise SystemExit('stats.pth does not match the requested channel count')
PY
    fi
fi

if (( START_INDEX <= 3 && STOP_INDEX >= 3 )); then
    run_entry diffusion "$GPUS" experiments/diffusion/train_diff_1h.py \
        --gpus "$TOTAL_PROCESSES" --nodes "$NODES" --output_dir "$OUTPUT_DIR" "ae_run=$AE_ID" "latent_dump=full69" \
        "train.blanket_size=6" "train.blanket_dt=1" "train.anchor_frame=4" "train.anchor_hour_interval=6" \
        "train.batch_element_per_gpu=$DIFF_MICROBATCH" "train.batch_size_per_step=$DIFF_GLOBAL_BATCH" \
        "train.update_steps=$DIFF_STEPS"
fi

OBS_COMMON=(
    "era_latent_dir=$LATENT_DIR" "output_dir=$OUTPUT_DIR" "++output_group=obs2era_1h"
    "data.era_data_path=$OBS_ERA_DATA"
    "++data.era_normalization.source_mean_path=$OBS_ERA_DATA/mean_std/era5_1h_mean.npy"
    "++data.era_normalization.source_std_path=$OBS_ERA_DATA/mean_std/era5_1h_std.npy"
    "++data.era_normalization.target_mean_path=$TARGET_MEAN"
    "++data.era_normalization.target_std_path=$TARGET_STD"
    "train.batch_size=1" "train.batch_size_per_step=$OBS_GLOBAL_BATCH" "train.update_steps=$OBS_STEPS"
)
if (( START_INDEX <= 4 && STOP_INDEX >= 4 )); then
    read -r -a instruments <<< "${RISDA_SATELLITES:-atms amsua mhs mwts mwhs}"
    if [[ "$INSTRUMENT" != all ]]; then instruments=("$INSTRUMENT"); fi
    for instrument in "${instruments[@]}"; do
        run_entry "satellite_$instrument" "$GPUS" experiments/observations/train_obs2era.py \
            "${OBS_COMMON[@]}" "data.era_cadence_hours=1" "data.sate_name=$instrument" \
            "data.use_satzen=true" "+id=obs2era_${instrument}_c${CHANNELS}"
    done
fi

if (( STOP_INDEX == 5 )); then
    run_entry convi "$GPUS" experiments/observations/train_convi_cressman_obs2era.py \
        "${OBS_COMMON[@]}" "data.alignment_mode=convi_qc_to_convi_qc" \
        "id=obs2era_convi_cressman_r10_qc_to_qc_c${CHANNELS}"
fi

printf 'Stages finished. ERA latent directory: %s\n' "$LATENT_DIR"
printf 'Denoiser checkpoints: %s/denoisers/<run_id>/\n' "$LATENT_DIR"
