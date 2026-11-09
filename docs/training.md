# Training, artifacts and resume

`python -m risda train C GPUS` invokes the preserved staged launcher:
`ae → dump → stats → diffusion → satellite → convi`.
Use `--skip-to` and `--stop-after` to select a contiguous range. Always reuse
the same `--output` and `--ae-id` when continuing the same pipeline.

Default architecture and schedules:

| Component | Settings |
| --- | --- |
| ERA AE | 69 input channels, resize 32, widths `[128,256,384,512,768,1024]`, 3 blocks/stage, latent C=128, noise 0.01 |
| AE optimization | SOAP, 100000 updates, global batch 64, microbatch 2 |
| Diffusion | DiT, 6 hourly frames, frame 4 at 00/06/12/18 UTC, 125000 updates, global batch 256, microbatch 4 |
| Observation Encoders | 100000 updates, global batch 8, microbatch 1, frozen ERA Decoder |
| Satellite inputs | Standardized TB + union mask + cosine zenith |
| CONVI inputs | Four surface variables + four observation masks, preprocessed into nine encoder input channels |

AE and diffusion training use 2017–2024 with Q1 2025 validation. Observation
training uses matched timestamps before 2025 and Q1 2025 validation. Latent dumping
covers 2017–2025; statistics use 2017–2024 only. Input arrays stored in hourly
normalization are explicitly converted to the target ERA mean/std basis. Never
change the normalization files without rebuilding the dependent models and caches.

Environment overrides include `AE_MICROBATCH`, `AE_GLOBAL_BATCH`, `AE_STEPS`,
`DIFF_MICROBATCH`, `DIFF_GLOBAL_BATCH`, `DIFF_STEPS`, `OBS_GLOBAL_BATCH`, `OBS_STEPS`
and `DUMP_BATCH_SIZE`. Global batches must divide over microbatch × all workers.

## Multiple nodes

Activate the same environment on each node and mount the same data and output
directories. Launch on **each** node with a different `NODE_RANK`:

```bash
# Node 0 (node 1 uses NODE_RANK=1 with the other values unchanged).
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.0.2.10 MASTER_PORT=29500 \
  python -m risda train 512 8 --stop-after ae --output /shared/risda_training
```

Replace the documentation address with the reachable address of node 0.
With multiple nodes, starting at AE defaults to stopping after AE. Run dump/stats
once with `--nodes 1 --node-rank 0`; then launch diffusion on every node. These
two export/statistics stages cannot run concurrently on multiple nodes.

## Resume

AE and observation training retain their original checkpoint-resume code. An
interrupted AE run can be restarted with the same output and run ID; checkpoint
availability and training-state recovery follow `train_1h.py` and `appa/save.py`.
This is not a claim of bitwise deterministic resumption across topology changes.
The diffusion trainer currently creates a fresh run ID on a normal invocation;
`--skip-to diffusion` is stage selection, not automatic exact-state resume.
Dump refuses to overwrite an existing completed `era5.h5`; restart at `stats`
after a completed export. Do not run two writers against the same training directory.

Native output layout for C=128:

```text
<training-output>/
  ae/era_c128/
    config.yaml, model_best.pth, model_last.pth, ...
    latents/full69/
      ae/config.yaml, ae/model.pth
      era5.h5, stats.pth
      denoisers/<run_id>/config.yaml, model_best.pth, ...
  obs2era_1h/obs2era_<satellite>_c128/
  obs2era_1h/obs2era_convi_cressman_r10_qc_to_qc_c128/
  logs/
```

Select these actual directories in the assimilation config. The source release
does not bundle trained checkpoints or convert historical checkpoint paths
automatically. Saved model configs also contain data/normalization paths: update
those when relocating artifacts **before** recalibrating. Cache identities are
strict and must not be manually changed to bypass a mismatch.

