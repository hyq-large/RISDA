#!/usr/bin/env python
"""Decoder residual calibration and equal-source constant variance.

Satellites use the frozen Direct E2O decoder with exact-hour cos(satzen).
CONVI uses the frozen ERA physical decoder and all valid QC surface observations.
All observations are in the original normalized observation coordinates.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from risda import calibrate as common
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from appa.config.hydra import compose
from experiments.observations import calibration_residuals as paper
from experiments.observations.evaluate_guided_six_frame import _build_convi_dataset

ROOT, OUTPUT, SOURCES = common.ROOT, common.OUTPUT, common.SOURCES


def directory(smoke=False):
    return common.stage_dir(smoke) / "decoder"


def mean_constants(moments):
    """Average within each source first, then average sources equally."""
    result = {}
    for field in ("variance", "mse"):
        values = [torch.as_tensor(moments[s][field], dtype=torch.float64).reshape(-1) for s in SOURCES]
        counts = [torch.as_tensor(moments[s]["counts"], dtype=torch.float64).reshape(-1) for s in SOURCES]
        if any(v.shape != n.shape or not torch.isfinite(v).all() or (v < 0).any() or (n <= 0).any() for v,n in zip(values,counts)):
            raise ValueError("Every source/channel must have valid moments and positive counts")
        result[field] = {
            "source_means": {s:float(v.mean()) for s,v in zip(SOURCES,values)},
            "equal_source": float(torch.stack([v.mean() for v in values]).mean()),
            "equal_channel": float(torch.cat(values).mean()),
            "count_weighted": float((torch.cat(values)*torch.cat(counts)).sum()/torch.cat(counts).sum()),
        }
    result.update(var_y=result["variance"]["equal_source"],
                  statistic="centered_population_variance",
                  aggregation="mean_over_sources(mean_over_channels(variance))",
                  source_weights={s:1.0 / len(SOURCES) for s in SOURCES},
                  other_aggregations_are_diagnostics_only=True)
    return result


def prepare():
    ref = common.REFERENCE
    cfg = compose(ref)
    OmegaConf.resolve(cfg)
    if tuple(cfg.enabled_sources) != SOURCES:
        raise ValueError("Calibration sources must match assimilation_config.enabled_sources")
    if cfg.calibration_period != common.PERIOD or Path(cfg.calibration_root).resolve() != OUTPUT:
        raise ValueError("Calibration output/period must match the assimilation configuration")
    era_dir = Path(cfg.model_path).resolve().parents[1]
    identity = common.paper.file_identity
    models = {}
    for source in SOURCES:
        entry = cfg.source_catalog[source].decoder
        physical = source == "convi"
        model_dir = era_dir / "ae" if physical else Path(entry.model_dir)
        data_model_dir = Path(entry.data_model_dir) if physical else model_dir
        model_config = OmegaConf.load(data_model_dir / "config.yaml")
        if Path(model_config.era_latent_dir).resolve() != era_dir:
            raise ValueError(f"{source} Decoder uses a different ERA latent space")
        checkpoint = "model" if physical else str(entry.checkpoint)
        path = model_dir / ("model.pth" if physical else f"model_{checkpoint}.pth")
        identities = {
            "checkpoint": identity(path, include_sha256=True),
            "model_config": identity(model_dir/"config.yaml", include_sha256=True),
            "data_config": identity(data_model_dir/"config.yaml", include_sha256=True),
            "era_latents": identity(era_dir/"era5.h5"),
            "ae_config": identity(era_dir/"ae/config.yaml", include_sha256=True),
        }
        norm = Path(cfg.convi.mean_std_dir if physical else model_config.data.sate_standard_dir)
        normalization_files = {str(p):identity(p,include_sha256=True) for p in sorted(norm.glob("*.npy"))}
        if physical:
            normalization = paper.physical_normalization_contract(era_dir/"ae/config.yaml",norm)
        else:
            assert paper.validate_direct_e2o_data_contract(model_config.data) == 1
            normalization = {"space":"standardized_tb"}
        models[source] = {
            "model_dir":str(model_dir.resolve()), "data_model_dir":str(data_model_dir.resolve()),
            "checkpoint":checkpoint, "identities":identities, "normalization_files":normalization_files,
            "precision":"float16",
            "decoder_kind":"frozen_era_autoencoder_physical" if physical else "direct_e2o_cos_satzen",
            "normalization":normalization,
        }
    protocol = {
        "version":1, "period":common.PERIOD, "window_start":common.START, "window_end":common.END,
        "time_scope":"hourly", "expected_hours":len(common.hourly_axis()), "sources":list(SOURCES),
        "statistic":"centered_population_variance", "aggregation":"mean_over_sources(mean_over_channels(variance))",
        "source_weights":{s:1.0 / len(SOURCES) for s in SOURCES}, "var_y_scale":1,
        "models":models, "reference":identity(ref,include_sha256=True),
        "convi":OmegaConf.to_container(cfg.convi,resolve=True),
        "script":identity(Path(__file__),include_sha256=True),
        "paper_calibrator":identity(Path(paper.__file__),include_sha256=True),
    }
    path = OUTPUT / "protocol/decoder_calibration_config.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError(f"Decoder calibration protocol changed: {path}")
    common.save_json(protocol,path)
    return protocol


def empty_totals(channels):
    return {"sum":torch.zeros(channels,dtype=torch.float64),
            "square":torch.zeros(channels,dtype=torch.float64),
            "counts":torch.zeros(channels,dtype=torch.int64)}


def add_dense(totals,residual,valid):
    if not torch.isfinite(residual[valid]).all():
        raise RuntimeError("Nonfinite Decoder residual at a valid observation")
    masked = torch.where(valid,residual,0.0).double()
    dims = tuple(range(masked.ndim-1))
    totals["sum"] += masked.sum(dim=dims)
    totals["square"] += masked.square().sum(dim=dims)
    totals["counts"] += valid.sum(dim=dims).long()


def worker(args):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    protocol = json.loads((OUTPUT/"protocol/decoder_calibration_config.json").read_text())
    spec = protocol["models"][args.source]
    common.check_identities(spec)
    times = common.requested_times(args.smoke)
    indices = list(range(args.rank,len(times),args.world_size))
    contract = {"protocol":protocol,"source":args.source,"rank":args.rank,"world_size":args.world_size,
                "smoke":args.smoke,"requested_times":[common.paper.format_hour(t) for t in times]}
    path = common.shard_path(directory(args.smoke),args.source,args.rank,args.world_size)
    state = common.load(path) if path.exists() else None
    if state is not None:
        if state["contract"] != contract:
            raise RuntimeError(f"Decoder shard contract changed: {path}")
        if state["complete"]:
            print(f"reuse {path}",flush=True)
            return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_cfg = OmegaConf.load(spec["identities"]["data_config"]["path"])
    era_dir = Path(spec["identities"]["era_latents"]["path"]).parent
    physical = args.source == "convi"
    if physical:
        dataset = _build_convi_dataset(model_cfg.data,OmegaConf.create(protocol["convi"]))
        decoder,latent_hw,_,output_hw,resize,*_ = paper.load_era_physical_decoder(era_dir,device)
        obs_index = common.paper._time_index(dataset.times)
        channels = 4
    else:
        dataset = paper.EraLatentSatelliteTbDataset(
            era_latent_path=spec["identities"]["era_latents"]["path"],cadence_hours=1,
            satellite_data_path=model_cfg.data.sate_data_path,satellite_name=model_cfg.data.sate_name,
            satellite_standard_dir=model_cfg.data.sate_standard_dir)
        channels = dataset.n_obs
        decoder = paper.build_direct_e2o_decoder(model_cfg,channels,device)
        decoder.load_state_dict(common.load(spec["identities"]["checkpoint"]["path"]),strict=True)
        decoder.eval().requires_grad_(False)
        obs_index = common.paper._time_index(dataset.dates)
    if state is None:
        state = {"contract":contract,"next_offset":0,"complete":False,"frames":[],"totals":empty_totals(channels)}
    totals = {k:v.to(device) for k,v in state["totals"].items()}
    started = time.monotonic()

    def checkpoint(complete=False):
        state.update(totals={k:v.cpu() for k,v in totals.items()},complete=complete,updated_utc=common.utc_now())
        common.save_torch(state,path)
        print(f"decoder {args.source} rank {args.rank}: {state['next_offset']}/{len(indices)} hours, "
              f"{time.monotonic()-started:.1f}s, complete={complete}",flush=True)

    with h5py.File(spec["identities"]["era_latents"]["path"],"r") as truth_file, torch.inference_mode():
        era_index = common.paper._time_index(np.asarray(truth_file["dates"]))
        for offset in range(state["next_offset"],len(indices)):
            timestamp = times[indices[offset]]
            key = common.paper.timestamp_key(timestamp)
            if key not in era_index:
                raise RuntimeError(f"Missing ERA latent truth: {timestamp}")
            frame = {"timestamp":common.paper.format_hour(timestamp),"observation_present":key in obs_index,"valid_values":0}
            before = totals["counts"].clone()
            if key in obs_index:
                if physical:
                    obs = dataset.prepare_surface_sparse(dataset.times[obs_index[key]])
                    if obs is not None:
                        pixel,channel,local,values = (obs[k].to(device) for k in ("pixel_index","channel_index","channel_local","values"))
                        assert torch.equal(channel,local+65)
                        latent = torch.as_tensor(truth_file["latents"][era_index[key]],dtype=torch.float32,device=device).unsqueeze(0)
                        with paper.precision_context(device,spec["precision"]):
                            decoded = paper.decode_era_latent(decoder,latent,latent_hw=latent_hw,output_hw=output_hw,resize=resize)
                            prediction = decoded[0,pixel,channel]
                        residual = (prediction.float()-values.float()).double()
                        if not torch.isfinite(residual).all():
                            raise RuntimeError("Nonfinite CONVI Decoder residual")
                        totals["sum"].scatter_add_(0,local,residual)
                        totals["square"].scatter_add_(0,local,residual.square())
                        totals["counts"].scatter_add_(0,local,torch.ones_like(local,dtype=torch.int64))
                else:
                    latent,target,valid,cos_satzen,date = dataset[obs_index[key]]
                    assert paper.timestamp_key(date.numpy()) == key
                    latent,target,valid,cos_satzen = (v.unsqueeze(0).to(device) for v in (latent,target,valid,cos_satzen))
                    with paper.precision_context(device,spec["precision"]):
                        prediction = decoder(latent,cos_satzen)
                    add_dense(totals,prediction.float()-target.float(),valid)
            delta = (totals["counts"]-before).cpu()
            frame["valid_values"] = int(delta.sum())
            frame.update({f"channel_{c}_count":int(v) for c,v in enumerate(delta)})
            state["frames"].append(frame)
            state["next_offset"] = offset+1
            if state["next_offset"] % 24 == 0:
                checkpoint()
    common.check_identities(spec)
    checkpoint(True)


def merge_source(source,world_size,smoke,protocol):
    root = directory(smoke)
    frames,totals = [],None
    spec = protocol["models"][source]
    common.check_identities(spec)
    for rank in range(world_size):
        state = common.load(common.shard_path(root,source,rank,world_size))
        assert state["complete"] and state["contract"]["protocol"] == protocol
        assert state["contract"]["source"] == source and state["contract"]["rank"] == rank
        assert [f["timestamp"] for f in state["frames"]] == [common.paper.format_hour(t) for t in common.requested_times(smoke)[rank::world_size]]
        if totals is None:
            totals = {k:v.clone() for k,v in state["totals"].items()}
        else:
            for k in totals:
                totals[k] += state["totals"][k]
        frames.extend(state["frames"])
    frames.sort(key=lambda f:f["timestamp"])
    assert [f["timestamp"] for f in frames] == [common.paper.format_hour(t) for t in common.requested_times(smoke)]
    assert int(totals["counts"].sum()) == sum(f["valid_values"] for f in frames)
    moments = paper.finalize_residual_moments(totals["sum"],totals["square"],totals["counts"])
    assert torch.allclose(moments["mse"],moments["variance"]+moments["bias"].square(),rtol=1e-10,atol=1e-12)
    result = {"kind":"hourly_decoder_centered_residual_var_y_v1","representation":"decoder",
              "source":source,"period":common.PERIOD,"window_start":common.START,"window_end":common.END,
              "window_end_inclusive":True,"expected_hours":len(common.hourly_axis()),"time_scope":"hourly",
              "smoke":smoke,"precision":spec["precision"],"num_requested_times":len(frames),
              "num_used_times":sum(f["valid_values"]>0 for f in frames),
              "missing_observation_timestamps":[f["timestamp"] for f in frames if not f["observation_present"]],
              "zero_support_timestamps":[f["timestamp"] for f in frames if f["observation_present"] and not f["valid_values"]],
              "selected_times":[f["timestamp"] for f in frames if f["observation_present"]],
              "used_timestamps":[f["timestamp"] for f in frames if f["valid_values"]],
              "space":spec["normalization"]["space"],"decoder_kind":spec["decoder_kind"],
              "metadata":spec,**moments,"var_y":moments["variance"].float(),
              "std":moments["variance"].sqrt(),"rmse":moments["mse"].sqrt()}
    if source == "convi":
        result.update(era_channel_indices=[65,66,67,68],
                      surface_channel_names=["t2m","u10","v10","msl"],
                      era_normalization=spec["normalization"])
    common.save_torch(result,root/"channelwise"/f"{source}_var_y.pth")
    common.write_csv(frames,root/"channelwise"/f"{source}_hourly_counts.csv")
    return result


def build(world_size,smoke,protocol):
    root = directory(smoke)
    moments = {s:merge_source(s,world_size,smoke,protocol) for s in SOURCES}
    constant = {"kind":"decoder_global_mean_var_y_v1","period":common.PERIOD,
                "window_start":common.START,"window_end":common.END,"expected_hours":len(common.hourly_axis()),
                "sources":list(SOURCES),"smoke":smoke,**mean_constants(moments),"var_y_scale":1,
                "created_utc":common.utc_now(),"channel_caches":{s:common.paper.file_identity(root/"channelwise"/f"{s}_var_y.pth",include_sha256=True) for s in SOURCES}}
    common.save_json(constant,root/"decoder_constant.json")
    channel_rows,rows = [],[]
    for s,m in moments.items():
        rows.append({"source":s,"channels":len(m["variance"]),"used_hours":m["num_used_times"],
                     "missing_hours":len(m["missing_observation_timestamps"]),"variance_mean":float(m["variance"].mean()),
                     "mse_mean":float(m["mse"].mean()),"bias_squared_mean":float(m["bias"].square().mean()),
                     "valid_values":int(m["counts"].sum()),"constant_source_weight":1.0 / len(SOURCES)})
        for c in range(len(m["variance"])):
            channel_rows.append({"source":s,"channel":c,"count":int(m["counts"][c]),
                                 **{k:float(m[k][c]) for k in common.FIELDS}})
    common.write_csv(rows,root/"summary.csv")
    common.write_csv(channel_rows,root/"channelwise_errors.csv")
    outputs = {"decoder_constant.json": common.paper.file_identity(root/"decoder_constant.json",include_sha256=True)}
    outputs.update({f"channelwise/{s}_var_y.pth": constant["channel_caches"][s] for s in SOURCES})
    common.save_json({"status":"complete","protocol":protocol,"smoke":smoke,"summary":rows,
                      "constant":constant,"outputs":outputs},root/"manifest.json")
    print(json.dumps({"summary":rows,"var_y_constant":constant["var_y"]},indent=2),flush=True)


def run(args):
    torch.set_num_threads(1)
    root = directory(args.smoke)
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/"run.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = prepare()
    gpus = args.gpus.split(",")
    if len(gpus) != len(set(gpus)) or not all(g.isdigit() for g in gpus):
        raise ValueError("--gpus must contain distinct GPU indices")
    status = {"pid":os.getpid(),"started_utc":common.utc_now(),"status":"running","gpus":gpus,
              "smoke":args.smoke,"completed_sources":[]}
    processes = []
    (root/"logs").mkdir(exist_ok=True)
    try:
        if args.after_encoder:
            status["status"] = "waiting_for_encoder"
            common.save_json(status,root/"status.json")
            while True:
                encoder_state = json.loads((common.stage_dir(False)/"status.json").read_text())
                if encoder_state["status"] == "complete":
                    break
                if encoder_state["status"] == "failed":
                    raise RuntimeError("Encoder calibration failed; Decoder queue not started")
                time.sleep(10)
            status["status"] = "running"
        for source in SOURCES:
            status.update(source=source,updated_utc=common.utc_now())
            common.save_json(status,root/"status.json")
            processes,handles = [],[]
            for rank,gpu in enumerate(gpus):
                cmd = [sys.executable,"-m","risda.calibrate_decoder","worker","--source",source,
                       "--rank",str(rank),"--world-size",str(len(gpus))]
                if args.smoke:
                    cmd.append("--smoke")
                handle = (root/"logs"/f"{source}_rank{rank}.log").open("a")
                handles.append(handle)
                handle.write(f"\n{common.utc_now()} {cmd!r}\n")
                handle.flush()
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1")
                processes.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None,0) for p in processes):
                    raise RuntimeError(f"Decoder {source} worker failed; see {root/'logs'}")
                time.sleep(2)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError(f"Decoder {source} failed; see {root/'logs'}")
            for handle in handles:
                handle.close()
            merge_source(source,len(gpus),args.smoke,protocol)
            status["completed_sources"].append(source)
            common.save_json(status,root/"status.json")
        build(len(gpus),args.smoke,protocol)
        status.update(status="complete",finished_utc=common.utc_now())
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        status.update(status="failed",error=str(exc),finished_utc=common.utc_now())
        raise
    finally:
        common.save_json(status,root/"status.json")


def main():
    global ROOT, OUTPUT, SOURCES
    from risda.calibration_config import configure
    remaining = configure(common)
    ROOT, OUTPUT, SOURCES = common.ROOT, common.OUTPUT, common.SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command",required=True)
    for mode in ("run","launch"):
        p = subs.add_parser(mode)
        p.add_argument("--gpus",default="0,1,2,3,4,5,6,7")
        p.add_argument("--smoke",action="store_true")
        p.add_argument("--after-encoder",action="store_true")
    subs.add_parser("prepare")
    p = subs.add_parser("worker")
    p.add_argument("--source",choices=SOURCES,required=True)
    p.add_argument("--rank",type=int,required=True)
    p.add_argument("--world-size",type=int,required=True)
    p.add_argument("--smoke",action="store_true")
    args = parser.parse_args(remaining)
    if args.command == "worker":
        worker(args)
    elif args.command == "prepare":
        prepare()
    elif args.command == "run":
        run(args)
    else:
        root = directory(args.smoke)
        root.mkdir(parents=True,exist_ok=True)
        cmd = [sys.executable,"-m","risda.calibrate_decoder","run","--gpus",args.gpus]
        if args.smoke:
            cmd.append("--smoke")
        if args.after_encoder:
            cmd.append("--after-encoder")
        with (root/"run.log").open("a") as log:
            proc = subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                    start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1"))
        (root/"run.pid").write_text(str(proc.pid)+"\n")
        print(f"Decoder calibration PID {proc.pid}; {root/'run.log'}")


if __name__ == "__main__":
    main()
