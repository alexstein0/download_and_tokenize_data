"""System utilities."""
"""this code is courtesy of Jonas"""
import socket
import sys

import os
import csv
import yaml
import psutil
import pynvml

import multiprocess  # hf uses this for some reason
import collections

import torch
import torch._inductor.config
# import transformers
from typing import List


import json
import random
import numpy as np
import time
import datetime
import tempfile

import logging
import hydra
from omegaconf import OmegaConf, open_dict

log = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "0"

def main_launcher(cfg, main_fn, job_name=""):
    """This is boiler-plate code for a launcher."""
    launch_time = time.time()
    # Set definitive random seed:
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    # TODO
    # Decide GPU and possibly connect to distributed setup
    setup, kWh_counter = system_startup(cfg)
    # # Initialize wanDB
    # if cfg.wandb.enabled:
    #     _initialize_wandb(setup, cfg)
    log.info("--------------------------------------------------------------")
    log.info(f"--------------Launching {job_name} run! ---------------------")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    metrics = main_fn(cfg, setup)
    metrics = collect_system_metrics(cfg, metrics, kWh_counter, setup)

    log.info("-------------------------------------------------------------")
    log.info(f"Finished running job {cfg.name} with total train time: " f"{str(datetime.timedelta(seconds=time.time() - launch_time))}")
    if is_main_process():
        metrics = flatten(metrics)
        dump_metrics(cfg, metrics)
        # Export to wandb:
        if cfg.wandb.enabled:
            import wandb

            for k, v in metrics.items():
                wandb.run.summary[k] = v

        # if torch.cuda.is_available():
        #     max_alloc = f"{torch.cuda.max_memory_allocated(setup['device'])/float(1024**3):,.3f} GB"
        #     max_reserved = f"{torch.cuda.max_memory_reserved(setup['device'])/float(1024**3):,.3f} GB"
        #     log.info(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
        #     log.info(f"{metrics['kWh']:.2e} kWh of electricity used for GPU(s) during job.")
    log.info("-----------------Shutdown complete.--------------------------")


def system_startup(cfg):
    """Decide and print GPU / CPU / hostname info. Generate local distributed setting if running in distr. mode.

    Set all required and interesting environment variables.
    """
    torch.backends.cudnn.benchmark = cfg.impl.benchmark
    torch.set_float32_matmul_precision(cfg.impl.matmul_precision)
    if cfg.impl.tf32_allowed:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    # Huggingface settings
    if cfg.impl.enable_huggingface_offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["SAFETENSORS_FAST_GPU"] = "1"

    allowed_cpus_available = get_cpus()

    try:
        ram = psutil.Process().rlimit(psutil.RLIMIT_RSS)[0] / (2**30)
    except:
        log.warning("Cannot find process")
        ram = 0

    # Distributed launch?
    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend=cfg.impl.dist_backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run = os.environ.get("TORCHELASTIC_RUN_ID", "unknown")
        threads_per_gpu = max(1, min(allowed_cpus_available // max(1, torch.cuda.device_count()), cfg.impl.threads))
        log.info(
            f"Distributed worker initialized on rank {global_rank} (local rank {local_rank}) "
            f"with {world_size} total processes. OMP Threads set to {threads_per_gpu}. Run ID is {run}."
        )
        log.setLevel(logging.INFO if is_main_process() else logging.ERROR)
    else:
        threads_per_gpu = max(1, min(allowed_cpus_available, cfg.impl.threads))
        global_rank = local_rank = 0

    torch.set_num_threads(threads_per_gpu)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_gpu)

    # datasets will automatically disable tokenizer parallelism when needed:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_RS_NUM_CPUS"] = str(threads_per_gpu)
    max_dataset_memory = f"{psutil.virtual_memory().total // 2 // max(torch.cuda.device_count(), 1)}"
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = max_dataset_memory

    # Construct setup dictionary:
    dtype = getattr(torch, cfg.impl.default_precision)  # :> dont mess this up
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        log.info(f"GPU : {torch.cuda.get_device_name(device=device)}. CUDA: {torch.version.cuda}.")

        # Populate kwH counter:
        pynvml.nvmlInit()
        miilijoule_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(device.index))
        kWh_counter = dict(initial_value=miilijoule_start * 1e-6 / 3600)  # kilojoule per hour
    else:
        kWh_counter = dict(initial_value=float("NaN"))
    setup = dict(device=device, dtype=dtype)
    python_version = sys.version.split(" (")[0]

    if local_rank == 0:
        log.info(f"Platform: {sys.platform}, Python: {python_version}, PyTorch: {torch.__version__}")
        log.info(f"CPUs: {allowed_cpus_available}, GPUs: {torch.cuda.device_count()} (ram: {ram}GB) on {socket.gethostname()}.")

    # 100% reproducibility?
    if cfg.impl.deterministic:
        set_deterministic()
    if cfg.seed is not None:
        if is_main_process():
            log.info(f"Seeding with random seed {cfg.seed} on rank 0.")
        set_random_seed(cfg.seed + 10 * global_rank)

    return setup, kWh_counter


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def num_processes():
    num_procs = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    return num_procs

# def save_summary(table_name, cfg, stats, local_time, setup, original_cwd=True):
#     """Save two summary tables. A detailed table of iterations/loss+acc and a summary of the end results."""
#     # 1) detailed table:
#     for step in range(len(stats["loss"])):
#         iteration = dict()
#         for key in stats:
#             iteration[key] = stats[key][step] if step < len(stats[key]) else None
#         save_to_table(".", f"{cfg.name}_convergence_results", dryrun=cfg.dryrun, **iteration)
#
#     def _maybe_record(key, step=-1):
#         try:
#             return stats[key][step]
#         except (IndexError, ValueError):
#             return ""
#
#     if "data" in cfg:
#         processed_dataset_dir = f"{cfg.data.name}_{checksum_config(cfg.data)}"
#     else:
#         processed_dataset_dir = None
#     base_name = cfg.base_dir.rstrip(os.sep).split(os.sep)[-1]
#     local_folder = os.getcwd().split(base_name)[1].lstrip(os.sep)
#
#     # 2) save a reduced summary
#     if table_name == "pretrain":
#         summary = dict(
#             name=cfg.name,
#             budget=cfg.budget,
#             dataset="_".join(processed_dataset_dir.split("_")[:-1]),
#             backend=cfg.impl.name,
#             arch=" ".join(cfg.arch.architectures),
#             loss=_maybe_record("loss"),
#             final_step=_maybe_record("step"),
#             final_epoch=_maybe_record("epoch"),
#             step_time=np.mean(stats["train_time"]) if len(stats["train_time"]) > 0 else "",
#             loss100k=_maybe_record("loss", step=100_000 // cfg.impl.print_loss_every_nth_step),
#             loss200k=_maybe_record("loss", step=200_000 // cfg.impl.print_loss_every_nth_step),
#             loss300k=_maybe_record("loss", step=300_000 // cfg.impl.print_loss_every_nth_step),
#             total_time=str(datetime.timedelta(seconds=local_time)).replace(",", ""),
#             batch_size=cfg.train.batch_size,
#             lr=cfg.train.optim.lr,
#             warmup=cfg.train.warmup_steps,
#             steps=cfg.train.steps,
#             # System settings:
#             seed=cfg.seed,
#             dataset_hash=processed_dataset_dir.split("_")[-1],
#             base_dir=cfg.base_dir,
#             impl_path=cfg.impl.path,
#             local_folder=local_folder,
#             # # Dump configs from here on:
#             **{f"Data_{k}": v for k, v in cfg.data.items()},
#             **{f"Arch_{k}": v for k, v in cfg.arch.items()},
#             **{f"Train_{k}": v for k, v in cfg.train.items()},
#         )
#     else:
#         summary = dict(
#             name=cfg.name,
#             backend=cfg.impl.name,
#             checkpoint=cfg.eval.checkpoint,
#             loss=_maybe_record("loss"),
#             avg_loss=_maybe_record("avg_loss"),
#             final_epoch=_maybe_record("epoch"),
#             step_time=np.mean(stats["train_time"]) if len(stats["train_time"]) > 0 else "",
#             total_time=str(datetime.timedelta(seconds=local_time)).replace(",", ""),
#             batch_size=cfg.eval.batch_size,
#             lr=cfg.eval.optim.lr,
#             warmup=cfg.eval.warmup_steps,
#             # System settings:
#             seed=cfg.seed,
#             base_dir=cfg.base_dir,
#             impl_path=cfg.impl.path,
#             local_folder=local_folder,
#             # # Dump configs from here on:
#             **{f"Eval_{k}": v for k, v in cfg.eval.items()},
#         )
#     location = os.path.join(cfg.original_cwd, "tables") if original_cwd else "tables"
#     save_to_table(location, f"{table_name}_reports", dryrun=cfg.dryrun, **summary)


def save_to_table(out_dir, table_name, dryrun, **kwargs):
    """Save keys to .csv files."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f"table_{table_name}.csv")
    fieldnames = list(kwargs.keys())
    # Read or write header
    try:
        with open(fname, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)  # noqa  # this line is testing the header
            # assert header == fieldnames[:len(header)]  # new columns are ok, but old columns need to be consistent
            # dont test, always write when in doubt to prevent erroneous table deletions
    except Exception as e:  # noqa
        if not dryrun:
            # print('Creating a new .csv table...')
            with open(fname, "w") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
                writer.writeheader()
        else:
            pass

    # Write a new row
    if not dryrun:
        # Add row for this experiment
        with open(fname, "a") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writerow(kwargs)
    else:
        pass


def set_random_seed(seed=233):
    """."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    # Can't be too careful :>


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def avg_n_dicts(dicts):
    """https://github.com/wronnyhuang/metapoison/blob/master/utils.py."""
    # given a list of dicts with the same exact schema, return a single dict with same schema whose values are the
    # key-wise average over all input dicts
    means = {}
    for dic in dicts:
        for key in dic:
            if key not in means:
                if isinstance(dic[key], list):
                    means[key] = [0 for entry in dic[key]]
                else:
                    means[key] = 0
            if isinstance(dic[key], list):
                for idx, entry in enumerate(dic[key]):
                    means[key][idx] += entry / len(dicts)
            else:
                means[key] += dic[key] / len(dicts)
    return means


def dump_metrics(cfg, metrics):
    """Simple yaml dump of metric values."""

    filepath = f"metrics_{cfg.name}.yaml"
    sanitized_metrics = dict()
    for metric, val in metrics.items():
        if not isinstance(val, List):
            print(metric, val)
        try:
            sanitized_metrics[metric] = np.asarray(val).item()
        except ValueError:
            sanitized_metrics[metric] = np.asarray(val).tolist()
    with open(filepath, "w") as yaml_file:
        yaml.dump(sanitized_metrics, yaml_file, default_flow_style=False)


def _initialize_wandb(setup, cfg):
    if is_main_process():
        import wandb

        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        settings = wandb.Settings(start_method="thread")
        settings.update({"git_root": cfg.original_cwd})
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            settings=settings,
            name=cfg.name,
            mode="disabled" if cfg.dryrun else None,
            tags=cfg.wandb.tags if len(cfg.wandb.tags) > 0 else None,
            config=config_dict,
        )
        run.summary["GPU"] = torch.cuda.get_device_name(device=setup["device"]) if torch.cuda.device_count() > 0 else ""
        run.summary["numGPUs"] = torch.cuda.device_count()


def wandb_log(stats, cfg):
    if cfg.wandb.enabled:
        if is_main_process():
            import wandb

            wandb.log({k: v[-1] for k, v in stats.items()}, step=stats["step"][-1] if "step" in stats else None)


def flatten(d, parent_key="", sep="_"):
    """Straight-up from https://stackoverflow.com/a/6027615/3775820."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_cpus() -> int:
    # Number of threads
    try:
        return min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))  # covering both affinity and phys.
    except:
        pass
    try:
        return os.cpu_count()  # when running on mac
    except:
        return 1


def collect_system_metrics(cfg, metrics, kWh_counter, setup):
    # Finalize some compute metrics:
    metrics["GPU"] = torch.cuda.get_device_name(device=setup["device"]) if torch.cuda.device_count() > 0 else ""
    metrics["numGPUs"] = torch.cuda.device_count()
    metrics["VRAM"] = torch.cuda.max_memory_allocated(setup["device"]) / float(1 << 30)
    metrics["RAM"] = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    if torch.cuda.device_count() == 1:
        metrics["kWh"] = get_kWh(kWh_counter, setup)
    else:
        if torch.distributed.is_initialized():
            local_kWh = get_kWh(kWh_counter, setup)
            kWh_comm = torch.as_tensor(local_kWh).cuda() if torch.cuda.is_available() else kWh_comm.float()
            torch.distributed.all_reduce(kWh_comm, torch.distributed.ReduceOp.SUM, async_op=False)
            metrics["kWh"] = kWh_comm.item()
        else:
            metrics["kWh"] = float("NaN")
    return metrics


def get_kWh(kWh_counter, setup):
    miilijoule_final = pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(setup["device"].index))
    kWh_final = miilijoule_final * 1e-6 / 3600  # kilojoule per hour
    kWh = kWh_final - kWh_counter["initial_value"]
    return kWh


def pathfinder(cfg):
    with open_dict(cfg):
        cfg.original_cwd = hydra.utils.get_original_cwd()
        # ugliest way to get the absolute path to output subdir
        if not os.path.isabs(cfg.base_dir):
            base_dir_full_path = os.path.abspath(os.getcwd())
            while os.path.basename(base_dir_full_path) != cfg.base_dir:
                base_dir_full_path = os.path.dirname(base_dir_full_path)
                if base_dir_full_path == "/":
                    raise ValueError("Cannot find base directory.")
            cfg.base_dir = base_dir_full_path

        cfg.impl.path = os.path.expanduser(cfg.impl.path)
        if not os.path.isabs(cfg.impl.path):
            cfg.impl.path = os.path.join(cfg.base_dir, cfg.impl.path)
    return cfg
