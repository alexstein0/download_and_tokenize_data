import hydra
import tokenization
from tokenization import data_utils
import logging

from typing import Dict
import time
import os
from hydra.utils import get_original_cwd
import datasets

from tokenization import utils

import multiprocessing
import psutil
import torch
import copy

from transformers import AutoTokenizer

log = logging.getLogger(__name__)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main_process(cfg, setup=None) -> Dict:
    log.info(f"\nCPUS:\n\tmultiprocessing:{multiprocessing.cpu_count()}\n"
             f"\tos_sched:{len(os.sched_getaffinity(0))}\n"
             f"\tos:{os.cpu_count()}\n"
             f"\tpsutil:{psutil.cpu_count(logical=False)}\n"
             f"\tcpu_affinity:{len(psutil.Process().cpu_affinity())}"
             )
    threads = utils.get_cpus()
    
    log.info("Loading data")
    raw_dataset_list = data_utils.get_data(cfg.data)
    dataset = {}
    for ds in raw_dataset_list:
        if isinstance(ds, datasets.DatasetDict):
            for split, v in ds.items():
                split_data = dataset.get(split, [])
                split_data.append(v)
                dataset[split] = split_data
        else:
            split = "train"
            split_data = dataset[split]
            split_data.append(ds)
            dataset[split] = split_data

            
    
    dataset = datasets.DatasetDict({k: data_utils.combine_and_shuffle(cfg.data, v) for k, v in dataset.items()})
    
    # if cfg.data.total_shards > 0:
    #     data_generator, dataset_length = data_utils.get_data_generator_shard(dataset, max_size=cfg.data.max_entries_in_dataset, max_pct=cfg.data.max_pct, shard_num=cfg.data.shard_num, total_shards=cfg.data.total_shards)
    # else:
    #     data_generator, dataset_length = data_utils.get_data_generator(dataset, max_size=cfg.data.max_entries_in_dataset, max_pct=cfg.data.max_pct)

    save_full = False
    if save_full:
        log.info(f"Saving data using {threads} threads")
        # cfg.data.max_entries_in_dataset = dataset_length
        # log.info(f"DATASET: {dataset} {f'but using {cfg.data.max_entries_in_dataset} rows' if cfg.data.max_entries_in_dataset != dataset_length else ''}")
        ds = {'_'.join(cfg.data.sources.keys())}
        ds = f"{'shuffled_' if cfg.data.shuffle else ''}{'sharded_' if cfg.data.total_shards > 0 else ''}{f'{cfg.data.data_sampling}_' if len(cfg.data.sources) > 1 else ''}{ds}"
        save_dir = cfg.data.data_dir
        if cfg.data.processed_extension is not None:
            save_dir = os.path.join(save_dir, cfg.data.processed_extension)
        dataset.save_to_disk(os.path.join(save_dir, 'processed', ds), num_proc=threads)
        log.info(f"Data Save Complete to: {ds}")

    # tokenization
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(
            [t + tokenizer.eos_token for t in example["text"]]),
        batched=True,
        batch_size=200,
        num_proc=threads,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    block_size = cfg.sequence_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples, idx):
        # Concatenate all texts.
        concatenated_examples = {
            k: [item for sublist in examples[k] for item in sublist] for k in examples.keys()
        }
        # concatenated_examples = examples
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["original_id"] = [idx for _ in range(len(result["input_ids"]))]

        return result

    if block_size > 0:
        tokenized_dataset = tokenized_dataset.map(
            lambda examples, idx: group_texts(examples, idx), 
            batched=True, 
            batch_size=200,
            num_proc=threads, 
            with_indices=True,
            desc="Grouping texts"
        )

    # [f'{i}: {len(x["input_ids"])}, {len(x["input_ids"])//block_size}' for i, x in enumerate(tokenized_dataset)]
    # [f'{x["original_id"]}: {len(x["input_ids"])}' for x in grouped_dataset]

    # post processing:
    tokenized_dataset = tokenized_dataset.select_columns(["input_ids", "attention_mask"])

    splits = list(tokenized_dataset.keys())
    if "input_ids" not in tokenized_dataset[splits[0]].column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    if "labels" not in tokenized_dataset[splits[0]].column_names:
        def add_labels(sample):
            sample["labels"] = copy.deepcopy(sample["input_ids"])
            return sample
        tokenized_dataset = tokenized_dataset.map(
            add_labels, desc="Adding labels", num_proc=threads)
    if "attention_mask" not in tokenized_dataset[splits[0]].column_names:
        def add_attention_mask(sample):
            sample["attention_mask"] = torch.ones(
                len(sample["input_ids"]), dtype=torch.int8)
            return sample
        tokenized_dataset = tokenized_dataset.map(
            add_attention_mask, desc="Adding attention mask", num_proc=threads)

    ds = f"{cfg.data.name}_tokenized_{cfg.tokenizer.split('/')[-1]}{'' if block_size == 0 else f'_{block_size}'}"
    save_dir = cfg.data.data_dir
    if cfg.data.processed_extension is not None:
        save_dir = os.path.join(save_dir, cfg.data.processed_extension)
    save_dir = os.path.join(save_dir, 'processed', ds)
    tokenized_dataset.save_to_disk(save_dir, num_proc=threads)
    log.info(f"Data Save Complete to: {save_dir}")

    return {}


@hydra.main(config_path="tokenization/config", config_name="preprocess_data", version_base="1.2")
def launch(cfg):

    tokenization.utils.main_launcher(cfg, main_process, job_name="download_and_save_data")


if __name__ == "__main__":
    launch()

