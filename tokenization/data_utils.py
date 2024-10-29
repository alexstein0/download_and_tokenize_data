import datasets
from datasets import load_dataset, load_from_disk
from typing import List, Dict, Any
import multiprocessing
import os
from tokenization import utils
import random
import logging

log = logging.getLogger(__name__)

threads = utils.get_cpus()

def get_dataset_from_source(source: str, data_set_config):
    hf_path = data_set_config.path

    qualifier = data_set_config.qualifier
    split = data_set_config.split
    fields = data_set_config.fields

    cache_dir="/fs/cml-projects/llm-pretraining/datasets/alex" # TODO
    if data_set_config.rows is not None:
        split = f"split{data_set_config.rows}"

    if qualifier is not None:
        data = load_dataset(hf_path, qualifier, split=split, num_proc=threads, cache_dir = cache_dir)
    else:
        data = load_dataset(hf_path, split=split, num_proc=threads, cache_dir = cache_dir)
    log.info(f"Loaded dataset: {source}")

    if len(fields) > 0:
        data = data.select_columns(fields)
    return data

def filter_dataset(data, data_set_config):
    def concat_columns(example: Dict[str, Any], cols: List[str]):
        # TODO  improve this
        text = ""
        for k in cols:
            text += example[k] + " "
        example["text"] = text[:-1]
        return example

    if data_set_config.max_entries_in_dataset is not None:
        rows = int(data_set_config.max_entries_in_dataset)
    else:
        rows = -1

    split = data_set_config.split
    fields = data_set_config.fields

    if split:
        # split into train/test/any other
        try:
            data = data[split]
        except:
            log.info(f"Split: {split} not in data")
        col_names = data.column_names
    else:
        col_names = data["train"].column_names
    
    if "text" not in col_names:
        # if "text" is there, use it.  Otherwise look at "fields" and convert to "text"
        if len(fields) > 0:
            data = data.select_columns(fields)
        data = data.map(concat_columns, num_proc=threads, fn_kwargs={"cols": fields})  #, batched=True)

    data = data.select_columns("text")

    if rows > 0:
        # if only want a certain number of rows
        start = len(data) - rows
        if start > 0:
            start = random.randrange(start)
        log.info(f"Selecting rows from range {start}-{start+rows}")
        data = data.select(range(start, start + rows - 1))

    return data

def get_data(cfg_data, save_raw: bool = False) -> List[datasets.Dataset]:
    data_list = []
    sources = cfg_data.sources.keys()
    data_dir = cfg_data.data_dir
    for source in sources:
        data_set_config = cfg_data.sources[source]
        raw_extension = data_set_config.get('raw_extension', source)
        raw_path = os.path.join(data_dir, "raw", raw_extension)
        
        try:
            data = load_from_disk(raw_path)
            log.info(f"Loaded dataset: {source} from local: {raw_path}")
        except:
            data = get_dataset_from_source(source, data_set_config)
            log.info(f"Loaded dataset: {source} from hf")
            if save_raw and raw_path is not None:
                data.save_to_disk(raw_path, num_proc=threads)
                log.info(f"Saved: {source} to {raw_path}")

        data = filter_dataset(data, data_set_config)
        log.info(f"Data filtered and processed")
        data_list.append(data)

    return data_list


def get_data_generator(ds_new: datasets.Dataset, max_size: int = -1, max_pct: float = 1.0):

    total_len = len(ds_new)
    total_len = int(total_len * max_pct)
    if max_size >= 0:
        total_len = min(max_size, total_len)

    def batch_iterator(batch_size=1024):
        for i in range(0, total_len, batch_size):
            rows = ds_new[i: i + batch_size]
            yield rows["text"]

    return batch_iterator, total_len


def get_data_generator_shard(ds_new: datasets.Dataset, max_size: int = -1, max_pct: float = 1.0, shard_num: int = -1, total_shards: int = -1):
    total_len = len(ds_new)
    total_len = int(total_len * max_pct)
    if shard_num > 0:
        total_shards = max(total_shards, shard_num)
        skip_start = int(total_len * ((shard_num - 1) / total_shards))
        skip_len = int(total_len * (1 / total_shards))
        total_len = int(total_len - skip_len)
    else:
        skip_start = total_len
        skip_len = 0
    if max_size >= 0:
        total_len = min(max_size, total_len)

    def batch_iterator(batch_size=1024):
        for i in range(0, total_len, batch_size):
            if i > skip_start:
                i += skip_len
            rows = ds_new[i: i + batch_size]
            yield rows["text"]

    return batch_iterator, total_len


def sample_equal_amount(raw_dataset_list: List[datasets.Dataset]) -> List[datasets.Dataset]:
    min_size = min(len(x) for x in raw_dataset_list)
    new_ds_list = []
    for ds in raw_dataset_list:
        start = len(ds) - min_size
        if start > 0:
            start = random.randrange(start)
        sliced_ds = ds.select(range(start, start+min_size - 1))
        new_ds_list.append(sliced_ds)
    return new_ds_list


def combine_and_shuffle(data_cfg, raw_dataset_list: List[datasets.Dataset]) -> datasets.Dataset:
    if data_cfg.data_sampling == "even":
        raw_dataset_list = sample_equal_amount(raw_dataset_list)

    log.info("Concatenating data")
    ds_new = datasets.concatenate_datasets(raw_dataset_list)
    if data_cfg.shuffle:
        log.info("Shuffling data")
        ds_new = ds_new.shuffle()

    return ds_new