import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except:
    pass
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except:
    pass


class Timer:
    def __init__(
        self,
        logger=None,
        format_str="{:.3f}[s]",
        prefix=None,
        suffix=None,
        sep=" ",
        verbose=0,
    ):
        if prefix:
            format_str = str(prefix) + sep + format_str
        if suffix:
            format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None
        self.verbose = verbose

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        if self.verbose is None:
            return
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


def reduce_mem_usage(df, verbose=True):
    """DataFrameの型変換してメモリ削減するやつ"""
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64))
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    dfs.append(df[col].astype(np.float32))
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    dfs.append(df[col].astype(np.float32))
                else:
                    dfs.append(df[col].astype(np.float64))
        else:
            dfs.append(df[col])

    df_out = pd.concat(dfs, axis=1)
    if verbose:
        end_mem = df_out.memory_usage().sum() / 1024**2
        num_reduction = str(100 * (start_mem - end_mem) / start_mem)
        print(
            f"Mem. usage decreased to {str(end_mem)[:3]}Mb:  {num_reduction[:2]}% reduction"
        )
    return df_out


def get_logger(filename):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}")
    handler2.setFormatter(Formatter("%(message)s"))
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger


def decorate(s: str, decoration=None):
    if decoration is None:
        decoration = "=" * 20
    else:
        decoration *= 20

    return " ".join([decoration, str(s), decoration])


def setup(cfg):
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # use kaggle api (need kaggle token)
    f = open(cfg.api_path, "r")
    json_data = json.load(f)
    os.environ["KAGGLE_USERNAME"] = json_data["username"]
    os.environ["KAGGLE_KEY"] = json_data["key"]

    cfg.input_path = os.path.join(cfg.base_path, "input")
    cfg.output = os.path.join(cfg.base_path, "output")
    cfg.dataset_path = os.path.join(cfg.base_path, "dataset")

    cfg.output_exp_path = os.path.join(cfg.output, cfg.exp)
    cfg.exp_model_path = os.path.join(cfg.output_exp_path, "model")
    cfg.exp_preds_path = os.path.join(cfg.output_exp_path, "preds")

    cfg.log_path = Path(cfg.base_path) / f"output/log"
    if not cfg.log_path.is_dir():
        cfg.log_path.mkdir(parents=True)

    # make dirs
    for d in [cfg.input_path, cfg.exp_model_path, cfg.exp_preds_path, cfg.log_path]:
        os.makedirs(d, exist_ok=True)

    if len(os.listdir(cfg.input_path)) == 0:
        # load dataset
        subprocess.run(
            f"kaggle competitions download -c {cfg.competition} -p {cfg.input_path}",
            shell=True,
        )
        filepath = os.path.join(cfg.input_path, cfg.competition + ".zip")
        subprocess.run(f"unzip -d {cfg.input_path} {filepath}", shell=True)

    seed_everything(cfg.seed)
    return cfg


def dataset_create_new(dataset_name, upload_dir):
    dataset_metadata = {}
    dataset_metadata["id"] = f'{os.environ["KAGGLE_USERNAME"]}/{dataset_name}'
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = dataset_name
    with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)
    api = KaggleApi()
    api.authenticate()
    api.dataset_create_new(folder=upload_dir, convert_to_csv=False, dir_mode="tar")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except:
        pass
