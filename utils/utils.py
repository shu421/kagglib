import os
import time
import json
import random
import subprocess
from pathlib import Path

import torch
import numpy as np
import pandas as pd

api_path = Path("/root/.kaggle/kaggle.json")
if api_path.is_file():
    from kaggle.api.kaggle_api_extended import KaggleApi


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
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
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

    cfg.INPUT = os.path.join(cfg.BASE_PATH, "input")
    cfg.OUTPUT = os.path.join(cfg.BASE_PATH, "output")
    cfg.SUBMISSION = os.path.join(cfg.BASE_PATH, "submission")
    cfg.DATASET = os.path.join(cfg.BASE_PATH, "dataset")

    cfg.OUTPUT_EXP = os.path.join(cfg.OUTPUT, cfg.EXP)
    cfg.EXP_MODEL = os.path.join(cfg.OUTPUT_EXP, "model")
    cfg.EXP_PREDS = os.path.join(cfg.OUTPUT_EXP, "preds")

    # make dirs
    for d in [cfg.INPUT, cfg.SUBMISSION, cfg.EXP_MODEL, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)

    if len(os.listdir(cfg.INPUT)) == 0:
        # load dataset
        subprocess.run(
            f"kaggle competitions download -c {cfg.COMPETITION} -p {cfg.INPUT}",
            shell=True,
        )
        filepath = os.path.join(cfg.INPUT, cfg.COMPETITION + ".zip")
        subprocess.run(f"unzip -d {cfg.INPUT} {filepath}", shell=True)

    for path in cfg.DATASET_PATH:
        datasetpath = os.path.join(cfg.DATASET, path.split("/")[1])
        if not os.path.exists(datasetpath):
            os.makedirs(datasetpath, exist_ok=True)
            subprocess.run(
                f"kaggle datasets download path -p {datasetpath}", shell=True
            )
            filepath = os.path.join(datasetpath, path.split("/")[1] + ".zip")
            subprocess.run(f"unzip -d {datasetpath} {filepath}", shell=True)

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
