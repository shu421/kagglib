import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from category_encoders import CountEncoder

from kagglib.utils.utils import Timer, decorate, reduce_mem_usage


class AbstractBaseBlock:
    """
    input_df
        targetを含む
    y
        input_dfに含まれるtargetの名前
    """

    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        raise NotImplementedError()


class IdentityBlock(AbstractBaseBlock):
    """そのまま使う特徴量"""

    def __init__(self, use_cols):
        self.use_cols = use_cols

    def transform(self, input_df):
        return input_df[self.use_cols].copy()


class WrapperBlock(AbstractBaseBlock):
    """関数のラッパー"""

    def __init__(self, func):
        self.func = func

    def transform(self, input_df: pd.DataFrame):
        return self.func(input_df.copy())


class LabelEncodingBlock(AbstractBaseBlock):
    """指定したカラムをラベルエンコード"""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, input_df, y=None):
        self.oe = OrdinalEncoder()
        self.oe.fit(input_df[self.cols])

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        output_df = input_df[self.cols].copy()
        output_df = self.oe.transform(output_df)
        output_df = pd.DataFrame(output_df)

        return output_df


class CountEncodingBlock(AbstractBaseBlock):
    """指定したカラムをカウントエンコード"""

    def __init__(self, cols, normalize):
        self.cols = cols
        self.normalize = normalize

    def fit(self, input_df, y=None):
        self.ce = CountEncoder(cols=self.cols, normalize=self.normalize)
        self.ce.fit(input_df[self.cols])

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        output_df = input_df[self.cols].copy()
        output_df = self.ce.transform(output_df)

        return output_df


class AggBlock(AbstractBaseBlock):
    """keyで集約したvaluesにfuncsを適応"""

    def __init__(self, key: str, values: list, funcs: dict):
        self.key = key
        self.values = values
        self.funcs = funcs

    def fit(self, input_df: pd.DataFrame, y=None):
        self.meta_df = input_df.groupby(self.key)[self.values].agg(self.funcs)

        # rename
        cols_level_0 = self.meta_df.columns.droplevel(0)
        cols_level_1 = self.meta_df.columns.droplevel(1)
        new_cols = [
            f"{cols_level_1[i]}_{cols_level_0[i]}_{self.key}"
            for i in range(len(cols_level_1))
        ]
        self.meta_df.columns = new_cols
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        output_df = (
            self.meta_df.copy()
            .reindex(input_df[self.key].values)
            .reset_index(drop=True)
        )
        return output_df


class TargetEncodingBlock(AbstractBaseBlock):
    """指定したカラムをtarget encofingする"""

    def __init__(self, col: str, func: str, cv_list: list):
        self.col = col
        self.func = func
        self.cv_list = cv_list

    def fit(self, input_df, y):
        output_df = input_df.copy()
        for i, (train_idx, valid_idx) in enumerate(self.cv_list):
            group = input_df.iloc[train_idx].groupby(self.col)[y]
            group = getattr(group, self.func)().to_dict()
            output_df.loc[valid_idx, f"{self.col}_{self.func}"] = input_df.loc[
                valid_idx, self.col
            ].map(group)

        self.group = input_df.groupby(self.col)[y]
        self.group = getattr(self.group, self.func)().to_dict()
        return output_df[[f"{self.col}_{self.func}"]].astype(np.float)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[f"{self.col}_{self.func}"] = (
            input_df[self.col].map(self.group).astype(np.float)
        )
        return output_df.astype(np.float)


class AdditiveTargetEncodingBlock(AbstractBaseBlock):
    """
    指定したカラムをadditive target encodingする
    ref:
        https://www.wikiwand.com/en/Additive_smoothing
        https://www.guruguru.science/competitions/19/discussions/857ed321-8ecf-41a8-93d1-93a2cfa39fa9/
    """

    def __init__(self, col: str, alpha: str, cv_list: list):
        self.col = col
        self.alpha = alpha
        self.cv_list = cv_list

    def additive_smoothing(self, x, alpha, p):
        """
        x
            input with target
        alpha
            smoothing parameter
        p
            input_df全体の平均値
        """
        n, k = x[0], x["target"]
        return (k + alpha * p) / (n + alpha)

    def fit(self, input_df, y):
        output_df = input_df.copy()
        self.p = input_df[y].mean()  # 全体のtargetの平均値
        for i, (train_idx, valid_idx) in enumerate(self.cv_list):
            add_te_dict = (
                pd.concat(
                    [
                        input_df.iloc[train_idx].groupby(self.col)[y].sum(),
                        input_df.iloc[train_idx].groupby(self.col).size(),
                    ],
                    axis=1,
                )
                .apply(lambda x: self.additive_smoothing(x, self.alpha, self.p), axis=1)
                .to_dict()
            )
            output_df.loc[valid_idx, f"{self.col}_alpha={self.alpha}"] = input_df.loc[
                valid_idx, self.col
            ].map(add_te_dict)

        # trainのfitの結果を保存
        self.add_te_dict = (
            pd.concat(
                [
                    input_df.iloc[train_idx].groupby(self.col)[y].sum(),
                    input_df.iloc[train_idx].groupby(self.col).size(),
                ],
                axis=1,
            )
            .apply(lambda x: self.additive_smoothing(x, self.alpha, self.p), axis=1)
            .to_dict()
        )
        return output_df[[f"{self.col}_alpha={self.alpha}"]].astype(float)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[f"{self.col}_alpha={self.alpha}"] = (
            input_df[self.col].map(self.add_te_dict).astype(float)
        )
        return output_df.astype(float)


class PercentileTargetEncodingBlock(AbstractBaseBlock):
    """
    指定したカラムをadditive target encodingする
    ref:
        https://www.guruguru.science/competitions/19/discussions/857ed321-8ecf-41a8-93d1-93a2cfa39fa9/
    """

    def __init__(self, col: str, alpha: str, cv_list: list):
        self.col = col
        self.alpha = alpha
        self.cv_list = cv_list

    def calc_percentile(self, x, y):
        n, k = x[0], x[y]
        ret = 0
        for i in range(k + 1):
            tmp = 1
            for x in range(n, n - i, -1):
                tmp *= x
            for x in range(i):
                tmp /= x + 1
            tmp *= self.POWER_p[i]
            tmp *= self.POWER_1_p[n - i]
            if i == k:
                tmp /= 2
            ret += tmp
        return ret

    def fit(self, input_df, y):
        output_df = input_df.copy()

        self.p = input_df[y].mean()
        # p^nと(1-p)^n
        self.POWER_p = [1] * 201
        for i in range(200):
            self.POWER_p[i + 1] *= self.POWER_p[i] * self.p
        self.POWER_1_p = [1] * 201
        for i in range(200):
            self.POWER_1_p[i + 1] *= self.POWER_1_p[i] * (1 - self.p)

        for i, (train_idx, valid_idx) in enumerate(self.cv_list):
            add_te_dict = (
                pd.concat(
                    [
                        input_df.iloc[train_idx].groupby(self.col)[y].sum(),
                        input_df.iloc[train_idx].groupby(self.col).size(),
                    ],
                    axis=1,
                )
                .apply(lambda x: self.calc_percentile(x, y), axis=1)
                .to_dict()
            )
            output_df.loc[valid_idx, f"{self.col}"] = input_df.loc[
                valid_idx, self.col
            ].map(add_te_dict)

        # trainのfitの結果を保存
        self.add_te_dict = (
            pd.concat(
                [
                    input_df.iloc[train_idx].groupby(self.col)[y].sum(),
                    input_df.iloc[train_idx].groupby(self.col).size(),
                ],
                axis=1,
            )
            .apply(lambda x: self.calc_percentile(x, y), axis=1)
            .to_dict()
        )
        return output_df[[f"{self.col}"]].astype(float)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[f"{self.col}"] = (
            input_df[self.col].map(self.add_te_dict).astype(float)
        )
        return output_df.astype(float)


class SVDBlock(AbstractBaseBlock):
    def __init__(self, cols, cfg, dim, title_vec):
        self.cols = cols
        self.cfg = cfg
        self.dim = dim
        self.svd = TruncatedSVD(n_components=dim, random_state=self.cfg.seed)
        self.title_vec = title_vec

    def fit(self, input_df, y=None):
        svd_dict = dict()

        output_df = pd.DataFrame()
        for svd_name, vec in zip(self.cols, [self.title_vec]):
            svd = TruncatedSVD(n_components=self.dim, random_state=self.cfg.seed)
            svd_vec = svd.fit_transform(vec)
            # svdと変換後のベクトルを保存
            svd_dict[svd_name] = svd
            for i_dim in range(self.dim):
                output_df[f"{svd_name}_{i_dim}"] = svd_vec[:, i_dim]

        pickle.dump(
            svd_dict,
            open(
                os.path.join(
                    self.cfg.OUTPUT_EXP, "_".join(self.cols) + "_svd_dict.pkl"
                ),
                "wb",
            ),
        )
        return output_df

    # def transform(self, input_df):
    #     tokenized_text = input_df[self.col].astype(str).parallel_apply(lambda x: " ".join(self.tokenizer.tokenize(x))).fillna("hogehoge")

    #     output_df = pd.DataFrame(self.pipe.transform(tokenized_text))
    #     output_df = output_df.add_prefix(f"Tfidf_SVD_{self.col}_")
    #     return output_df


def run_blocks(input_df, blocks, y=None, is_test=False):
    """_summary_

    Args:
        input_df (_type_): _description_
        blocks (_type_): _description_
        y (_type_, optional): _description_. Defaults to None.
        is_test (bool, optional): _description_. Defaults to False.

    Returns:
        output_df (pd.DataFrame): 特徴量が格納されたデータフレーム

    Usage:
        blocks = [IdentityBlock(use_cols=numeric_cols),
                LabelEncodingBlock(cols=cat_cols),
                CountEncodingBlock(cols=cat_cols, normalize=False),
                # *[TargetEncodingBlock(col=col,
                #                       func=func,
                #                       cv_list=cv_list) for col in cat_cols for func in ['mean', 'max', 'min', 'median']],
                *[AggBlock(key=key,
                            values=numeric_cols,
                            funcs=['min', 'max', 'mean', 'sum', 'std']) for key in cat_cols],
                *[WrapperBlock(func=func) for func in funcs],
                *[TfidfBlock(col=col, dim=128) for col in tfidf_cols]
                ]
    """
    output_df = pd.DataFrame()

    print(decorate("start run blocks...", "*"))

    with Timer(prefix=f"run is_test={is_test}"):
        for block in blocks:
            with Timer(prefix=f"\t- {str(block)}"):
                if not is_test:
                    out_i = block.fit(input_df, y=y)
                else:
                    out_i = block.transform(input_df)

            assert len(input_df) == len(out_i), block
            out_i = reduce_mem_usage(out_i, verbose=False)
            name = block.__class__.__name__
            output_df = pd.concat([output_df, out_i.add_suffix(f"@{name}")], axis=1)
    assert len(output_df.columns) == len(set(output_df.columns)), "col name duplicates"
    return output_df
