import pickle

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt


class AbstractGBDT:
    """GBDTの抽象クラス"""

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError


class XGBoost(AbstractGBDT):
    """xgboostのラッパー
    gbdt_model = "XGBoost"
    stopping_rounds = 50
    log_evaluation = False
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "random_state": seed,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        }
    train_params = {
        "num_boost_round": 99999,
        "verbose_eval": log_evaluation,
    }
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_params = self.cfg.model_params
        self.train_params = self.cfg.train_params

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        # train mode
        if X_valid is not None:
            self.train_mode = True
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            self.booster = xgb.train(
                self.model_params,
                dtrain,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                callbacks=[
                    xgb.callback.EarlyStopping(
                        self.cfg.stopping_rounds,
                        save_best=True,
                        maximize=False,
                    )
                ],
                **self.train_params,
            )

        else:
            self.train_mode = False
            assert (
                self.train_params["num_boost_round"] >= 9999
            ), "num_boost_round should be set."
            dtrain = xgb.DMatrix(X_train, label=y_train)

            self.booster = xgb.train(
                self.model_params,
                dtrain,
                **self.train_params,
            )

    def predict(self, features):
        dtest = xgb.DMatrix(features)
        return self.booster.predict(dtest)


class LightGBM(AbstractGBDT):
    """lightgbmのラッパー

    gbdt_model = "LightGBM"
    model_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "binary_logloss",
        "learning_rate": 0.3,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "drop_seed": seed,
        'verbose': -1,
        "n_jobs": -1

    }
    train_params = {
        "num_boost_round": 100000,
    }
    stopping_rounds = 50
    log_evaluation = 25
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_params = self.cfg.model_params
        self.train_params = self.cfg.train_params

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_valid is not None:
            self.train_mode = True

            d_train = lgb.Dataset(X_train, label=y_train)
            d_valid = lgb.Dataset(X_valid, label=y_valid)

            self.booster = lgb.train(
                params=self.model_params,
                train_set=d_train,
                valid_sets=[d_train, d_valid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.cfg.stopping_rounds, verbose=True
                    ),
                    lgb.log_evaluation(self.cfg.log_evaluation),
                ],
                **self.train_params,
            )
        else:
            self.train_mode = False
            assert (
                self.train_params["num_boost_round"] >= 9999
            ), "num_boost_round should be set."
            d_train = lgb.Dataset(X_train, label=y_train)

            self.booster = lgb.train(
                params=self.model_params,
                train_set=d_train,
                **self.train_params,
            )

    def predict(self, features):
        return self.booster.predict(features)


class CatBoost(AbstractGBDT):
    """catboostのラッパー

    gbdt_model = "CatBoost"
    model_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "task_type": "GPU",
        "iterations": 100000,
        "learning_rate": 0.05,
        "depth": 4,
        "l2_leaf_reg": 10,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 100,
        "verbose": 100,
    }
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_params = self.cfg.model_params
        self.train_params = self.cfg.train_params

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_valid is not None:
            self.train_mode = True
            d_train = cbt.Pool(X_train, label=y_train)
            d_valid = cbt.Pool(X_valid, label=y_valid)

            self.booster = cbt.train(
                params=self.model_params,
                dtrain=d_train,
                evals=d_valid,
                # **self.train_params,
            )
        else:
            self.train_mode = False
            assert (
                self.train_params["num_boost_round"] >= 9999
            ), "num_boost_round should be set."
            d_train = cbt.Pool(X_train, label=y_train)

            self.booster = cbt.train(
                params=self.model_params,
                dtrain=d_train,
                **self.train_params,
            )

    def predict(self, features):
        return self.booster.predict(features)


def get_model(cfg):
    if cfg.gbdt_model == "LightGBM":
        model = LightGBM(cfg)
    elif cfg.gbdt_model == "XGBoost":
        model = XGBoost(cfg)
    elif cfg.gbdt_model == "CatBoost":
        model = CatBoost(cfg)
    return model


def save_model(model, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
