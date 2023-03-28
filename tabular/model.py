import pickle

import xgboost as xgb
import lightgbm as lgb


class XGBoost:
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

    def fit(self, X_train, y_train, X_valid, y_valid):

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        self.model = xgb.train(
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

    def predict(self, features):
        dtest = xgb.DMatrix(features)
        return self.model.predict(dtest)


class LightGBM:
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

    def fit(self, X_train, y_train, X_valid, y_valid):

        d_train = lgb.Dataset(X_train, label=y_train)
        d_valid = lgb.Dataset(X_valid, label=y_valid)

        self.model = lgb.train(
            params=self.model_params,
            train_set=d_train,
            valid_sets=[d_train, d_valid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(1000),
            ],
            **self.train_params,
        )

    def predict(self, features):
        return self.model.predict(features)


def get_model(cfg, gbdt_model):
    if gbdt_model == "LightGBM":
        model = LightGBM(cfg)
    elif gbdt_model == "XGBoost":
        model = XGBoost(cfg)
    return model


def save_model(filepath, model):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
