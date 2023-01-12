import os
import pickle

import numpy as np

from kagglib.utils.utils import decorate
from kagglib.tabular.model import get_model, save_model, load_model


def train_cv(cfg, train_feat_df, target_df, cv_list, metrics_dict, LOGGER):
    """交差検証を実行"""
    oof = np.zeros(len(train_feat_df))
    models = []

    for i_fold, (train_idx, valid_idx) in enumerate(cv_list):
        LOGGER.info(decorate(f"Fold{i_fold}", decoration="=="))
        filepath = os.path.join(cfg.OUTPUT_EXP, f"{cfg.gbdt_model}_fold_{i_fold}.pkl")

        X_train = train_feat_df.iloc[train_idx].to_numpy()
        X_valid = train_feat_df.iloc[valid_idx].to_numpy()
        y_train = target_df.iloc[train_idx].to_numpy()
        y_valid = target_df.iloc[valid_idx].to_numpy()

        model = get_model(cfg.gbdt_model)
        model.fit(X_train, y_train, X_valid, y_valid)
        save_model(filepath, model)

        model = load_model(filepath)
        models.append(model)
        y_prob = model.predict(X_valid)
        y_prob = y_prob[:, 1].squeeze()
        y_preds_ = np.where(y_prob >= 0.5, 1, 0)
        metrics_dict_scored = metrics_dict(y_valid, y_preds_)

        for key in metrics_dict_scored.keys():
            LOGGER.info(f"{key}: {np.round(metrics_dict_scored[key], 5)}")
        oof[valid_idx] = y_prob

    LOGGER.info(decorate("OOF"))
    oof_ = np.where(oof >= 0.5, 1, 0)
    metrics_dict_scored = metrics_dict(target_df, oof_)
    for key in metrics_dict_scored.keys():
        LOGGER.info(f"Fold{i_fold} {key}: {np.round(metrics_dict_scored[key], 5)}")

    pickle.dump(oof, open(os.path.join(cfg.EXP_PREDS, "oof.pkl"), "wb"))
    return oof, models


def predict_cv(cfg, test_feat_df):
    """Inference"""
    prob_folds = []

    for i_fold in range(cfg.n_fold):
        filepath = os.path.join(cfg.OUTPUT_EXP, f"{cfg.gbdt_model}_fold_{i_fold}.pkl")
        model = load_model(filepath)
        y_prob = model.predict(test_feat_df)
        preds = np.where(y_prob >= 0.5, 1, 0)
        prob_folds.append(y_prob)

    pickle.dump(prob_folds, open(os.path.join(cfg.EXP_PREDS, "prob_folds.pkl"), "wb"))
    return prob_folds
