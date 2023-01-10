from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


def get_StratifiedKFold_list(X, y=None, groups=None, n_splits=5, seed=42):
    """StratifiedKFoldのlistを取得"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_list = list(cv.split(X, y))
    return cv_list


def get_StratifiedGroupKFold_list(X, y=None, groups=None, n_splits=5, seed=42):
    """StratifiedGroupKFoldのlistを取得"""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_list = list(cv.split(X, y, groups))
    return cv_list
