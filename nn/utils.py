import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(
        self,
        patience=7,
        mode="max",
        delta=0.0,
        verbose=False,
        save_model=True,
        model_path="checkpoint.pt",
    ):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.save_model = save_model
        self.model_path = model_path
        self.early_stop = False
        self.counter = 0
        self.best_pred = None
        self.best_loss = None
        self.best_epoch = None

        if self.mode == "min":
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

    def __call__(self, score, pred, loss, model, epoch=None):
        if self.mode == "min":
            improved_cond = score < self.best_score + self.delta
        elif self.mode == "max":
            improved_cond = score > self.best_score + self.delta
        if improved_cond:
            self.save_checkpoint(score, model)
            self.best_score = score
            self.best_pred = pred
            self.best_loss = loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(
                f"Validation score improved ({self.best_score:.4f} --> {score:.4f}). Saving model to {self.model_path}"
            )
        if self.save_model:
            torch.save(model.state_dict(), self.model_path)
