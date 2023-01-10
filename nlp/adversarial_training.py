import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.cuda.amp import autocast


class FGM:
    """
    Reference:
        https://www.kaggle.com/competitions/tweet-sentiment-extraction/discussion/143764#809408
    Usage:
        fgm = FGM(model)

        for epoch in cfg.n_epoch:
            for inputs, labels in loader:

                scaler.scale(loss).backward()

                fgm.attack()
                with autocast(enabled=cfg.apex):
                    loss_adv, _ = model(inputs, labels)
                scaler.scale(loss_adv).backward()
                fgm.restore()
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}


class AWP:
    """
    Usage:
        for epoch in cfg.n_epoch:
            awp = AWP(
                model,
                optimizer,
                adv_lr=cfg.awp_lr,
                adv_eps=cfg.awp_eps
            )

            for inputs, labels in loader:

                scaler.scale(loss).backward()

                if epoch >= cfg.start_epoch:
                    loss = awp.attack_backward(inputs, labels)
                    scaler.scale(loss).backward()
                    awp._restore()
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        adv_param: str = "weight",
        adv_lr: float = 1.0,
        adv_eps: float = 0.01,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs: dict, labels: dict) -> Tensor:
        with autocast():
            self._save()
            self._attack_step()  # モデルを近傍の悪い方へ改変
            adv_loss, _ = self.model(inputs, labels)
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 直前に損失関数に通してパラメータの勾配を取得できるようにしておく必要あり
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
