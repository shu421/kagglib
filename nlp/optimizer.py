from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n],
            "lr": decoder_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


def get_optimizer_grouped_parameters(cfg, model):
    """Layerwise Learning Rate Decay"""
    model_type = "model"
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if model_type not in n],
            "lr": cfg.decoder_lr,
            "weight_decay": 0.0,
        },
    ]
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(
        getattr(model, model_type).encoder.layer
    )
    layers.reverse()
    lr = cfg.encoder_lr
    for layer in layers:
        optimizer_grouped_parameters += [
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": cfg.weight_decay,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

        lr *= cfg.lr_weight_decay
    return optimizer_grouped_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * cfg.num_warmup_steps_rate),
            num_training_steps=num_train_steps,
        )
    elif cfg.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * cfg.num_warmup_steps_rate),
            num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles,
        )
    return scheduler
