import json

import wandb


def set_wandb(cfg, name, group, config_path="/.kaggle/wandb.json"):
    try:
        wandb_config = json.load(open(config_path, "rb"))
        secret_value_0 = wandb_config["key"]
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = "must"
        print(
            "If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize"
        )

    def class2dict(f):
        return dict(
            (name, getattr(f, name)) for name in dir(f) if not name.startswith("__")
        )

    run = wandb.init(
        project=cfg.COMPETITION,
        name=name,
        config=class2dict(cfg),
        group=group,
        job_type="train",
        anonymous=anony,
    )
    return run
