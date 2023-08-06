import torch.nn as nn
from transformers import AutoConfig, AutoModel

from kagglib.nn import AttentionPooling, freeze


class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gpu_optimize_config = cfg.gpu_optimize_config
        self.config = AutoConfig.from_pretrained(
            cfg.MODEL_PATH, output_hidden_states=True
        )
        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout": 0.0,
                "hidden_dropout_prob": 0.0,
                "attention_dropout": 0.0,
                "attention_probs_dropout_prob": 0,
            }
        )
        self.model = AutoModel.from_pretrained(cfg.MODEL_PATH, config=self.config)
        self.pool = AttentionPooling(self.config.hidden_size)
        # self.weighted_layer_pool = WeightedLayerPooling(self.config.num_hidden_layers)
        # self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, cfg.num_classes)
        self._init_weights(self.fc)
        self.ln = nn.LayerNorm(self.config.hidden_size)
        self._init_weights(self.ln)

        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)

        # Freeze
        if self.gpu_optimize_config["freezing"]:
            freeze(self.model.encoder.layer[:4])

        # Gradient Checkpointing
        if self.gpu_optimize_config["gradient_checkpoint"]:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_state = outputs[0]
        feature = self.pool(last_state, inputs["attention_mask"])
        # all_layer_embeddings = outputs[1]
        # feature = self.weighted_layer_pool(all_layer_embeddings)
        # feature = self.pool(feature, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        # batch, hidden_size
        feature = self.feature(inputs)
        feature = self.ln(feature)
        # feature1 = self.drop1(feature)
        # feature2 = self.drop2(feature)
        # feature3 = self.drop3(feature)
        # feature4 = self.drop4(feature)
        # feature5 = self.drop5(feature)
        # feature = (feature1 + feature2 + feature3 + feature4 + feature5) / 5
        output = self.fc(feature)
        return output.squeeze()


# initialize layer
def reinit_bert(model):
    """_summary_

    Args:
        model (AutoModel): _description_

    Returns:
        model (AutoModel): _description_

    Usage:
        model = reinit_bert(model)
    """
    for layer in model.model.encoder.layer[-1:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    return model
