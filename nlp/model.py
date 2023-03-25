import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
from transformers import AutoConfig, AutoModel


def freeze(module):
    """
    Freezes module's parameters.
    """

    for parameter in module.parameters():
        parameter.requires_grad = False


class AttentionPooling(nn.Module):
    """
    Usage:
        self.pool = AttentionPooling(self.config.hidden_size)
    """
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

        self._init_weights(self.attention)

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

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class WeightedLayerPooling(nn.Module):
    """
    Usage:
        def __init__():
            self.weighted_layer_pool = WeightedLayerPooling(self.config.num_hidden_layers)
        def feature(inputs):
            outputs = self.model(**inputs)
            all_layer_embeddings = outputs[1]
            feature = self.weighted_layer_pool(all_layer_embeddings)
            feature = self.pool(feature, inputs['attention_mask'])
            return feature
    """
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(
                torch.tensor(
                    [1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float
                )
            )
        )

    def forward(self, ft_all_layers):
        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start :, :, :, :]

        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(all_layer_embedding.size())
        )
        weighted_average = (weight_factor * all_layer_embedding).sum(
            dim=0
        ) / self.layer_weights.sum()

        return weighted_average


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


class Mixout(InplaceFunction):
    """
    Usage: model = replace_mixout(model)
    """

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "A mix probability of mixout has to be between 0 and 1,"
                " but got {}".format(p)
            )
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training
        if ctx.p == 0 or not ctx.training:
            return input
        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)
        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)
        if ctx.p == 1:
            output = target
        else:
            output = (
                (1 - ctx.noise) * target + ctx.noise * output - ctx.p * target
            ) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)


class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.target = target
        self.p = p

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(
            input, mixout(self.weight, self.target, self.p, self.training), self.bias
        )

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out",
            self.p,
            self.in_features,
            self.out_features,
            self.bias is not None,
        )


def replace_mixout(model, prob=0.2):
    for sup_module in model.modules():
        for name, module in sup_module.named_children():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.Linear):
                target_state_dict = module.state_dict()
                bias = True if module.bias is not None else False
                new_module = MixLinear(
                    module.in_features,
                    module.out_features,
                    bias,
                    target_state_dict["weight"],
                    prob,
                )
                new_module.load_state_dict(target_state_dict)
                setattr(sup_module, name, new_module)
    return model


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
