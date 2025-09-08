# Side Classifier (2D-Encoder + Attention)
# For Neural Foraminal Narrowing & Subarticular Stenosis severity classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from typing import Optional

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False, flatten=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps
        self.flatten = flatten

    def _gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def forward(self, x):
        ret = self._gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            return ret[:, :, 0, 0]
        else:
            return ret


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class RSNA2024Loss(nn.Module):
    def __init__(self,
                 conditions: list[str] = ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing',
                                          'right_neural_foraminal_narrowing', 'left_subarticular_stenosis',
                                          'right_subarticular_stenosis'],
                 levels: list[str] = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'],
                 ce_loss: dict = dict(name='CrossEntropyLoss', weight=[1.0, 2.0, 4.0]),
                 condition_weight: Optional[list[float]] = None,
                 sevear_loss: bool = False,
                 overall_loss_weight: float = 1.0,
                 sevear_loss_weight: float = 0.5):
        super().__init__()
        self.conditions = conditions
        self.levels = levels
        self.sevear_loss = sevear_loss
        self.overall_loss_weight = overall_loss_weight
        self.sevear_loss_weight = sevear_loss_weight

        ce_loss_name = ce_loss.pop('name')
        if ce_loss_name == 'CrossEntropyLoss':
            if 'weight' in ce_loss:
                weight = ce_loss.pop('weight')
                ce_loss['weight'] = torch.tensor(weight)
            self.ce_loss = nn.CrossEntropyLoss(**ce_loss)
        elif ce_loss_name == 'FocalLoss':
            if 'alpha' in ce_loss:
                alpha = ce_loss.pop('alpha')
                ce_loss['alpha'] = torch.tensor(alpha)
            self.ce_loss = FocalLoss(**ce_loss)
        else:
            raise ValueError(f'{ce_loss_name} is not supported.')

        if condition_weight is None:
            condition_weight = [1.0] * len(conditions)
        self.condition_weight = condition_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = dict()
        partial_losses = []
        for cond_idx, cond in enumerate(self.conditions):
            for level_idx, level in enumerate(self.levels):
                logit = logits[:, cond_idx, level_idx]
                target = targets[:, cond_idx, level_idx]
                if torch.any(target != -100):
                    partial_loss = self.ce_loss(logit, target) * self.condition_weight[cond_idx]
                    partial_losses.append(partial_loss)

        overall_loss = torch.mean(torch.stack(partial_losses))
        losses['overall_loss'] = overall_loss.item()
        losses['loss'] = overall_loss
        return losses


# ===============================
# Model Blocks
# ===============================
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class RSNA2024TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class RSNA2024AttentionNet(nn.Module):
    def __init__(self,
                 timm_model: dict,
                 loss: dict,
                 num_degenerative_conditions: int = 5,
                 num_segments: int = 5,
                 num_classes: int = 3,
                 use_planes: list = ['sagittal_t1', 'sagittal_t2', 'axial_t2']):
        super().__init__()
        self._num_degenerative_conditions = num_degenerative_conditions
        self._num_segments = num_segments
        self._num_classes = num_classes
        self._use_planes = use_planes
        self._in_channels = timm_model.get('in_chans', 3)

        base_model = None
        for plane in self._use_planes:
            base_model = timm.create_model(**timm_model)
            layers = list(base_model.children())[:-2]
            setattr(self, f'{plane}_backbone', nn.Sequential(*layers))
            setattr(self, f'{plane}_gap', GeM(flatten=True, p_trainable=True))

        if "efficientnet" in timm_model['model_name']:
            backbone_out_channels = base_model.num_features
        else:
            backbone_out_channels = base_model.num_features

        self.transformer = RSNA2024TransformerBlock(input_dim=backbone_out_channels,
                                                    num_heads=8, ff_hidden_dim=512)

        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.fc = nn.Linear(backbone_out_channels,
                            num_degenerative_conditions * num_segments * num_classes)
        self.target_loss = self._build_loss(**loss)

    def _build_loss(self, name: str, **kwargs: dict) -> nn.Module:
        if name == 'RSNA2024Loss':
            return RSNA2024Loss(**kwargs)
        else:
            raise ValueError(f'{name} is not supported.')

    def forward(self,
                sagittal_t1_images: torch.Tensor,
                sagittal_t2_images: torch.Tensor,
                axial_t2_images: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                force_loss_execute: bool = False) -> dict:
        outputs = dict()
        images = dict(
            sagittal_t1=sagittal_t1_images,
            sagittal_t2=sagittal_t2_images,
            axial_t2=axial_t2_images,
        )

        feats = dict()
        for plane in self._use_planes:
            plane_images = images[plane]
            plane_backbone = getattr(self, f'{plane}_backbone')
            plane_gap = getattr(self, f'{plane}_gap')

            batch_size, plane_num_slices, h, w = plane_images.shape
            plane_images = plane_images.view(-1, self._in_channels, h, w)
            plane_feats = plane_backbone(plane_images)
            plane_feats = plane_gap(plane_feats)
            plane_feats = plane_feats.view(batch_size, plane_num_slices, -1)
            feats[plane] = plane_feats

        combined_features = torch.cat(list(feats.values()), dim=1)
        combined_features = combined_features.permute(1, 0, 2)
        combined_features = self.transformer(combined_features)
        combined_features = combined_features.mean(dim=0)

        if self.training:
            logits = sum([self.fc(dropout(combined_features)) for dropout in self.dropouts]) / len(self.dropouts)
        else:
            logits = self.fc(combined_features)

        logits = logits.view(-1, self._num_degenerative_conditions, self._num_segments, self._num_classes)
        outputs['logits'] = logits

        if self.training or force_loss_execute:
            losses = self.loss(logits, targets)
            outputs['losses'] = losses

        return outputs

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        return self.target_loss(logits, targets)


# ===============================
# Test Drive
# ===============================
if __name__ == '__main__':
    model = RSNA2024AttentionNet(
        timm_model={"model_name": "efficientnet_b0", "pretrained": False,
                    'features_only': False, 'in_chans': 1, 'drop_rate': 0.3, 'drop_path_rate': 0.2},
        loss={"name": "RSNA2024Loss"},
        num_degenerative_conditions=5,
        num_segments=5,
        num_classes=3
    )
    out = model(torch.randn(4, 15, 224, 224),
                torch.randn(4, 15, 224, 224),
                torch.randn(4, 10, 224, 224),
                torch.randint(0, 3, (4, 5, 5)))
    print(out['logits'].shape)
    if 'losses' in out:
        print(out['losses'])
