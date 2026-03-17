from __future__ import annotations

from typing import Optional, Tuple

import torchvision.models as tv_models
from torchvision.models import get_model_weights
import torchvision.transforms as T
from LossFunctions import UnTargeted, Targeted
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

_DATASET_NUM_CLASSES = {
    "imagenet": 1000,
    "imagenet1k": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "fashionmnist": 10,
    "svhn": 10,
    "caltech101": 101,
    "caltech256": 256,
}


def split_transform_from_weights(weights):

    resize = weights.transforms().resize_size
    crop = weights.transforms().crop_size
    mean = weights.transforms().mean
    std = weights.transforms().std

    spatial = T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=mean, std=std)

    return spatial, normalize

def get_torchvision_model(
    model_name,
    dataset_name=None,
    pretrained=True,
    num_classes=None,
):

    if not hasattr(tv_models, model_name):
        raise ValueError(f"Unknown model {model_name}")

    model_fn = getattr(tv_models, model_name)

    if pretrained:
        weights_enum = get_model_weights(model_name).DEFAULT
        model = model_fn(weights=weights_enum)

        spatial, normalize = split_transform_from_weights(weights_enum)

        return model, spatial, normalize

    kwargs = {}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes

    model = model_fn(weights=None, **kwargs)

    return model, None, None


def build_intersection_objective(base_model, model_name, normalize_transform, x_clean_hwc, target_class, device,
                                 explain_method="simple_grad", ig_steps=100):
    clean_map_np = compute_explain_map(
        base_model=base_model,
        model_name=model_name,
        normalize_transform=normalize_transform,
        x_hwc=x_clean_hwc,
        target_class=target_class,
        device=device,
        explain_method=explain_method,
        ig_steps=ig_steps,
    )

    def objective2_fn(x_adv_hwc):
        adv_maps = compute_explain_map_batch(
            base_model=base_model,
            model_name=model_name,
            normalize_transform=normalize_transform,
            x_bhwc=np.expand_dims(np.asarray(x_adv_hwc, dtype=np.float32), axis=0),
            target_class=target_class,
            device=device,
            explain_method=explain_method,
            ig_steps=ig_steps,
        )
        return get_intersection(clean_map_np, adv_maps[0])

    def objective2_batch(x_adv_bhwc):
        adv_maps = compute_explain_map_batch(
            base_model=base_model,
            model_name=model_name,
            normalize_transform=normalize_transform,
            x_bhwc=x_adv_bhwc,
            target_class=target_class,
            device=device,
            explain_method=explain_method,
            ig_steps=ig_steps,
        )
        return [get_intersection(clean_map_np, adv_map) for adv_map in adv_maps]

    objective2_fn.batch = objective2_batch

    return objective2_fn

def compute_explain_map(base_model, model_name, normalize_transform, x_hwc, target_class, device,
                        explain_method="simple_grad", ig_steps=100):
    maps = compute_explain_map_batch(
        base_model=base_model,
        model_name=model_name,
        normalize_transform=normalize_transform,
        x_bhwc=np.expand_dims(np.asarray(x_hwc, dtype=np.float32), axis=0),
        target_class=target_class,
        device=device,
        explain_method=explain_method,
        ig_steps=ig_steps,
    )
    return np.asarray(maps[0], dtype=np.float32)


def compute_explain_map_batch(base_model, model_name, normalize_transform, x_bhwc, target_class, device,
                              explain_method="simple_grad", ig_steps=100):
    normalized_forward = NormalizedForward(base_model, normalize_transform).to(device)
    normalized_forward.eval()

    x_bhwc = np.asarray(x_bhwc, dtype=np.float32)
    x_tensor = torch.from_numpy(x_bhwc).permute(0, 3, 1, 2).to(device)
    target_tensor = torch.full((x_tensor.size(0),), int(target_class), device=device, dtype=torch.long)

    if explain_method == "simple_grad":
        saliency, _ = simple_gradient_map(normalized_forward, x_tensor, target_tensor)
        return saliency.detach().cpu().numpy()

    if explain_method == "integrated_gradients":
        saliency, _ = integrated_gradients(normalized_forward, x_tensor, target_tensor, steps=ig_steps)
        return saliency.detach().cpu().numpy()

    if explain_method == "gradcam":
        x_norm = torch.stack([normalize_transform(xi) for xi in x_tensor], dim=0)
        cam, _ = get_gradcam_map(base_model, model_name, x_norm, target_tensor)
        return np.asarray(cam, dtype=np.float32)

    raise ValueError(f"Unknown explain_method: {explain_method}")


class TorchvisionModelWrapper:
    def __init__(self, model, normalize, device):
        self.model = model
        self.normalize = normalize
        self.device = device

    def predict(self, x):
        x = x.to(self.device)
        x = self.normalize(x)
        with torch.no_grad():
            logits = self.model(x)
        return logits.detach().cpu()


class NormalizedForward(nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, x):
        return self.model(self.normalize(x))


def get_intersection(clean_map, adv_map):
    clean_map = np.asarray(clean_map, dtype=np.float32)
    adv_map = np.asarray(adv_map, dtype=np.float32)
    inter = np.minimum(clean_map, adv_map).sum()
    union = np.maximum(clean_map, adv_map).sum() + 1e-12
    return float(inter / union)