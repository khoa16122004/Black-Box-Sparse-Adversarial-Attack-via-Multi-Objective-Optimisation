import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import Optional



def simple_gradient_map(model, input_tensor, target_class=None):

    model.eval()

    x = input_tensor.clone().detach()
    x.requires_grad_(True)

    model.zero_grad()

    output = model(x)
    output_logits = output.detach()

    # choose class per sample
    if target_class is None:
        target_class = output.argmax(dim=1)

    # gather scores for each sample
    score = output.gather(1, target_class.view(-1,1)).sum()

    score.backward()

    grad = x.grad

    # sum RGB
    saliency = grad.abs().sum(dim=1)

    H, W = saliency.shape[-2:]

    # normalize per image
    saliency = (H*W) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1,1,1) + 1e-8)

    return saliency.detach(), output_logits


def integrated_gradients(model, input_tensor, target_class=None, steps=100, baseline=None):

    model.eval()

    x = input_tensor.clone().detach()
    B = x.size(0)

    if baseline is None:
        baseline = torch.zeros_like(x)

    if target_class is None:
        with torch.no_grad():
            target_class = model(x).argmax(dim=1)

    grads = torch.zeros_like(x)

    for i in range(1, steps+1):

        alpha = float(i)/steps
        inp = baseline + alpha * (x - baseline)

        inp.requires_grad_(True)

        model.zero_grad()

        output = model(inp)
        output_logits = output.detach()

        score = output.gather(1, target_class.view(-1,1)).sum()

        score.backward()

        grads += inp.grad.detach()

    avg_grad = grads / steps

    ig = (x - baseline) * avg_grad

    saliency = ig.abs().sum(dim=1)

    H, W = saliency.shape[-2:]

    saliency = (H*W) * saliency / (saliency.view(B,-1).sum(dim=1).view(-1,1,1) + 1e-8)

    return saliency.detach(), output_logits


def _vit_reshape_transform(tensor, height: Optional[int] = None, width: Optional[int] = None):
    """Reshape ViT tokens to feature map with optional manual grid override."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected ViT activations with shape [B, N, C], got {tuple(tensor.shape)}")

    # Drop class token: [B, 1 + HW, C] -> [B, HW, C]
    tensor = tensor[:, 1:, :]
    num_tokens = int(tensor.size(1))

    if height is None and width is None:
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise ValueError(
                f"Cannot infer square token grid from {num_tokens} tokens. "
                "Pass vit_height/vit_width explicitly."
            )
        height = side
        width = side
    elif height is None:
        if width is None or width <= 0 or num_tokens % width != 0:
            raise ValueError(f"Invalid width={width} for {num_tokens} tokens")
        height = num_tokens // width
    elif width is None:
        if height <= 0 or num_tokens % height != 0:
            raise ValueError(f"Invalid height={height} for {num_tokens} tokens")
        width = num_tokens // height

    if height * width != num_tokens:
        raise ValueError(
            f"height*width must equal token count after removing cls token: "
            f"{height}*{width} != {num_tokens}"
        )

    tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return tensor.permute(0, 3, 1, 2)


def get_gradcam_target_layer(model, model_name, vit_height: Optional[int] = None, vit_width: Optional[int] = None):
    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        return [model.layer4[-1]], None

    if model_name.startswith("vgg"):
        return [model.features[-1]], None

    if model_name.startswith("vit"):
        # If not provided, infer from model metadata when available.
        if vit_height is None or vit_width is None:
            image_size = getattr(model, "image_size", None)
            patch_size = getattr(model, "patch_size", None)

            if image_size is not None and patch_size is not None:
                if isinstance(image_size, tuple):
                    image_h, image_w = int(image_size[0]), int(image_size[1])
                else:
                    image_h = image_w = int(image_size)

                if isinstance(patch_size, tuple):
                    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
                else:
                    patch_h = patch_w = int(patch_size)

                if patch_h > 0 and patch_w > 0:
                    inferred_h = image_h // patch_h
                    inferred_w = image_w // patch_w
                    vit_height = inferred_h if vit_height is None else vit_height
                    vit_width = inferred_w if vit_width is None else vit_width

        if vit_height is None and vit_width is None:
            return [model.encoder.layers[-1].ln_1], _vit_reshape_transform

        return [model.encoder.layers[-1].ln_1], lambda tensor: _vit_reshape_transform(
            tensor,
            height=vit_height,
            width=vit_width,
        )
    
    if model_name.startswith("densenet"):
        return [model.features[-1]], None

    raise ValueError(f"Grad-CAM target layer is not configured for model {model_name}")


def get_gradcam_map(
    model,
    model_name,
    input_tensor,
    target_class=None,
    vit_height: Optional[int] = None,
    vit_width: Optional[int] = None,
):
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)


    target_layers, reshape_transform = get_gradcam_target_layer(
        model,
        model_name,
        vit_height=vit_height,
        vit_width=vit_width,
    )
    
    if target_class is None:
        targets = None
    else:
        if isinstance(target_class, torch.Tensor):
            target_values = target_class.detach().view(-1).tolist()
        elif isinstance(target_class, (list, tuple)):
            target_values = list(target_class)
        else:
            target_values = [target_class] * int(input_tensor.size(0))

        if len(target_values) == 1 and input_tensor.size(0) > 1:
            target_values = target_values * int(input_tensor.size(0))

        targets = [ClassifierOutputTarget(int(t)) for t in target_values]



    cam_kwargs = {
        "model": model,
        "target_layers": target_layers,
    }
    if reshape_transform is not None:
        cam_kwargs["reshape_transform"] = reshape_transform

    with GradCAM(**cam_kwargs) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    output_logits = cam.outputs.detach()

    return grayscale_cam, output_logits


