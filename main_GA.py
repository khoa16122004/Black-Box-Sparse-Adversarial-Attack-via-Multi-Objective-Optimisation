from MOAA.GA_baseline import AttackGA
from LossFunctions import UnTargeted, Targeted
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from util import get_torchvision_model
from MOAA.explain_method import simple_gradient_map, integrated_gradients, get_gradcam_map

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


def save_weighted_best_outputs(result_path, base_model, model_name, normalize_transform, device, explain_method, ig_steps):
    if not os.path.exists(result_path):
        return

    result = np.load(result_path, allow_pickle=True).item()
    out_dir = os.path.dirname(result_path) if os.path.dirname(result_path) else "."

    weighted_best = result.get("weighted_best_final", None)
    if isinstance(weighted_best, dict):
        best_img = weighted_best.get("image", None)
        if best_img is not None:
            best_img = np.clip(np.asarray(best_img, dtype=np.float32), 0.0, 1.0)
            plt.imsave(os.path.join(out_dir, "weighted_best_image.png"), best_img)
            true_label = int(result.get("true_label", -1))
            if true_label >= 0:
                best_map = compute_explain_map(
                    base_model=base_model,
                    model_name=model_name,
                    normalize_transform=normalize_transform,
                    x_hwc=best_img,
                    target_class=true_label,
                    device=device,
                    explain_method=explain_method,
                    ig_steps=ig_steps,
                )
                plt.imsave(os.path.join(out_dir, "weighted_best_map.png"), best_map, cmap="hot")

        fitnesses = weighted_best.get("fitnesses", [])
        fitness_score = weighted_best.get("fitness_score", np.nan)
        with open(os.path.join(out_dir, "weighted_best_score.txt"), "w", encoding="utf-8") as f:
            if len(fitnesses) >= 2:
                f.write(f"obj0 {float(fitnesses[0]):.8f}\n")
                f.write(f"obj1 {float(fitnesses[1]):.8f}\n")
            f.write(f"weighted_sum {float(fitness_score):.8f}\n")

    weighted_best_process = np.asarray(result.get("weighted_best_process", []), dtype=float)
    process_path = os.path.join(out_dir, "weighted_best_process.txt")
    if weighted_best_process.size > 0:
        weighted_best_process = np.atleast_2d(weighted_best_process)
        np.savetxt(process_path, weighted_best_process, fmt="%.8f")
    else:
        with open(process_path, "w", encoding="utf-8") as f:
            f.write("")

    process_records = result.get("weighted_best_process_records", [])
    process_dir = os.path.join(out_dir, "weighted_best_process")
    os.makedirs(process_dir, exist_ok=True)
    metrics_path = os.path.join(process_dir, "weighted_best_process_scores.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("iter obj0 obj1 weighted_sum pred_label is_adversarial\n")
        for idx, record in enumerate(process_records):
            if not isinstance(record, dict):
                continue
            fitnesses = np.asarray(record.get("fitnesses", []), dtype=float)
            obj0 = float(fitnesses[0]) if fitnesses.size > 0 else np.nan
            obj1 = float(fitnesses[1]) if fitnesses.size > 1 else np.nan
            weighted_sum = float(record.get("fitness_score", np.nan))
            pred_label = int(record.get("pred_label", -1))
            is_adv = int(bool(record.get("is_adversarial", False)))
            f.write(f"{idx} {obj0:.8f} {obj1:.8f} {weighted_sum:.8f} {pred_label} {is_adv}\n")

if __name__ == "__main__":
    np.random.seed(0)
    pc = 0.1
    pm = 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="resnet50", type=str,
                        help="torchvision model name, e.g. resnet50")
    parser.add_argument("--image_path", default="test_imgs/cat.png", type=str,
                        help="path to a single image")
    parser.add_argument("--save_directory", default="attack_result.npy", type=str)
    parser.add_argument("--target", type=int, default=None,
                        help="target class for targeted attack; omit for untargeted")
    parser.add_argument("--visualize", action="store_true",
                        help="visualize attack result after run")
    parser.add_argument("--visualize_only", action="store_true",
                        help="only read saved result and visualize")
    parser.add_argument("--viz_output", type=str, default=None,
                        help="optional path to save visualization image")
    parser.add_argument("--no_show", action="store_true",
                        help="do not call plt.show(); useful on headless machines")
    parser.add_argument("--verbose", action="store_true",
                        help="print progress in GA main loop")
    parser.add_argument("--print_every", type=int, default=10,
                        help="print metrics every N iterations when --verbose is set")
    parser.add_argument("--eps", type=int, default=24,
                        help="number of pixels allowed to change")
    parser.add_argument("--explain_method", type=str, default="simple_grad",
                        choices=["simple_grad", "integrated_gradients", "gradcam"],
                        help="which explain map method to use")
    parser.add_argument("--ig_steps", type=int, default=5,
                        help="number of steps for integrated gradients")
    parser.add_argument("--lambda_1", type=float, default=0.5, help="weight for objective 1 (loss)")
    parser.add_argument("--lambda_2", type=float, default=0.5, help="weight for objective 2 (intersection)")
    parser.add_argument("--tournament_size", type=int, default=4, help="tournament size for GA selection")
    parser.add_argument("--pop_size", type=int, default=50, help="population size for GA")
    parser.add_argument("--iterations", type=int, default=500, help="number of generations/iterations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, spatial_transform, normalize_transform = get_torchvision_model(
        model_name=args.model_name,
        pretrained=True,
    )
    base_model = base_model.to(device)
    base_model.eval()
    model = TorchvisionModelWrapper(base_model, normalize_transform, device)

    img = Image.open(args.image_path).convert("RGB")
    x_tensor = spatial_transform(img)
    x_test = x_tensor.permute(1, 2, 0).numpy()

    if args.visualize_only:
        from main import visualize_attack_result
        visualize_attack_result(
            save_path=args.save_directory,
            x_clean=x_test,
            base_model=base_model,
            model_name=args.model_name,
            normalize_transform=normalize_transform,
            device=device,
            explain_method=args.explain_method,
            ig_steps=args.ig_steps,
            output_path=args.viz_output,
            show_plot=not args.no_show,
        )
        raise SystemExit(0)

    clean_logits = model.predict(x_tensor.unsqueeze(0)).flatten()
    y_true = int(torch.argmax(clean_logits))

    if args.target is not None:
        loss = Targeted(model, y_true, args.target, to_pytorch=True)
    else:
        loss = UnTargeted(model, y_true, to_pytorch=True)

    objective2_fn = build_intersection_objective(
        base_model=base_model,
        model_name=args.model_name,
        normalize_transform=normalize_transform,
        x_clean_hwc=x_test,
        target_class=y_true,
        device=device,
        explain_method=args.explain_method,
        ig_steps=args.ig_steps,
    )

    def objective3_fn(x_adv_hwc):
        return np.linalg.norm(x_adv_hwc - x_test)
    def objective3_batch(x_adv_bhwc):
        return [np.linalg.norm(x_adv - x_test) for x_adv in x_adv_bhwc]
    objective3_fn.batch = objective3_batch

    save_dir = os.path.dirname(args.save_directory)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    params = {
        "x": x_test,
        "eps": args.eps,
        "iterations": args.iterations,
        "pc": pc,
        "pm": pm,
        "pop_size": args.pop_size,
        "zero_probability": 0.3,
        "include_dist": True,
        "max_dist": 1e-5,
        "p_size": 2.0,
        "tournament_size": args.tournament_size,
        "save_directory": args.save_directory,
        "verbose": args.verbose,
        "print_every": args.print_every,
        "objective2_fn": objective2_fn,
        "objective3_fn": objective3_fn,
        "lambda_1": args.lambda_1,
        "lambda_2": args.lambda_2,
    }

    attack = AttackGA(params)
    attack.attack(loss)
    save_weighted_best_outputs(
        result_path=args.save_directory,
        base_model=base_model,
        model_name=args.model_name,
        normalize_transform=normalize_transform,
        device=device,
        explain_method=args.explain_method,
        ig_steps=args.ig_steps,
    )

    if args.visualize:
        from main import visualize_attack_result
        visualize_attack_result(
            save_path=args.save_directory,
            x_clean=x_test,
            base_model=base_model,
            model_name=args.model_name,
            normalize_transform=normalize_transform,
            device=device,
            explain_method=args.explain_method,
            ig_steps=args.ig_steps,
            output_path=args.viz_output,
            show_plot=not args.no_show,
        )
