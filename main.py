# from ProposedMethod.QueryEfficient.Scratch import Attack
from MOAA.MOAA import Attack
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


def visualize_attack_result(save_path, x_clean=None, base_model=None, model_name=None, normalize_transform=None,
                            device=None, explain_method="simple_grad", ig_steps=100,
                            output_path=None, show_plot=True):
    result = np.load(save_path, allow_pickle=True).item()

    front0_imgs = result.get("front0_imgs", [])
    front0_fitness = result.get("front0_fitness", [])
    fitness_process = np.array(result.get("fitness_process", []), dtype=float)

    if len(front0_imgs) == 0:
        print("No adversarial candidates found in front0_imgs.")
        return

    if x_clean is not None:
        x_clean = np.clip(x_clean, 0.0, 1.0)

    clean_map = None
    if x_clean is not None and base_model is not None and normalize_transform is not None and device is not None:
        clean_map = compute_explain_map(
            base_model=base_model,
            model_name=model_name,
            normalize_transform=normalize_transform,
            x_hwc=x_clean,
            target_class=result.get("true_label"),
            device=device,
            explain_method=explain_method,
            ig_steps=ig_steps,
        )

    adv_maps = []
    if base_model is not None and normalize_transform is not None and device is not None:
        for x_adv in front0_imgs:
            adv_maps.append(
                compute_explain_map(
                    base_model=base_model,
                    model_name=model_name,
                    normalize_transform=normalize_transform,
                    x_hwc=np.clip(x_adv, 0.0, 1.0),
                    target_class=result.get("true_label"),
                    device=device,
                    explain_method=explain_method,
                    ig_steps=ig_steps,
                )
            )

    n_rank0 = len(front0_imgs)
    fig, axes = plt.subplots(n_rank0 + 3, 2, figsize=(12, 4 * (n_rank0 + 3)))
    if n_rank0 + 3 == 1:
        axes = np.array([axes])

    ax_img_clean, ax_map_clean = axes[0]

    if x_clean is not None:
        ax_img_clean.imshow(x_clean)
        ax_img_clean.set_title("Clean image (after spatial transform)")
    else:
        ax_img_clean.text(0.5, 0.5, "Clean image not provided", ha="center", va="center")
        ax_img_clean.set_title("Clean image")
    ax_img_clean.axis("off")

    if clean_map is not None:
        ax_map_clean.imshow(clean_map, cmap="inferno")
        ax_map_clean.set_title(f"Clean explain map ({explain_method})")
    else:
        ax_map_clean.text(0.5, 0.5, "Explain map not available", ha="center", va="center")
        ax_map_clean.set_title(f"Clean explain map ({explain_method})")
    ax_map_clean.axis("off")

    adversarial_labels = result.get("adversarial_labels", [])
    for idx, x_adv in enumerate(front0_imgs):
        ax_img_adv, ax_map_adv = axes[idx + 1]
        x_adv = np.clip(x_adv, 0.0, 1.0)
        ax_img_adv.imshow(x_adv)
        pred = adversarial_labels[idx] if idx < len(adversarial_labels) else None
        fit = front0_fitness[idx] if idx < len(front0_fitness) else None
        if fit is not None and len(fit) > 1:
            title = f"Rank0 #{idx} | pred={pred} | obj0={float(fit[0]):.4f} | obj1={float(fit[1]):.4f}"
        elif fit is not None and len(fit) > 0:
            title = f"Rank0 #{idx} | pred={pred} | obj0={float(fit[0]):.4f}"
        else:
            title = f"Rank0 #{idx} | pred={pred}"
        ax_img_adv.set_title(title)
        ax_img_adv.axis("off")

        if idx < len(adv_maps):
            ax_map_adv.imshow(adv_maps[idx], cmap="inferno")
            ax_map_adv.set_title(f"Rank0 #{idx} explain map ({explain_method})")
        else:
            ax_map_adv.text(0.5, 0.5, "Explain map not available", ha="center", va="center")
            ax_map_adv.set_title(f"Rank0 #{idx} explain map ({explain_method})")
        ax_map_adv.axis("off")

    ax_curve_loss, ax_curve_dist = axes[-2]
    if fitness_process.size > 0:
        ax_curve_loss.plot(fitness_process[:, 0], label="loss objective")
        ax_curve_loss.set_title("Objective 0 over iterations")
        ax_curve_loss.set_xlabel("iteration")
        ax_curve_loss.set_ylabel("objective 0")
        ax_curve_loss.grid(True, alpha=0.3)

        if fitness_process.shape[1] > 1:
            ax_curve_dist.plot(fitness_process[:, 1], color="tab:orange", label="intersection objective")
            ax_curve_dist.set_ylabel("objective 1")
        else:
            ax_curve_dist.text(0.5, 0.5, "Objective 1 not available", ha="center", va="center")

        ax_curve_dist.set_title("Objective 1 over iterations")
        ax_curve_dist.set_xlabel("iteration")
        ax_curve_dist.grid(True, alpha=0.3)
    else:
        ax_curve_loss.text(0.5, 0.5, "fitness_process is empty", ha="center", va="center")
        ax_curve_dist.text(0.5, 0.5, "fitness_process is empty", ha="center", va="center")
        ax_curve_loss.set_title("Objective 0 over iterations")
        ax_curve_dist.set_title("Objective 1 over iterations")

    # --- Pareto fronts overlaid: generation 1 (red) and final generation (blue) ---
    init_front0_fitness = result.get("init_front0_fitness", [])
    ax_pareto = axes[-1, 0]
    axes[-1, 1].axis("off")

    has_any_front = False
    if len(init_front0_fitness) > 0:
        pf_init = np.array(init_front0_fitness, dtype=float)
        if pf_init.ndim == 1:
            pf_init = pf_init[None, :]
        ax_pareto.scatter(
            pf_init[:, 0], pf_init[:, 1], c="red", s=60, zorder=3,
            edgecolors="darkred", linewidths=0.5, label="Rank0 - Generation 1"
        )
        has_any_front = True

    if len(front0_fitness) > 0:
        pf_final = np.array(front0_fitness, dtype=float)
        if pf_final.ndim == 1:
            pf_final = pf_final[None, :]
        ax_pareto.scatter(
            pf_final[:, 0], pf_final[:, 1], c="blue", s=60, zorder=3,
            edgecolors="navy", linewidths=0.5, label="Rank0 - Final Generation"
        )
        has_any_front = True

    ax_pareto.set_xlabel("Objective 0 (loss)")
    ax_pareto.set_ylabel("Objective 1 (intersection)")
    ax_pareto.set_title("Pareto Front Overlay: Gen 1 vs Final")
    ax_pareto.grid(True, alpha=0.3)

    if has_any_front:
        ax_pareto.legend(loc="best")
    else:
        ax_pareto.text(0.5, 0.5, "No data", ha="center", va="center")
    meta = (
        f"success={result.get('success')} | queries={result.get('queries')} | "
        f"true={result.get('true_label')} | front0={len(front0_imgs)}"
    )
    fig.suptitle(meta)
    fig.tight_layout()

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    """
    Non-Targeted
    pc = 0.1
    pm = 0.4
    
    Targeted:
    pc = 0.1
    pm = 0.2
    """
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
                        help="print progress in MOAA main loop")
    parser.add_argument("--print_every", type=int, default=10,
                        help="print metrics every N iterations when --verbose is set")
    parser.add_argument("--eps", type=int, default=24,
                        help="number of pixels allowed to change")
    parser.add_argument("--explain_method", type=str, default="simple_grad",
                        choices=["simple_grad", "integrated_gradients", "gradcam"],
                        help="which explain map method to use")
    parser.add_argument("--ig_steps", type=int, default=5,
                        help="number of steps for integrated gradients")
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

    save_dir = os.path.dirname(args.save_directory)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    params = {
        "x": x_test, # Image is assume to be numpy array of shape height * width * 3
        "eps": args.eps, # number of changed pixels
        "iterations": 1000 // 2, # model query budget / population size
        "pc": pc, # crossover parameter
        "pm": pm, # mutation parameter
        "pop_size": 50, # population size
        "zero_probability": 0.3,
        "include_dist": True, # objective-2 is intersection score (minimize)
        "max_dist": 1e-5, # threshold on intersection objective to declare early success
        "p_size": 2.0, # Perturbation values have {-p_size, p_size, 0}. Change this if you want smaller perturbations.
        "tournament_size": 4, #Number of parents compared to generate new solutions, cannot be larger than the population
        "save_directory": args.save_directory,
        "verbose": args.verbose,
        "print_every": args.print_every,
        "objective2_fn": objective2_fn,
    }
    attack = Attack(params)
    attack.attack(loss)

    if args.visualize:
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
