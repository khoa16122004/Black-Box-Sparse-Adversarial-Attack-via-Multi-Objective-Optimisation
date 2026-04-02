import matplotlib.pyplot as plt
from main_GA import compute_explain_map, compute_explain_map_batch
import argparse
import os
import json
import numpy as np
import torch
from PIL import Image
from MOAA.GA_baseline import AttackGA
from LossFunctions import UnTargeted, Targeted
from util import get_torchvision_model
from main_GA import TorchvisionModelWrapper, build_intersection_objective

def iter_samples(runned_data):
    for class_name, values in runned_data.items():
        if isinstance(values, str):
            yield class_name, values
        elif isinstance(values, list):
            for p in values:
                yield class_name, p

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.runned_data_path, "r", encoding="utf-8") as f:
        runned_data = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, spatial_transform, normalize_transform = get_torchvision_model(
        model_name=args.model_name,
        pretrained=True,
    )
    base_model = base_model.to(device)
    base_model.eval()
    model = TorchvisionModelWrapper(base_model, normalize_transform, device)

    samples = iter_samples(runned_data)
    if args.max_samples is not None:
        import itertools
        samples = itertools.islice(samples, args.max_samples)
    for run_id, (class_name, raw_img_path) in enumerate(samples):
        img_name = os.path.splitext(os.path.basename(raw_img_path))[0]
        sample_dir = os.path.join(args.output_dir, class_name, img_name)
        os.makedirs(sample_dir, exist_ok=True)
        print(f"[{run_id}] class={class_name} image={img_name}")
        img = Image.open(raw_img_path).convert("RGB")
        x_tensor = spatial_transform(img)
        x_test = x_tensor.permute(1, 2, 0).numpy()
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
        result_path = os.path.join(sample_dir, "result.npy")
        params = {
            "x": x_test,
            "eps": args.eps,
            "iterations": args.iterations,
            "pc": args.pc,
            "pm": args.pm,
            "pop_size": args.pop_size,
            "zero_probability": args.zero_probability,
            "include_dist": True,
            "max_dist": args.max_dist,
            "p_size": args.p_size,
            "tournament_size": args.tournament_size,
            "save_directory": result_path,
            "verbose": args.verbose,
            "print_every": args.print_every,
            "objective2_fn": objective2_fn,
            "objective3_fn": objective3_fn,
            "lambda_1": args.lambda_1,
            "lambda_2": args.lambda_2,
        }
        attack = AttackGA(params)
        attack.attack(loss)
        # --- Logging and visualization ---
        result = np.load(result_path, allow_pickle=True).item()
        # Save min score of each objective per generation
        fitness_process = np.asarray(result.get("fitness_process", []), dtype=float)
        mins_txt_path = os.path.join(sample_dir, "objective_mins.txt")
        if fitness_process.size > 0:
            if fitness_process.ndim == 1:
                fitness_process = fitness_process[None, :]
            np.savetxt(mins_txt_path, fitness_process, fmt="%.8f")
        else:
            with open(mins_txt_path, "w", encoding="utf-8") as f:
                f.write("")
        # Save Pareto front chart
        def save_pareto_chart(init_front0_fitness, final_front0_fitness, output_path):
            pf_init = np.array(init_front0_fitness, dtype=float)
            pf_final = np.array(final_front0_fitness, dtype=float)
            if pf_init.ndim == 1:
                pf_init = pf_init[None, :]
            if pf_final.ndim == 1:
                pf_final = pf_final[None, :]
            n_obj = pf_init.shape[1] if pf_init.shape[1] > 0 else pf_final.shape[1]
            if n_obj == 2:
                fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                has_any_front = False
                if pf_init.shape[0] > 0:
                    ax.scatter(pf_init[:, 0], pf_init[:, 1], c="red", s=60,
                               edgecolors="darkred", linewidths=0.5, label="Rank0 - Generation 1")
                    has_any_front = True
                if pf_final.shape[0] > 0:
                    ax.scatter(pf_final[:, 0], pf_final[:, 1], c="blue", s=60,
                               edgecolors="navy", linewidths=0.5, label="Rank0 - Final Generation")
                    has_any_front = True
                ax.set_title("Pareto Front Overlay: Gen 1 vs Final")
                ax.set_xlabel("Objective 0 (loss)")
                ax.set_ylabel("Objective 1 (intersection)")
                ax.grid(True, alpha=0.3)
                if has_any_front:
                    ax.legend(loc="best")
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                fig.tight_layout()
                fig.savefig(os.path.join(sample_dir, "pareto_rank0_first_vs_last.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                fig.tight_layout()
                fig.savefig(os.path.join(sample_dir, "pareto_rank0_first_vs_last.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

        def save_rank0_artifacts(sample_dir, result, base_model, model_name, normalize_transform, device,
                                 explain_method, ig_steps, x_clean_hwc):
            rank0_dir = os.path.join(sample_dir, "rank0")
            os.makedirs(rank0_dir, exist_ok=True)
            true_label = int(result.get("true_label", -1))
            front0_imgs = result.get("front0_imgs", [])
            front0_fitness = result.get("front0_fitness", [])
            adversarial_labels = result.get("adversarial_labels", [])
            clean_img = np.clip(np.asarray(x_clean_hwc, dtype=np.float32), 0.0, 1.0)
            clean_map = compute_explain_map(
                base_model=base_model,
                model_name=model_name,
                normalize_transform=normalize_transform,
                x_hwc=clean_img,
                target_class=true_label,
                device=device,
                explain_method=explain_method,
                ig_steps=ig_steps,
            )
            plt.imsave(os.path.join(sample_dir, "clean_image.png"), clean_img)
            plt.imsave(os.path.join(sample_dir, "clean_map.png"), clean_map, cmap="inferno")
            if len(front0_imgs) > 0:
                adv_imgs = np.clip(np.asarray(front0_imgs, dtype=np.float32), 0.0, 1.0)
                adv_maps = compute_explain_map_batch(
                    base_model=base_model,
                    model_name=model_name,
                    normalize_transform=normalize_transform,
                    x_bhwc=adv_imgs,
                    target_class=true_label,
                    device=device,
                    explain_method=explain_method,
                    ig_steps=ig_steps,
                )
            else:
                adv_imgs = np.zeros((0, *clean_img.shape), dtype=np.float32)
                adv_maps = np.zeros((0, clean_map.shape[0], clean_map.shape[1]), dtype=np.float32)
            n_obj = 0
            if len(front0_fitness) > 0:
                fit0 = front0_fitness[0]
                n_obj = len(fit0)
            header = ["idx", "pred_label"] + [f"obj{i}" for i in range(n_obj)]
            with open(os.path.join(sample_dir, "rank0_scores.txt"), "w", encoding="utf-8") as f:
                f.write(" ".join(header) + "\n")
                for idx in range(len(front0_imgs)):
                    fit = front0_fitness[idx] if idx < len(front0_fitness) else [np.nan]*n_obj
                    pred = adversarial_labels[idx] if idx < len(adversarial_labels) else -1
                    line = [str(idx), str(pred)] + [f"{float(fit[i]):.8f}" if i < len(fit) else "nan" for i in range(n_obj)]
                    f.write(" ".join(line) + "\n")
                    plt.imsave(os.path.join(rank0_dir, f"adv_{idx:03d}.png"), adv_imgs[idx])
                    plt.imsave(os.path.join(rank0_dir, f"map_{idx:03d}.png"), adv_maps[idx], cmap="inferno")

        # Save logs and plots
        save_pareto_chart(
            init_front0_fitness=result.get("init_front0_fitness", []),
            final_front0_fitness=result.get("front0_fitness", []),
            output_path=os.path.join(sample_dir, "pareto_rank0_first_vs_last.png"),
        )
        save_rank0_artifacts(
            sample_dir=sample_dir,
            result=result,
            base_model=base_model,
            model_name=args.model_name,
            normalize_transform=normalize_transform,
            device=device,
            explain_method=args.explain_method,
            ig_steps=args.ig_steps,
            x_clean_hwc=x_test,
        )
        print(f"Saved: {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runned_data_path", type=str, default="model_evaluation_results/resnet18.json")
    parser.add_argument("--output_dir", type=str, default="outdir")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--eps", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--pc", type=float, default=0.1)
    parser.add_argument("--pm", type=float, default=0.4)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--zero_probability", type=float, default=0.3)
    parser.add_argument("--max_dist", type=float, default=1e-5)
    parser.add_argument("--p_size", type=float, default=2.0)
    parser.add_argument("--tournament_size", type=int, default=4)
    parser.add_argument("--explain_method", type=str, default="simple_grad", choices=["simple_grad", "integrated_gradients", "gradcam"])
    parser.add_argument("--ig_steps", type=int, default=5)
    parser.add_argument("--lambda_1", type=float, default=0.5)
    parser.add_argument("--lambda_2", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print_every", type=int, default=10)
    main(parser.parse_args())
