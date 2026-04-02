

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import ttest_ind


def spear_rank_correlation_cal(saliency_map, perturbation_map):
    saliency_flat = saliency_map.flatten()
    perturbation_flat = perturbation_map.flatten()
    correlation = np.corrcoef(saliency_flat, perturbation_flat)[0, 1]
    return correlation


def parse_rank_file(rank_path):
    rows = []
    with open(rank_path, "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            idx = parts[0]
            pred_label = int(parts[1])
            obj1 = float(parts[2])
            obj2 = float(parts[3])
            obj3 = float(parts[4]) if len(parts) > 4 else None
            rows.append({"idx": idx, "pred_label": pred_label, "obj1": obj1, "obj2": obj2, "obj3": obj3})
    return rows


def find_best_candidate(rows):
    valid = [r for r in rows if r["obj1"] < 0]

    if rows[0]["obj3"] is None:
        if valid:
            best = min(valid, key=lambda r: r["obj2"])
        else:
            best = min(rows, key=lambda r: r["obj1"])
    else:
        if valid:
            k = max(1, int(len(valid) * 0.1))
            top_k = sorted(valid, key=lambda r: r["obj2"])[:k]
            best = min(top_k, key=lambda r: r["obj3"])
        else:
            best = min(rows, key=lambda r: r["obj1"])

    retain_class = int(any(r["obj1"] < 0 for r in rows))
    return {**best, "retain_class": retain_class}


def collect_best_candidates(result_dir):
    best_candidates = []
    if not os.path.isdir(result_dir):
        return best_candidates

    for class_name in sorted(os.listdir(result_dir)):
        class_path = os.path.join(result_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for sample_name in sorted(os.listdir(class_path)):
            sample_path = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_path):
                continue

            rank_path = os.path.join(sample_path, "rank0_scores.txt")
            original_img_path = os.path.join(sample_path, "clean_image.png")
            original_map_path = os.path.join(sample_path, "clean_map.png")
            if not os.path.exists(rank_path):
                continue

            rows = parse_rank_file(rank_path)
            if not rows:
                continue

            best = find_best_candidate(rows)
            best_rank0_id = int(best["idx"])
            best_rank0_advimg = os.path.join(sample_path, "rank0", f"adv_{best_rank0_id:03d}.png")
            best_rank0_advmap = os.path.join(sample_path, "rank0", f"map_{best_rank0_id:03d}.png")

            required_files = [original_img_path, original_map_path, best_rank0_advimg, best_rank0_advmap]
            if not all(os.path.exists(p) for p in required_files):
                continue

            saliency_map = np.array(Image.open(original_map_path).convert("L"))
            perturbation_map = np.array(Image.open(best_rank0_advmap).convert("L"))
            best["spearman_corr"] = spear_rank_correlation_cal(saliency_map, perturbation_map)

            original_img = np.array(Image.open(original_img_path))
            best_adv_img = np.array(Image.open(best_rank0_advimg))
            best["l0"] = int(np.sum(np.any(original_img != best_adv_img, axis=-1)))

            best_candidates.append(best)

    return best_candidates


def summarize_values(values):
    if len(values) == 0:
        return {"mean": 0.0, "min": 0.0, "std": 0.0}

    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "std": float(np.std(values)),
    }


def compute_statistics(best_candidates):
    metric_values = {
        "obj1": np.array([b["obj1"] for b in best_candidates], dtype=float),
        "obj2": np.array([b["obj2"] for b in best_candidates], dtype=float),
        "l0": np.array([b["l0"] for b in best_candidates], dtype=float),
        "retain_class": np.array([b["retain_class"] for b in best_candidates], dtype=float),
        "spearman_corr": np.array([b["spearman_corr"] for b in best_candidates], dtype=float),
    }
    stats = {metric: summarize_values(values) for metric, values in metric_values.items()}
    return stats, metric_values


def print_statistics(label, num_samples, stats):
    print(f"\n===== {label} =====")
    print(f"Num run sample: {num_samples}")
    print(
        f"obj1 -> mean: {stats['obj1']['mean']}, min: {stats['obj1']['min']}, std: {stats['obj1']['std']}"
    )
    print(
        f"obj2 -> mean: {stats['obj2']['mean']}, min: {stats['obj2']['min']}, std: {stats['obj2']['std']}"
    )
    print(f"l0 -> mean: {stats['l0']['mean']}, min: {stats['l0']['min']}, std: {stats['l0']['std']}")
    print(
        "Retain class rate -> "
        f"mean: {stats['retain_class']['mean']}, min: {stats['retain_class']['min']}, std: {stats['retain_class']['std']}"
    )
    print(
        "Spearman correlation -> "
        f"mean: {stats['spearman_corr']['mean']}, min: {stats['spearman_corr']['min']}, std: {stats['spearman_corr']['std']}"
    )


def run_ttest_against_best_per_metric(all_metric_values, alpha=0.05):
    metric_names = ["obj1", "obj2", "l0", "retain_class", "spearman_corr"]

    print(f"\n===== Welch t-test vs best epsilon per metric (lowest mean, alpha={alpha}) =====")
    for metric in metric_names:
        available = []
        for eps, metric_values in all_metric_values.items():
            values = metric_values[metric]
            if len(values) == 0:
                continue
            available.append((eps, values, float(np.mean(values))))

        if len(available) < 2:
            print(f"- {metric}: skip (need at least 2 epsilons with non-empty values)")
            continue

        best_eps, best_values, best_mean = min(available, key=lambda x: x[2])
        print(f"- {metric}: best epsilon={best_eps} (mean={best_mean:.6g})")

        for eps, values, mean_val in available:
            if eps == best_eps:
                continue
            if len(best_values) < 2 or len(values) < 2:
                print(f"  compare eps={eps}: skip (need at least 2 samples per group)")
                continue

            _, pvalue = ttest_ind(best_values, values, equal_var=False)
            significant = bool((not np.isnan(pvalue)) and (pvalue < alpha))
            print(
                f"  compare eps={eps} (mean={mean_val:.6g}) vs best={best_eps}: "
                f"p-value={pvalue:.6g}, significant={significant}"
            )


def collect_iteration_min_curves(result_dir):
    obj1_curves = []
    obj2_curves = []
    if not os.path.isdir(result_dir):
        return np.array([]), np.array([]), 0

    for class_name in sorted(os.listdir(result_dir)):
        class_path = os.path.join(result_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for sample_name in sorted(os.listdir(class_path)):
            sample_path = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_path):
                continue

            objective_mins_path = os.path.join(sample_path, "objective_mins.txt")
            if not os.path.exists(objective_mins_path):
                continue

            values = np.loadtxt(objective_mins_path)
            values = np.atleast_2d(values)
            if values.shape[1] < 2:
                continue

            obj1_curve = np.minimum.accumulate(values[:, 0])
            obj2_curve = np.minimum.accumulate(values[:, 1])
            obj1_curves.append(obj1_curve)
            obj2_curves.append(obj2_curve)

    if len(obj1_curves) == 0:
        return np.array([]), np.array([]), 0

    min_len = min(len(curve) for curve in obj1_curves)
    stacked_obj1 = np.stack([curve[:min_len] for curve in obj1_curves], axis=0)
    stacked_obj2 = np.stack([curve[:min_len] for curve in obj2_curves], axis=0)
    return np.mean(stacked_obj1, axis=0), np.mean(stacked_obj2, axis=0), len(obj1_curves)


def save_iteration_curves_csv(epsilon_curve_data, output_csv_path):
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epsilon", "iteration", "mean_min_obj1", "mean_min_obj2"])
        for eps_token, curve_data in epsilon_curve_data.items():
            obj1_curve = curve_data["obj1_curve"]
            obj2_curve = curve_data["obj2_curve"]
            for iteration_idx, (obj1_value, obj2_value) in enumerate(zip(obj1_curve, obj2_curve), start=1):
                writer.writerow([eps_token, iteration_idx, float(obj1_value), float(obj2_value)])


def plot_min_curves(epsilon_curve_data, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for eps_token, curve_data in epsilon_curve_data.items():
        obj1_curve = curve_data["obj1_curve"]
        obj2_curve = curve_data["obj2_curve"]
        sample_count = curve_data["num_samples"]
        iterations = np.arange(1, len(obj1_curve) + 1)
        label = rf"$\epsilon={eps_token}$ (n={sample_count})"
        axes[0].plot(iterations, obj1_curve, linewidth=2, label=label)
        axes[1].plot(iterations, obj2_curve, linewidth=2, label=label)

    axes[0].set_title(r"$\mathrm{Mean\ cumulative\ min}\ \mathcal{L}_{\mathrm{confidence}}$")
    axes[0].set_xlabel(r"Iteration $t$")
    axes[0].set_ylabel("Mean min $\mathcal{L}_\text{confidence}$")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(r"$\mathrm{Mean\ cumulative\ min}\ \mathcal{L}_{\mathrm{saliency}}$")
    axes[1].set_xlabel(r"Iteration")
    axes[1].set_ylabel("Mean min $\mathcal{L}_\text{saliency}$")
    axes[1].grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(1, min(5, len(labels))),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_epsilons(epsilons_arg):
    eps_tokens = [x.strip() for x in epsilons_arg.split(",") if x.strip()]
    eps_values = [float(x) for x in eps_tokens]
    return eps_tokens, eps_values


def process_single_result_dir(result_dir):
    best_candidates = collect_best_candidates(result_dir)
    stats, _ = compute_statistics(best_candidates)
    print_statistics(label=result_dir, num_samples=len(best_candidates), stats=stats)


def process_multi_epsilon(result_dir_template, epsilons_arg, alpha, plot_name):
    eps_tokens, _ = parse_epsilons(epsilons_arg)
    all_metric_values = {}
    epsilon_curve_data = {}

    for eps_token in eps_tokens:
        result_dir = result_dir_template.format(epsilon=eps_token)
        best_candidates = collect_best_candidates(result_dir)
        stats, metric_values = compute_statistics(best_candidates)
        print_statistics(label=f"epsilon={eps_token}", num_samples=len(best_candidates), stats=stats)
        all_metric_values[eps_token] = metric_values

        obj1_curve, obj2_curve, num_curve_samples = collect_iteration_min_curves(result_dir)
        if len(obj1_curve) == 0:
            print(f"No iteration curve data found for epsilon={eps_token} (missing objective_mins.txt)")
            continue

        epsilon_curve_data[eps_token] = {
            "obj1_curve": obj1_curve,
            "obj2_curve": obj2_curve,
            "num_samples": num_curve_samples,
        }

    run_ttest_against_best_per_metric(all_metric_values, alpha=alpha)

    output_dir = os.path.dirname(result_dir_template.split("{epsilon}")[0].rstrip("/"))
    if output_dir == "":
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, plot_name)

    if len(epsilon_curve_data) == 0:
        print("\nNo curve figure was saved because no objective_mins.txt data was found.")
        return

    plot_min_curves(epsilon_curve_data, output_path)
    output_csv_path = os.path.join(output_dir, "epsilon_iteration_curves.csv")
    save_iteration_curves_csv(epsilon_curve_data, output_csv_path)
    print(f"\nSaved min-curves figure to: {output_path}")
    print(f"Saved iteration-curve data to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process result folder and compute statistics.")
    parser.add_argument(
        "--result_dir",
        required=True,
        help="Path to folder containing sample subfolders. Use {epsilon} in path for multi-epsilon mode.",
    )
    parser.add_argument(
        "--epsilons",
        default="0.0,0.2,0.5,0.8,1.0",
        help="Comma-separated epsilons for multi-epsilon mode.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for t-test.")
    parser.add_argument("--plot_name", default="epsilon_min_curves.png", help="Output plot filename.")

    args = parser.parse_args()
    if "{epsilon}" in args.result_dir:
        process_multi_epsilon(
            result_dir_template=args.result_dir,
            epsilons_arg=args.epsilons,
            alpha=args.alpha,
            plot_name=args.plot_name,
        )
    else:
        process_single_result_dir(args.result_dir)


