import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def discover_pairs_in_folder(folder: Path) -> List[Tuple[Path, Path]]:
    """Find sample folders under a single method folder that contain both result files."""
    pairs: List[Tuple[Path, Path]] = []
    for objective_path in folder.rglob("objective_mins.txt"):
        rank_path = objective_path.with_name("rank0_scores.txt")
        if rank_path.exists():
            pairs.append((objective_path, rank_path))
    return pairs


def discover_pairs_by_method(roots: List[Path]) -> Dict[str, List[Tuple[Path, Path]]]:
    """Find sample folders that contain both objective_mins and rank0_scores, grouped by method."""
    method_to_pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    for root in roots:
        if not root.exists():
            continue
        for objective_path in root.rglob("objective_mins.txt"):
            rank_path = objective_path.with_name("rank0_scores.txt")
            if rank_path.exists():
                relative_parent = objective_path.relative_to(root)
                parts = relative_parent.parts
                method_name = parts[0] if len(parts) > 1 else root.name
                method_to_pairs.setdefault(method_name, []).append((objective_path, rank_path))
    return method_to_pairs


def infer_model_and_method(method_folder: Path) -> Tuple[str, str, str]:
    """Infer model name, explain method, and summary key from output/<model>/<method>."""
    if not method_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {method_folder}")
    if not method_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {method_folder}")

    model_name = method_folder.parent.name
    explain_method = method_folder.name
    if not model_name:
        raise ValueError(
            "Could not infer model name from input folder. Expected path like output/<model>/<method>."
        )

    return model_name, explain_method, f"{model_name}_{explain_method}"


def read_objective_mins(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def read_rank0_scores(path: Path) -> np.ndarray:
    # Columns in repo are: idx pred_label obj0 obj1
    data = np.genfromtxt(path, skip_header=1)
    if data.size == 0:
        return np.empty((0, 4), dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def compute_mean_curve(curves: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    min_len = min(curve.shape[0] for curve in curves)
    num_scores = curves[0].shape[1]
    aligned = np.zeros((len(curves), min_len, num_scores), dtype=float)

    for i, curve in enumerate(curves):
        aligned[i] = curve[:min_len]

    mean_curve = aligned.mean(axis=0)
    return mean_curve, min_len


def smooth_curve(curve: np.ndarray, window: int, passes: int = 1) -> np.ndarray:
    """Apply moving average smoothing to each score column."""
    if window <= 1:
        return curve.copy()

    smoothed = curve.copy()
    kernel = np.ones(window, dtype=float) / window
    for _ in range(max(1, passes)):
        next_smoothed = np.zeros_like(smoothed)
        for col in range(curve.shape[1]):
            padded = np.pad(
                smoothed[:, col],
                (window // 2, window - 1 - window // 2),
                mode="edge",
            )
            next_smoothed[:, col] = np.convolve(padded, kernel, mode="valid")
        smoothed = next_smoothed
    return smoothed


def build_downward_trend(mean_curve: np.ndarray, smooth_window: int) -> np.ndarray:
    """Create a smooth, non-increasing trend curve to highlight minimization progress."""
    first_pass_window = max(3, smooth_window)
    second_pass_window = max(3, smooth_window // 2)

    smooth_first = smooth_curve(mean_curve, first_pass_window, passes=2)
    trend = np.zeros_like(smooth_first)
    for col in range(smooth_first.shape[1]):
        trend[:, col] = np.minimum.accumulate(smooth_first[:, col])

    smooth_trend = smooth_curve(trend, second_pass_window, passes=2)
    for col in range(smooth_trend.shape[1]):
        smooth_trend[:, col] = np.minimum.accumulate(smooth_trend[:, col])

    return smooth_trend


def save_mean_curve_csv(mean_curve: np.ndarray, output_path: Path) -> None:
    num_scores = mean_curve.shape[1]
    header = ["iteration"] + [f"mean_score{j}" for j in range(num_scores)]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(mean_curve.shape[0]):
            row = [i] + [float(mean_curve[i, j]) for j in range(num_scores)]
            writer.writerow(row)


def plot_score_curves(
    curves: List[np.ndarray],
    mean_curve: np.ndarray,
    smoothed_mean_curve: np.ndarray,
    downward_trend_curve: np.ndarray,
    output_dir: Path,
) -> None:
    num_scores = mean_curve.shape[1]

    for score_idx in range(num_scores):
        plt.figure(figsize=(10, 5))
        for curve in curves:
            plt.plot(curve[:, score_idx], color="tab:gray", alpha=0.15, linewidth=1)

        plt.plot(
            mean_curve[:, score_idx],
            color="tab:blue",
            alpha=0.35,
            linewidth=1.5,
            label=f"Raw mean score{score_idx}",
        )
        plt.plot(
            smoothed_mean_curve[:, score_idx],
            color="tab:orange",
            alpha=0.45,
            linewidth=1.6,
            label=f"Smoothed mean score{score_idx}",
        )
        plt.plot(
            downward_trend_curve[:, score_idx],
            color="tab:red",
            linewidth=2.8,
            label=f"Downward trend score{score_idx}",
        )

        ymin = np.min(downward_trend_curve[:, score_idx])
        ymax = np.max(downward_trend_curve[:, score_idx])
        span = ymax - ymin
        margin = max(span * 0.08, 1e-6)
        plt.ylim(ymin - margin, ymax + margin)
        plt.xlabel("Iteration")
        plt.ylabel(f"Score {score_idx}")
        plt.title(f"Score {score_idx} curve across samples")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"score{score_idx}_curve.png", dpi=150)
        plt.close()


def plot_combined_methods(
    method_to_downward_curve: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    if not method_to_downward_curve:
        return

    min_len = min(curve.shape[0] for curve in method_to_downward_curve.values())
    num_scores = next(iter(method_to_downward_curve.values())).shape[1]

    for score_idx in range(num_scores):
        plt.figure(figsize=(11, 5))
        for method_name in sorted(method_to_downward_curve.keys()):
            curve = method_to_downward_curve[method_name][:min_len]
            plt.plot(curve[:, score_idx], linewidth=2.5, label=method_name)

        all_values = np.concatenate(
            [
                method_to_downward_curve[method_name][:min_len, score_idx]
                for method_name in sorted(method_to_downward_curve.keys())
            ]
        )
        ymin = float(np.min(all_values))
        ymax = float(np.max(all_values))
        span = ymax - ymin
        margin = max(span * 0.08, 1e-6)
        plt.ylim(ymin - margin, ymax + margin)

        plt.xlabel("Iteration")
        plt.ylabel(f"Score {score_idx}")
        plt.title(f"Downward trend score{score_idx}: method comparison")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"combined_methods_score{score_idx}.png", dpi=150)
        plt.close()


def choose_best_rank0_candidate(rank_data: np.ndarray) -> Optional[Dict[str, float]]:
    if rank_data.shape[0] == 0:
        return None

    mask = rank_data[:, 2] < 0
    valid = rank_data[mask]
    if valid.shape[0] == 0:
        return None

    best_idx = np.argmin(valid[:, 3])
    best_row = valid[best_idx]
    return {
        "idx": float(best_row[0]),
        "pred_label": float(best_row[1]),
        "obj0": float(best_row[2]),
        "obj1": float(best_row[3]),
    }


def build_mean_best_candidate(sample_best_rows: List[Dict[str, object]]) -> Optional[Dict[str, float]]:
    if not sample_best_rows:
        return None

    obj0_values = np.array([float(row["obj0"]) for row in sample_best_rows], dtype=float)
    obj1_values = np.array([float(row["obj1"]) for row in sample_best_rows], dtype=float)
    return {
        "mean_obj0": float(obj0_values.mean()),
        "mean_obj1": float(obj1_values.mean()),
        "num_samples_used": int(len(sample_best_rows)),
    }


def build_method_summary(
    method_name: str,
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    roots: List[Path],
    smooth_window: int,
) -> Dict[str, object]:
    curves: List[np.ndarray] = []
    sample_best_rows: List[Dict[str, object]] = []

    for objective_path, rank_path in pairs:
        objective_curve = read_objective_mins(objective_path)
        curves.append(objective_curve)

        rank_data = read_rank0_scores(rank_path)
        best = choose_best_rank0_candidate(rank_data)
        if best is not None:
            sample_best_rows.append(
                {
                    "sample_dir": str(objective_path.parent),
                    "objective_path": str(objective_path),
                    "rank0_path": str(rank_path),
                    **best,
                }
            )

    mean_curve, used_iterations = compute_mean_curve(curves)
    smoothed_mean_curve = smooth_curve(mean_curve, smooth_window, passes=3)
    downward_trend_curve = build_downward_trend(mean_curve, smooth_window)

    method_output_dir = output_dir / method_name
    method_output_dir.mkdir(parents=True, exist_ok=True)
    save_mean_curve_csv(mean_curve, method_output_dir / "mean_curve.csv")
    save_mean_curve_csv(smoothed_mean_curve, method_output_dir / "mean_curve_smoothed.csv")
    save_mean_curve_csv(downward_trend_curve, method_output_dir / "mean_curve_downward_trend.csv")
    np.savetxt(method_output_dir / "mean_curve.txt", mean_curve, fmt="%.8f")
    np.savetxt(method_output_dir / "mean_curve_smoothed.txt", smoothed_mean_curve, fmt="%.8f")
    np.savetxt(method_output_dir / "mean_curve_downward_trend.txt", downward_trend_curve, fmt="%.8f")
    plot_score_curves(
        [c[:used_iterations] for c in curves],
        mean_curve,
        smoothed_mean_curve,
        downward_trend_curve,
        method_output_dir,
    )

    overall_best = build_mean_best_candidate(sample_best_rows)
    best_single_candidate = None
    if sample_best_rows:
        best_single_candidate = min(sample_best_rows, key=lambda x: x["obj1"])

    summary = {
        "method": method_name,
        "roots": [str(r) for r in roots],
        "num_samples": len(pairs),
        "num_samples_with_obj0_lt_0_in_rank0": len(sample_best_rows),
        "used_iterations_for_mean": used_iterations,
        "smoothing_window": smooth_window,
        "num_scores": int(mean_curve.shape[1]),
        "overall_best": overall_best,
        "best_single_candidate": best_single_candidate,
    }

    with (method_output_dir / "sample_best_under_obj0_lt_0.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_dir",
                "objective_path",
                "rank0_path",
                "idx",
                "pred_label",
                "obj0",
                "obj1",
            ],
        )
        writer.writeheader()
        for row in sample_best_rows:
            writer.writerow(row)

    with (method_output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{method_name}] Found {len(pairs)} samples with objective/rank files.")
    print(f"[{method_name}] Mean curve saved to: {method_output_dir / 'mean_curve.csv'}")
    print(
        f"[{method_name}] Smoothed mean curve saved to: "
        f"{method_output_dir / 'mean_curve_smoothed.csv'}"
    )
    print(
        f"[{method_name}] Downward trend curve saved to: "
        f"{method_output_dir / 'mean_curve_downward_trend.csv'}"
    )
    print(f"[{method_name}] Curves saved to: {method_output_dir}")
    if overall_best is None:
        print(f"[{method_name}] No valid candidate found where obj0 < 0 in rank0_scores.")
    else:
        print(f"[{method_name}] Mean scores across per-sample best candidates:")
        print(json.dumps(overall_best, indent=2))
        print(f"[{method_name}] Best single candidate (obj0 < 0 and lowest obj1):")
        print(json.dumps(best_single_candidate, indent=2))

    summary["_downward_trend_curve"] = downward_trend_curve
    return summary


def build_summary(roots: List[Path], output_dir: Path, smooth_window: int) -> None:
    pairs_by_method = discover_pairs_by_method(roots)
    if not pairs_by_method:
        raise FileNotFoundError(
            "No objective_mins.txt + rank0_scores.txt pairs found in provided roots."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_method_summaries: List[Dict[str, object]] = []
    method_to_downward_curve: Dict[str, np.ndarray] = {}

    for method_name in sorted(pairs_by_method.keys()):
        pairs = pairs_by_method[method_name]
        if not pairs:
            continue
        summary = build_method_summary(method_name, pairs, output_dir, roots, smooth_window)
        method_to_downward_curve[method_name] = summary.pop("_downward_trend_curve")
        all_method_summaries.append(summary)

    plot_combined_methods(method_to_downward_curve, output_dir)

    with (output_dir / "summary_all_methods.json").open("w", encoding="utf-8") as f:
        json.dump(all_method_summaries, f, indent=2)


def build_summary_for_input_folder(
    input_folder: Path,
    output_dir: Path,
    smooth_window: int,
) -> None:
    model_name, explain_method, method_name = infer_model_and_method(input_folder)
    pairs = discover_pairs_in_folder(input_folder)
    if not pairs:
        raise FileNotFoundError(
            f"No objective_mins.txt + rank0_scores.txt pairs found in input folder: {input_folder}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_method_summary(method_name, pairs, output_dir, [input_folder], smooth_window)
    summary["model_name"] = model_name
    summary["explain_method"] = explain_method
    summary["input_folder"] = str(input_folder)
    summary.pop("_downward_trend_curve", None)

    with (output_dir / "summary_all_methods.json").open("w", encoding="utf-8") as f:
        json.dump([summary], f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate objective_mins and rank0_scores, plot score curves, "
            "compute mean per iteration, and select best rank0 candidate."
        )
    )
    parser.add_argument(
        "--input-folder",
        help=(
            "Single explain-method folder to process, for example output/vgg16/integrated_gradients. "
            "The script will infer model name and explain method from the path."
        ),
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["output"],
        help="Root folders to search recursively when --input-folder is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write plots and summary files. If omitted and --input-folder is used, "
            "defaults to result_process_<model>. Otherwise defaults to processed_results."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Moving-average window size for smoothing mean curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_folder:
        input_folder = Path(args.input_folder)
        model_name, _, _ = infer_model_and_method(input_folder)
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"result_process_{model_name}")
        build_summary_for_input_folder(input_folder, output_dir, args.smooth_window)
        return

    roots = [Path(p) for p in args.roots]
    output_dir = Path(args.output_dir) if args.output_dir else Path("processed_results")
    build_summary(roots, output_dir, args.smooth_window)


if __name__ == "__main__":
    main()
