

import os
import numpy as np
import glob
import csv
from PIL import Image
def parse_rank_file(rank_path):
    rows = []
    with open(rank_path, "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            idx = parts[0]
            pred_label = int(parts[1])
            obj1 = float(parts[2])
            obj2 = float(parts[3])
            rows.append({"idx": idx, "pred_label": pred_label, "obj1": obj1, "obj2": obj2})
    return rows


def process_all_samples(root_dir, true_label):
    sample_dirs = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
    best_candidates = []
    for sampel_dir in sample_dirs:
        rank_path = os.path.join(sampel_dir, "rank_0_score.txt")
        if not os.path.exists(rank_path):
            continue
        rows = parse_rank_file(rank_path)
        if not rows:
            continue
        best = find_best_candidate(rows, true_label)
        best_candidates.append(best)
import os
import numpy as np
import glob
import csv
import argparse

def parse_rank_file(rank_path):
    rows = []
    with open(rank_path, "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 4:
                continue
            idx = parts[0]
            pred_label = int(parts[1])
            obj1 = float(parts[2])
            obj2 = float(parts[3])
            rows.append({"idx": idx, "pred_label": pred_label, "obj1": obj1, "obj2": obj2})
    return rows

def find_best_candidate(rows):
    valid = [r for r in rows if r["obj1"] < 0]
    if valid:
        best = min(valid, key=lambda r: r["obj2"])
    else:
        best = min(rows, key=lambda r: r["obj1"])
    retain_class = int(any(r["obj1"] < 0 for r in rows))
    return {**best, "retain_class": retain_class}

def process_all_samples(result_dir):
    # Recursively find all folders containing rank_0_score.txt
    best_candidates = []
    for class_name in os.listdir(result_dir):
        class_path = os.path.join(result_dir, class_name)
        for sample_name in os.listdir(class_path): # just one sampple path
            sample_path = os.path.join(class_path, sample_name)

        rank_path = os.path.join(sample_path, "rank0_scores.txt")
        original_img =  os.path.join(sample_path, "clean_image.png")
        original_map =  os.path.join(sample_path, "clean_map.png")
        rows = parse_rank_file(rank_path)
        if not rows:
            continue
        best = find_best_candidate(rows)
        best_rank0_id = best["idx"]
        best_rank0_advimg = os.path.join(sample_path, "rank0", f"adv_{int(best_rank0_id):03d}.png") # adv_000 dạng như là 000 rồi số id á


        # calculate l0
        original_img = np.array(Image.open(original_img))
        best_adv_img = np.array(Image.open(best_rank0_advimg))
        l0 = np.sum(np.any(original_img != best_adv_img, axis=-1))
        best["l0"] = l0

        best_candidates.append(best)

    mean_obj1 = float(np.mean([b["obj1"] for b in best_candidates])) if best_candidates else 0.0
    mean_obj2 = float(np.mean([b["obj2"] for b in best_candidates])) if best_candidates else 0.0
    mean_l0 = float(np.mean([b["l0"] for b in best_candidates])) if best_candidates else 0.0
    retain_class_rate = float(np.mean([b["retain_class"] for b in best_candidates])) if best_candidates else 0.0

    print(f"Mean obj1: {mean_obj1}")
    print(f"Mean obj2: {mean_obj2}")
    print(f"Mean l0: {mean_l0}")
    print(f"Retain class rate: {retain_class_rate}")

    # with open(os.path.join(result_dir, "best_candidates.csv"), "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=["idx", "pred_label", "obj1", "obj2", "l0", "retain_class"])
    #     writer.writeheader()
    #     for b in best_candidates:
    #         writer.writerow(b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process result folder and compute statistics.")
    parser.add_argument("--result_dir", required=True, help="Path to folder containing sample subfolders.")
    
    args = parser.parse_args()
    process_all_samples(args.result_dir)


