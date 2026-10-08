# -*- coding: utf-8 -*-
import csv
import re
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips

GT_DIR = Path("./gt_UpIMG")
PRED_DIR = Path("./pre_UpIMG")
OUT_CSV = Path("image_metrics_per_sample.csv")
EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def extract_img_index(path):
    """
    Extract the common numeric sample id from filenames such as:
        IMG0_Ori_IMG.jpg
        IMG0_pre_UpIMG.jpg
        IMG123_gt_UpIMG.jpg
    Returns an integer id, or None if no leading IMG<number> pattern is found.
    """
    m = re.match(r'^IMG(\d+)', path.stem, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def collect_images(folder):
    mapping = {}
    skipped = []
    for p in folder.iterdir():
        if not (p.is_file() and p.suffix.lower() in EXTS):
            continue
        idx = extract_img_index(p)
        if idx is None:
            skipped.append(p.name)
            continue
        if idx in mapping:
            raise RuntimeError(
                f"Duplicate sample index IMG{idx} in {folder}: "
                f"{mapping[idx].name} and {p.name}"
            )
        mapping[idx] = p
    return mapping, skipped

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def nrmse(pred, gt):
    pred = pred.astype(np.float64); gt = gt.astype(np.float64)
    return float(np.sqrt(np.sum((pred-gt)**2)) / max(np.sqrt(np.sum(gt**2)), 1e-12))

def nmae(pred, gt):
    pred = pred.astype(np.float64); gt = gt.astype(np.float64)
    return float(np.sum(np.abs(pred-gt)) / max(np.sum(np.abs(gt)), 1e-12))

def to_lpips_tensor(img01, device):
    x = torch.from_numpy(img01).permute(2,0,1).unsqueeze(0)
    return (x*2.0 - 1.0).to(device)

def main():
    gt_map, skipped_gt = collect_images(GT_DIR)
    pred_map, skipped_pred = collect_images(PRED_DIR)
    common = sorted(set(gt_map) & set(pred_map))
    only_gt = sorted(set(gt_map) - set(pred_map))
    only_pred = sorted(set(pred_map) - set(gt_map))

    print(f"GT indexed images   : {len(gt_map)}")
    print(f"Pred indexed images : {len(pred_map)}")
    print(f"Matched pairs       : {len(common)}")
    print(f"GT-only indices     : {len(only_gt)}")
    print(f"Pred-only indices   : {len(only_pred)}")
    print(f"Skipped GT names    : {len(skipped_gt)}")
    print(f"Skipped Pred names  : {len(skipped_pred)}")

    if not common:
        raise RuntimeError(
            "No matched IMG<number> sample indices found. "
            "Expected names such as IMG0_Ori_IMG.jpg and IMG0_pre_UpIMG.jpg."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device).eval()

    rows = []
    with torch.no_grad():
        for i, idx in enumerate(common, 1):
            gt = load_rgb(gt_map[idx])
            pr = load_rgb(pred_map[idx])
            if gt.shape != pr.shape:
                raise ValueError(f"Shape mismatch for IMG{idx}: GT {gt.shape}, Pred {pr.shape}")
            psnr_v = peak_signal_noise_ratio(gt, pr, data_range=1.0)
            ssim_v = structural_similarity(gt, pr, data_range=1.0, channel_axis=2)
            lpips_v = float(lpips_model(to_lpips_tensor(pr, device), to_lpips_tensor(gt, device)).item())
            rows.append([idx, gt_map[idx].name, pred_map[idx].name, psnr_v, ssim_v, lpips_v, nrmse(pr,gt), nmae(pr,gt)])
            if i % 100 == 0 or i == len(common):
                print(f"Processed {i}/{len(common)}")

    arr = np.asarray([r[3:] for r in rows], dtype=np.float64)
    names = ["PSNR_dB","SSIM","LPIPS","NRMSE","NMAE"]
    print("\n========== Image generation metrics ==========")
    for j,name in enumerate(names):
        vals = arr[:,j]
        print(f"{name:8s}: mean={vals.mean():.6f}, std={vals.std():.6f}, median={np.median(vals):.6f}")
    print("FID     : use your existing pytorch-fid script separately.")

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["index", "gt_filename", "pred_filename"] + names)
        w.writerows(rows)
    print(f"Saved: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
