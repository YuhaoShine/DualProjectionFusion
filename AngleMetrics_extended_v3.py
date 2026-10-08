# -*- coding: utf-8 -*-
"""
Extended angle evaluation.
Keeps the SAME scalar angular-error definition as the original testSHANnew.py,
so Acc@1/2/3/... remains directly comparable with the existing paper tables.

Input:
    PitchRollAngGT.txt
    PitchRollAngPRED.txt

Each row:
    pitch_deg, roll_deg

Outputs:
    - Acc@ thresholds
    - MAE / RMSE / Median / P95 computed from the SAME scalar angular error
    - optional true SO(3) geodesic error (reported separately)
    - component-wise pitch/roll errors
    - per-sample CSV
"""

import math
import csv
import re
import numpy as np
from pathlib import Path

GT_PATH = Path("PitchRollAngGT.txt")
PRED_PATH = Path("PitchRollAngPRED.txt")
OUT_CSV = Path("angle_metrics_per_sample.csv")
THRESHOLDS = [1, 2, 3, 4, 5, 7.5, 10, 12, 15]


def load_pitch_roll_txt(path):
    """
    Robust loader for lines such as:
        85, -73
        83.47732543945312, -72.91093444824219
    Also tolerates brackets or extra spaces.
    """
    rows = []
    number_pattern = re.compile(
        r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    )
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            nums = number_pattern.findall(line)
            if not nums:
                continue
            if len(nums) < 2:
                raise ValueError(
                    f"{path}: line {line_no} has fewer than two numeric values: {line.strip()}"
                )
            rows.append([float(nums[0]), float(nums[1])])

    if not rows:
        raise ValueError(f"No valid pitch/roll rows found in {path}.")
    return np.asarray(rows, dtype=np.float64)


def safe_acos(x):
    return math.acos(float(np.clip(x, -1.0, 1.0)))


def original_scalar_angle_error_deg(gt_pr, pred_pr):
    """
    EXACTLY follows the logic of the original testSHANnew.py:

      ErrPR = pred - gt
      R = Rx(delta_roll) @ Ry(delta_pitch)
      upright = [0,0,1]
      error = angle(upright, R @ upright)

    Use this error for threshold accuracy AND MAE/RMSE/Median/P95 so the
    additional statistics are consistent with the existing Acc@ thresholds.
    """
    diff_pitch = float(pred_pr[0] - gt_pr[0])
    diff_roll = float(pred_pr[1] - gt_pr[1])

    p = math.radians(diff_pitch)
    r = math.radians(diff_roll)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(r), -math.sin(r)],
        [0.0, math.sin(r),  math.cos(r)],
    ], dtype=np.float64)

    Ry = np.array([
        [ math.cos(p), 0.0, math.sin(p)],
        [0.0,          1.0, 0.0],
        [-math.sin(p), 0.0, math.cos(p)],
    ], dtype=np.float64)

    R = Rx @ Ry
    upright = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    upright_pred = R @ upright
    cosang = np.dot(upright, upright_pred) / (
        np.linalg.norm(upright) * np.linalg.norm(upright_pred)
    )
    return math.degrees(safe_acos(cosang))


def absolute_rotation_matrix(pitch_deg, roll_deg):
    """
    Same pitch/roll convention, but built from the absolute pose.
    Used only for the optional SO(3) geodesic metric.
    """
    p = math.radians(float(pitch_deg))
    r = math.radians(float(roll_deg))

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(r), -math.sin(r)],
        [0.0, math.sin(r),  math.cos(r)],
    ], dtype=np.float64)

    Ry = np.array([
        [ math.cos(p), 0.0, math.sin(p)],
        [0.0,          1.0, 0.0],
        [-math.sin(p), 0.0, math.cos(p)],
    ], dtype=np.float64)

    return Rx @ Ry


def so3_geodesic_error_deg(gt_pr, pred_pr):
    """
    Optional full rotation geodesic error:
        theta = acos((trace(R_pred R_gt^T)-1)/2)
    yaw is fixed to zero in both poses.
    """
    R_gt = absolute_rotation_matrix(gt_pr[0], gt_pr[1])
    R_pr = absolute_rotation_matrix(pred_pr[0], pred_pr[1])
    R_rel = R_pr @ R_gt.T
    cosang = (np.trace(R_rel) - 1.0) / 2.0
    return math.degrees(safe_acos(cosang))


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "MAE": float(np.mean(np.abs(x))),
        "RMSE": float(np.sqrt(np.mean(x ** 2))),
        "Median": float(np.median(x)),
        "P95": float(np.percentile(x, 95)),
    }


def main():
    gt = load_pitch_roll_txt(GT_PATH)
    pred = load_pitch_roll_txt(PRED_PATH)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape}, PRED {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 2:
        raise ValueError(f"Expected Nx2 arrays, got GT {gt.shape}")

    # Keep full-precision predictions. Do NOT round to 2 decimals.
    scalar_err = np.asarray(
        [original_scalar_angle_error_deg(g, p) for g, p in zip(gt, pred)],
        dtype=np.float64
    )
    geodesic_err = np.asarray(
        [so3_geodesic_error_deg(g, p) for g, p in zip(gt, pred)],
        dtype=np.float64
    )

    diff = pred - gt
    pitch_abs = np.abs(diff[:, 0])
    roll_abs = np.abs(diff[:, 1])

    print("========== Angle evaluation ==========")
    print(f"Samples          : {len(gt)}")

    print("\nThreshold accuracy (same definition as original testSHANnew.py):")
    for t in THRESHOLDS:
        acc = np.mean(scalar_err <= t) * 100.0
        print(f"Acc@{t:g}°          : {acc:.2f}%")

    s = summarize(scalar_err)
    print("\nAdditional statistics of the SAME scalar angular error:")
    print(f"MAE             : {s['MAE']:.4f} deg")
    print(f"RMSE            : {s['RMSE']:.4f} deg")
    print(f"Median          : {s['Median']:.4f} deg")
    print(f"P95             : {s['P95']:.4f} deg")

    g = summarize(geodesic_err)
    print("\nOptional SO(3) geodesic error (yaw fixed to 0):")
    print(f"Geodesic MAE    : {g['MAE']:.4f} deg")
    print(f"Geodesic RMSE   : {g['RMSE']:.4f} deg")
    print(f"Geodesic Median : {g['Median']:.4f} deg")
    print(f"Geodesic P95    : {g['P95']:.4f} deg")

    print("\nComponent-wise errors:")
    print(f"Pitch MAE       : {pitch_abs.mean():.4f} deg")
    print(f"Pitch RMSE      : {np.sqrt(np.mean(diff[:,0]**2)):.4f} deg")
    print(f"Roll MAE        : {roll_abs.mean():.4f} deg")
    print(f"Roll RMSE       : {np.sqrt(np.mean(diff[:,1]**2)):.4f} deg")

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "index",
            "gt_pitch_deg", "gt_roll_deg",
            "pred_pitch_deg", "pred_roll_deg",
            "abs_pitch_error_deg", "abs_roll_error_deg",
            "scalar_angle_error_deg_original_definition",
            "so3_geodesic_error_deg",
        ])
        for i, (gtr, pr, pa, ra, se, ge) in enumerate(
            zip(gt, pred, pitch_abs, roll_abs, scalar_err, geodesic_err)
        ):
            w.writerow([i, gtr[0], gtr[1], pr[0], pr[1], pa, ra, se, ge])

    print(f"\nSaved per-sample results to: {OUT_CSV.resolve()}")
    print("======================================")


if __name__ == "__main__":
    main()
