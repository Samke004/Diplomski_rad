"""
metrics.py — Multi-class segmentation metrics
────────────────────────────────────────────────────────────────────────────
Classes: 0=background, 1=trunk, 2=branches, 3=support

All functions take:
  pred : [H, W] int array  — argmax of model output, values 0..C-1
  gt   : [H, W] int array  — ground truth labels,    values 0..C-1
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import binary_dilation

NUM_CLASSES  = 4
CLASS_NAMES  = ["background", "trunk", "branches", "support"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _boundary(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask.astype(bool), iterations=1)
    return mask.astype(bool) & ~eroded


# ─────────────────────────────────────────────────────────────────────────────
# 1. Overall Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def overall_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred == gt))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-class IoU + Mean IoU
# ─────────────────────────────────────────────────────────────────────────────

def per_class_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> List[float]:
    """IoU for each class. Returns list of length num_classes."""
    ious = []
    for c in range(num_classes):
        p = pred == c
        g = gt   == c
        intersection = np.logical_and(p, g).sum()
        union        = np.logical_or(p, g).sum()
        ious.append(float(intersection / (union + 1e-10)))
    return ious



def frequency_weighted_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    """Frequency-weighted IoU — klase s više piksela imaju veći utjecaj."""
    ious  = per_class_iou(pred, gt, num_classes)
    total = gt.size
    fw    = 0.0
    for c in range(num_classes):
        freq  = float((gt == c).sum()) / total
        fw   += freq * ious[c]
    return fw


def pixel_accuracy_fg(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel accuracy samo na foreground pikselima (ignorira background klasu 0)."""
    fg = gt != 0
    if fg.sum() == 0:
        return float("nan")
    return float(np.mean(pred[fg] == gt[fg]))


def frequency_weighted_iou_fg(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    """fwIoU samo za foreground klase (1..C-1), frekvencije renormalizirane."""
    ious  = per_class_iou(pred, gt, num_classes)
    fg_mask = gt != 0
    total_fg = fg_mask.sum()
    if total_fg == 0:
        return float("nan")
    fw = 0.0
    for c in range(1, num_classes):
        freq = float((gt == c).sum()) / total_fg
        fw  += freq * ious[c]
    return fw

def mean_iou(pred: np.ndarray, gt: np.ndarray, ignore_bg: bool = True) -> float:
    """Mean IoU — optionally ignores background class (class 0)."""
    ious = per_class_iou(pred, gt)
    start = 1 if ignore_bg else 0
    return float(np.mean(ious[start:]))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-class Precision, Recall, F1
# ─────────────────────────────────────────────────────────────────────────────

def per_class_precision_recall_f1(
    pred: np.ndarray,
    gt:   np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, List[float]]:
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.logical_and(pred == c, gt == c).sum()
        fp = np.logical_and(pred == c, gt != c).sum()
        fn = np.logical_and(pred != c, gt == c).sum()
        p  = float(tp / (tp + fp + 1e-10))
        r  = float(tp / (tp + fn + 1e-10))
        f  = 2 * p * r / (p + r + 1e-10)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return {"precision": precisions, "recall": recalls, "f1": f1s}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Boundary F1 (per class)
# ─────────────────────────────────────────────────────────────────────────────

def boundary_f1_per_class(
    pred:  np.ndarray,
    gt:    np.ndarray,
    theta: int = 2,
    num_classes: int = NUM_CLASSES,
) -> List[float]:
    """Boundary F1 for each class."""
    f1s = []
    struct = np.ones((2 * theta + 1, 2 * theta + 1), dtype=bool)
    for c in range(num_classes):
        pred_b = _boundary(pred == c)
        gt_b   = _boundary(gt   == c)

        if pred_b.sum() == 0 and gt_b.sum() == 0:
            f1s.append(1.0); continue
        if pred_b.sum() == 0 or gt_b.sum() == 0:
            f1s.append(0.0); continue

        pred_dil = binary_dilation(pred_b, structure=struct)
        gt_dil   = binary_dilation(gt_b,   structure=struct)
        prec = float(np.logical_and(pred_b, gt_dil).sum()  / (pred_b.sum() + 1e-10))
        rec  = float(np.logical_and(gt_b,   pred_dil).sum()/ (gt_b.sum()   + 1e-10))
        f1s.append(2 * prec * rec / (prec + rec + 1e-10))
    return f1s


def mean_boundary_f1(pred, gt, theta=2, ignore_bg=True):
    f1s   = boundary_f1_per_class(pred, gt, theta)
    start = 1 if ignore_bg else 0
    return float(np.mean(f1s[start:]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Branch Recall (class 2) + Occluded Branch Recall
# ─────────────────────────────────────────────────────────────────────────────

def branch_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Recall for branch class (class index 2)."""
    tp = np.logical_and(pred == 2, gt == 2).sum()
    fn = np.logical_and(pred != 2, gt == 2).sum()
    return float(tp / (tp + fn + 1e-10))


def occluded_branch_recall(
    pred:      np.ndarray,
    gt:        np.ndarray,
    occlusion: np.ndarray,
) -> float:
    """Recall for branch pixels (class 2) that lie in occluded regions."""
    occ_branch_gt = (gt == 2) & occlusion.astype(bool)
    n = occ_branch_gt.sum()
    if n == 0:
        return float("nan")
    tp = np.logical_and(pred == 2, occ_branch_gt).sum()
    return float(tp / n)


# ─────────────────────────────────────────────────────────────────────────────
# 6 & 7. Difficulty Indices
# ─────────────────────────────────────────────────────────────────────────────

def occlusion_difficulty_index(gt: np.ndarray, occlusion: np.ndarray) -> float:
    branch_pixels = (gt == 2).sum()
    if branch_pixels == 0:
        return float("nan")
    return float(np.logical_and(gt == 2, occlusion.astype(bool)).sum() / branch_pixels)


def depth_difficulty_index(depth: np.ndarray, gt: np.ndarray) -> float:
    branch_mask = gt == 2
    if branch_mask.sum() == 0:
        return float("nan")
    return float(np.std(depth[branch_mask]))


# ─────────────────────────────────────────────────────────────────────────────
# Master function
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    pred:           np.ndarray,
    gt:             np.ndarray,
    occlusion:      Optional[np.ndarray] = None,
    depth:          Optional[np.ndarray] = None,
    boundary_theta: int = 2,
) -> Dict:
    """
    Compute all metrics for one image.
    pred / gt : [H, W] int arrays with class indices 0..3
    """
    ious    = per_class_iou(pred, gt)
    prf     = per_class_prf = per_class_precision_recall_f1(pred, gt)
    bf1s    = boundary_f1_per_class(pred, gt, theta=boundary_theta)

    results = {
        "overall_accuracy":       overall_accuracy(pred, gt),
        "pixel_accuracy":         overall_accuracy(pred, gt),
        "pixel_accuracy_fg":      pixel_accuracy_fg(pred, gt),
        "frequency_weighted_iou":    frequency_weighted_iou(pred, gt),
        "frequency_weighted_iou_fg": frequency_weighted_iou_fg(pred, gt),
        "mean_iou":         mean_iou(pred, gt, ignore_bg=True),
        "mean_boundary_f1": mean_boundary_f1(pred, gt, boundary_theta),
        "branch_recall":    branch_recall(pred, gt),
    }

    # Per-class metrics
    for i, name in enumerate(CLASS_NAMES):
        results[f"iou_{name}"]       = ious[i]
        results[f"precision_{name}"] = prf["precision"][i]
        results[f"recall_{name}"]    = prf["recall"][i]
        results[f"f1_{name}"]        = prf["f1"][i]
        results[f"boundary_f1_{name}"] = bf1s[i]

    if occlusion is not None:
        results["occluded_branch_recall"]     = occluded_branch_recall(pred, gt, occlusion)
        results["occlusion_difficulty_index"] = occlusion_difficulty_index(gt, occlusion)

    if depth is not None:
        results["depth_difficulty_index"] = depth_difficulty_index(depth, gt)

    return results


def aggregate_metrics(metric_list: list) -> Dict:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    agg  = {}
    for k in keys:
        vals = [m[k] for m in metric_list
                if k in m and not np.isnan(float(m[k]))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg
