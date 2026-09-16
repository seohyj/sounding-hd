import numpy as np
from sklearn.metrics import average_precision_score
import torch


def get_ap(pred, gt):
    """
    Args:
        pred: list or np.array of predicted scores (higher = more confident)
        gt: list or np.array of ground truth scores (continuous values between 0 and 1)
    Returns:
        average precision using continuous scores
    """
    pred = np.array(pred)
    gt = np.array(gt)
    
    if pred.max() - pred.min() > 0:
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    
    indices = np.argsort(-pred)
    sorted_gt = gt[indices]
    
    gt_cumsum = np.cumsum(sorted_gt)
    total_gt = np.sum(gt)
    
    precision = gt_cumsum / np.arange(1, len(gt) + 1)
    recall = gt_cumsum / (total_gt + 1e-8)
    
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else 0
        ap += p / 11
    
    return ap


def compute_top5_map(all_scores, all_labels):
    """
    For each video, get top-5 scored clips and compute AP **only on these 5 clips**.
    This correctly implements the "mAP at top-5" metric for TVSum.
    """
    ap_list = []
    for scores, labels in zip(all_scores, all_labels):
        topk = min(5, len(scores))
        if topk == 0:
            continue

        indices = np.argsort(scores)[-topk:]

        top_k_preds = scores[indices]
        top_k_labels = labels[indices]

        ap = get_ap(top_k_preds, top_k_labels)
        ap_list.append(ap)
        
    return np.mean(ap_list) if ap_list else 0.0

def compute_hit_at_1(all_scores, all_labels):
    hit_list = []
    for scores, labels in zip(all_scores, all_labels):
        top1_idx = np.argmax(scores)
        hit = labels[top1_idx]
        hit_list.append(hit)
    return np.mean(hit_list)


def compute_map(all_scores, all_labels, top_k=None):
    ap_list = []
    for scores, labels in zip(all_scores, all_labels):
        if top_k is not None:
            indices = np.argsort(scores)[-top_k:]
            pred = np.zeros_like(scores)
            pred[indices] = scores[indices]
        else:
            pred = scores
        ap = get_ap(pred, labels)
        ap_list.append(ap)
    return np.mean(ap_list)


def generate_mrhisum_seg_scores(cp_frame_scores, uniform_clip=5):
    """MR.HiSum: frame-level scores to uniform segments"""
    if isinstance(cp_frame_scores, torch.Tensor):
        cp_frame_scores = cp_frame_scores.detach().cpu().numpy()
    
    cp_frame_scores = np.asarray(cp_frame_scores, dtype=np.float32)
    
    cp_frame_scores = np.nan_to_num(cp_frame_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    if len(cp_frame_scores) == 0:
        return np.array([0.0])
    
    if uniform_clip >= len(cp_frame_scores):
        return np.array([np.mean(cp_frame_scores)])
    
    splits = []
    for i in range(0, len(cp_frame_scores), uniform_clip):
        split = cp_frame_scores[i:i+uniform_clip]
        if len(split) > 0:
            splits.append(np.mean(split))
    
    return np.array(splits) if splits else np.array([0.0])

def top50_summary(scores):
    """MR.HiSum: select top 50% segments.

    Selects the binary summary directly from the top-scoring segments, rather than
    solving the 0/1 knapsack over shot boundaries the way Mr.HiSum's own benchmark
    code does — this matches the paper's own stated protocol ("select the top 50% of
    segments"), applied uniformly to both TVSum and Mr.HiSum.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    scores = np.asarray(scores, dtype=np.float32)
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    if len(scores) <= 1:
        return [1] if len(scores) == 1 else [0]
    
    median_index = max(1, len(scores) // 2)
    sort_idx = np.argsort(scores)[::-1]
    filtered_sort_idx = sort_idx[:median_index]
    
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        if 0 <= index < len(scores):
            selected_segs[index] = 1
    
    return selected_segs

def top15_summary(scores):
    """MR.HiSum: select top 15% segments"""
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    scores = np.asarray(scores, dtype=np.float32)
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    if len(scores) <= 1:
        return [1] if len(scores) == 1 else [0]
    
    filter_index = max(1, int(len(scores) * 0.15))
    sort_idx = np.argsort(scores)[::-1]
    filtered_sort_idx = sort_idx[:filter_index]
    
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        if 0 <= index < len(scores):
            selected_segs[index] = 1
    
    return selected_segs

def compute_f1_score(all_scores, all_labels, threshold=0.5):
    """MR.HiSum: compute F1 score"""
    f1s = []
    for scores, labels in zip(all_scores, all_labels):
        try:
            scores = np.asarray(scores, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)
            
            scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
            labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            
            if len(scores) == 0 or len(labels) == 0 or np.sum(labels) == 0:
                continue
            
            # MR.HiSum: Binary intersection-based F1
            S = np.array(top50_summary(scores), dtype=int)
            G = np.array(top50_summary(labels), dtype=int)
            
            min_len = min(len(S), len(G))
            S = S[:min_len]
            G = G[:min_len]
            
            overlapped = S & G
            sum_S = np.sum(S)
            sum_G = np.sum(G)
            sum_overlapped = np.sum(overlapped)
            
            precision = sum_overlapped / (sum_S + 1e-8)
            recall = sum_overlapped / (sum_G + 1e-8)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
            
            if np.isnan(f1) or np.isinf(f1):
                f1 = 0.0
                
            f1s.append(f1)
            
        except Exception as e:
            print(f"Warning: Error in F1 computation: {e}, skipping sample")
            continue
    
    return np.mean(f1s) if f1s else 0.0

def compute_map_at_threshold(all_scores, all_labels, threshold=0.15):
    """MR.HiSum: compute mAP@15/50"""
    from sklearn.metrics import average_precision_score
    
    ap_list = []
    for scores, labels in zip(all_scores, all_labels):
        try:
            scores = np.asarray(scores, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)
            
            scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
            labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            
            if len(scores) == 0 or len(labels) == 0 or np.sum(labels) == 0:
                continue
            
            gt_seg_scores = generate_mrhisum_seg_scores(labels, uniform_clip=5)
            pred_seg_scores = generate_mrhisum_seg_scores(scores, uniform_clip=5)
            
            min_len = min(len(gt_seg_scores), len(pred_seg_scores))
            gt_seg_scores = gt_seg_scores[:min_len]
            pred_seg_scores = pred_seg_scores[:min_len]
            
            if len(pred_seg_scores) == 0:
                pred_seg_scores = np.array([0.0])
            else:
                pred_seg_scores = np.clip(pred_seg_scores, -50, 50)
                exp_scores = np.exp(pred_seg_scores)
                sum_exp = np.sum(exp_scores)
                
                if sum_exp <= 1e-15:
                    pred_seg_scores = np.ones_like(pred_seg_scores) / len(pred_seg_scores)
                else:
                    pred_seg_scores = exp_scores / sum_exp
            
            pred_seg_scores = np.nan_to_num(pred_seg_scores, nan=0.0, posinf=1.0, neginf=0.0)
            
            if threshold == 0.15:
                gt_binary = top15_summary(gt_seg_scores)
            else:
                gt_binary = top50_summary(gt_seg_scores)
            
            gt_binary = np.array(gt_binary, dtype=int)
            
            min_len = min(len(gt_binary), len(pred_seg_scores))
            gt_binary = gt_binary[:min_len]
            pred_seg_scores = pred_seg_scores[:min_len]
            
            if np.isnan(pred_seg_scores).any() or np.isinf(pred_seg_scores).any():
                print(f"Warning: NaN/Inf detected in pred_seg_scores before sklearn, replacing with zeros")
                pred_seg_scores = np.nan_to_num(pred_seg_scores, nan=0.0, posinf=1.0, neginf=0.0)
            
            if np.isnan(gt_binary).any() or np.isinf(gt_binary).any():
                print(f"Warning: NaN/Inf detected in gt_binary before sklearn, replacing with zeros")
                gt_binary = np.nan_to_num(gt_binary, nan=0, posinf=1, neginf=0).astype(int)
            
            if np.sum(gt_binary) == 0:
                continue
                
            ap = average_precision_score(gt_binary, pred_seg_scores)
            
            if np.isnan(ap) or np.isinf(ap):
                ap = 0.0
                
            ap_list.append(ap)
            
        except Exception as e:
            print(f"Warning: Error in mAP@{threshold} computation: {e}, skipping sample")
            continue
        
    return np.mean(ap_list) if ap_list else 0.0
