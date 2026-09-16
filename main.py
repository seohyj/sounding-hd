import argparse
import torch
import os
import random
import numpy as np
from sklearn.model_selection import KFold, train_test_split

from model.configs import Config
from model.solver import Solver

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def single_run(config, args, train_loader, val_loader, test_loader):
    print(f"\n==== Fold/Split {config.fold_id} | Run {config.run_id} | Seed {config.seed} ====")
    set_seed(config.seed)

    solver = Solver(config, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    solver.build()
    test_metrics = None

    if args.mode == "train":
        solver.train()
        test_metrics = solver.test(config.ckpt_path)
    else:
        if args.ckpt_path is None:
            print("Please provide --ckpt_path for test mode")
            return None, config
        solver.model.load_state_dict(torch.load(args.ckpt_path))
        test_metrics = solver.test(args.ckpt_path)
    
    return test_metrics, config

def aggregate_and_print_results(metrics_list, title, log_path=None):
    if not metrics_list:
        print(f"No metrics to aggregate for {title}")
        return {}
        
    all_keys = set().union(*(d.keys() for d in metrics_list))
    aggregated_metrics = {key: [m.get(key) for m in metrics_list if m.get(key) is not None] for key in all_keys}
    
    print("\n" + "="*50)
    print(f"Aggregated Results: {title}")
    print("="*50)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    f = open(log_path, 'w') if log_path else None
    if f:
        f.write(f"Aggregated Results: {title}\n")
        f.write("Metric           | Mean  ±  Std\n")
        f.write("-----------------|------------------\n")

    mean_metrics = {}
    metrics_to_scale = ['top5_mAP', 'hit_at_1', 'mAP', 'F1', 'mAP@15', 'mAP@50']
    
    for key, values in sorted(aggregated_metrics.items()):
        if not values: continue
        mean = np.mean(values)
        std = np.std(values)
        mean_metrics[key] = mean
        
        if key in metrics_to_scale:
            display_mean, display_std = (mean * 100, std * 100)
            result_str = f"{key:<16} | {display_mean:.2f} ± {display_std:.2f}"
        else:
            result_str = f"{key:<16} | {mean:.4f} ± {std:.4f}"
        
        print(result_str)
        if f: f.write(result_str + '\n')
    
    print("="*50)
    if f:
        f.close()
        print(f"Saved aggregated results to {log_path}")
        
    return mean_metrics

def get_dataloaders(args, train_keys, val_keys, test_keys):
    config = Config(args)
    config.train_keys = train_keys
    config.val_keys = val_keys
    config.test_keys = test_keys
    if hasattr(config, 'use_hdf5') and config.use_hdf5:
        from dataset.loaders.dataset_use_hdf5 import get_dataloader
    elif config.mel_spec_dir:
        from dataset.loaders.dataset_melspec import get_dataloader
    else:
        from dataset.loaders.dataset import get_dataloader

    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    test_loader = get_dataloader(config, split='test')
    return train_loader, val_loader, test_loader, config

def run_tvsum(args):
    """Runs the 5-fold cross-validation protocol for TVSum."""
    print("Running TVSum with 5-Fold Cross-Validation Protocol.")
    temp_config = Config(args)
    all_video_ids = np.array(sorted([f[:-4] for f in os.listdir(temp_config.audio_dir) if f.endswith(".npy")]))
    
    if args.repeat:
        seeds = [42, 123, 456, 777, 999]
        all_run_metrics = []
        all_fold_metrics = []
        final_config = None
        
        for run_idx, seed in enumerate(seeds, 1):
            args.run_id = run_idx
            args.seed = seed
            set_seed(seed)
            
            print(f"\n{'#'*25} STARTING RUN {run_idx}/5 (Seed {seed}) {'#'*25}")
            
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            
            fold_metrics_for_run = []
            
            for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_video_ids)):
                args.fold_id = fold_idx + 1
                print(f"\n  {'-'*20} FOLD {args.fold_id}/5 (Run {run_idx}, Seed {seed}) {'-'*20}")
                
                train_ids = all_video_ids[train_indices]
                val_ids = all_video_ids[val_indices]
                
                # In 5FCV, the validation set is also the test set for this fold
                train_loader, val_loader, test_loader, config_run = get_dataloaders(args, train_ids, val_ids, val_ids)
                
                test_metrics, final_config = single_run(config_run, args, train_loader, val_loader, test_loader)
                if test_metrics:
                    fold_metrics_for_run.append(test_metrics)
                    all_fold_metrics.append(test_metrics)
            
            if fold_metrics_for_run:
                if final_config:
                    run_dir = os.path.dirname(final_config.save_dir)
                else:
                    base_exp_name = f"{args.dataset}_{args.tag}" if args.tag else f"{args.dataset}_repeat"
                    run_dir = os.path.join(temp_config.output_root, base_exp_name, f"run_{run_idx}_seed_{seed}")
                
                run_summary_path = os.path.join(run_dir, 'run_summary.txt')
                mean_run_metrics = aggregate_and_print_results(
                    fold_metrics_for_run, 
                    f"Run {run_idx} (Seed {seed}) - Average over 5 folds", 
                    run_summary_path
                )
                
                if mean_run_metrics:
                    all_run_metrics.append(mean_run_metrics)
        
        if all_fold_metrics:
            base_exp_name = f"{args.dataset}_{args.tag}" if args.tag else f"{args.dataset}_repeat"
            base_exp_dir = os.path.join(temp_config.output_root, base_exp_name)
            final_summary_path = os.path.join(base_exp_dir, 'final_summary.txt')
            aggregate_and_print_results(
                all_fold_metrics,
                "Final Repeated 5-Fold Cross-Validation (Mean ± Std over 25 folds: 5 runs × 5 folds)", 
                final_summary_path
            )
    
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        all_fold_metrics = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_video_ids)):
            args.fold_id = fold_idx + 1
            print(f"\n{'#'*25} STARTING FOLD {args.fold_id}/5 {'#'*25}")
            
            train_ids = all_video_ids[train_indices]
            val_ids = all_video_ids[val_indices]
            
            train_loader, val_loader, test_loader, config_run = get_dataloaders(args, train_ids, val_ids, val_ids)
            test_metrics, _ = single_run(config_run, args, train_loader, val_loader, test_loader)
            if test_metrics:
                all_fold_metrics.append(test_metrics)
        
        final_summary_config = Config(args)
        base_exp_dir = os.path.dirname(final_summary_config.save_dir)
        final_summary_path = os.path.join(base_exp_dir, 'final_summary.txt')
        aggregate_and_print_results(all_fold_metrics, "5-Fold Cross-Validation (Single Run)", final_summary_path)

def run_mrhisum(args):
    """Runs MR-HiSum with its predefined splits; the Solver builds its own dataloaders."""
    print("Running MR-HiSum with predefined splits.")
    def mrhisum_single_run(run_id, seed, args):
        print(f"\n==== Run {run_id} | Seed {seed} ====")
        set_seed(seed)
        
        args.run_id = run_id
        args.seed = seed
        config = Config(args)

        solver = Solver(config)
        solver.build()
        if args.mode == "train":
            solver.train()
            return solver.test(config.ckpt_path), config
        else:
            if args.ckpt_path is None: return None, config
            return solver.test(args.ckpt_path), config

    if args.repeat:
        seeds = [42, 123, 456, 777, 999]
        all_test_metrics = []
        final_config = None
        for i, seed in enumerate(seeds, 1):
            test_metrics, final_config = mrhisum_single_run(i, seed, args)
            if test_metrics: all_test_metrics.append(test_metrics)
        
        if final_config:
            summary_path = os.path.join(os.path.dirname(final_config.save_dir), 'summary_results.txt')
            aggregate_and_print_results(all_test_metrics, "MR-HiSum over 5 seeds", summary_path)
    else:
        mrhisum_single_run(args.run_id, args.seed, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["train", "test"], default="train")
    parser.add_argument('--dataset', type=str, choices=["tvsum", "mrhisum"], default="tvsum")
    parser.add_argument('--tag', type=str, default="", help="A custom tag for the experiment directory.")
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--top_ratio', type=float, default=0.5, help="Ratio for selecting top scored frames in some evaluation metrics.")

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--l2_reg', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping based on validation metric')
    parser.add_argument('--patience', type=int, default=15, help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--save_all_metric_ckpts', action='store_true', default=False,
                        help="Also write a separate checkpoint for the best epoch of every secondary metric "
                             "(best_model_map15.pt, best_model_kendall.pt, ...). Off by default: each state dict "
                             "is ~760MB, so this writes several GB per epoch. best_model.pt (selected on the "
                             "validation F1) is always saved and is the checkpoint used for reported results.")
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--use_hdf5', action='store_true', help='Use HDF5 format dataset (for MR-HiSum).')
    
    parser.add_argument('--repeat', action='store_true', help="Repeat experiments with 5 different seeds.")
    parser.add_argument('--run_id', type=int, default=1, help="Identifier for a single run.")
    parser.add_argument('--seed', type=int, default=42, help="Main random seed for reproducibility.")
    
    parser.add_argument('--fold_id', type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.dataset == 'tvsum':
        run_tvsum(args)
    else:
        run_mrhisum(args)

if __name__ == "__main__":
    main()