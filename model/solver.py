import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import kendalltau, spearmanr

from networks.ProposedAVHDNet import ProposedAVHDNet
from model.utils.evaluate_map import (
    compute_map,
    compute_top5_map,
    compute_hit_at_1, 
    compute_f1_score,
    compute_map_at_threshold,
)

class Solver(object):
    def __init__(self, config, train_loader=None, val_loader=None, test_loader=None):
        self.config = config
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Pre-computed dataloaders are used for TVSum's 5-fold CV, since the fold's
        # train/val split has to be known before the Solver is constructed.
        if train_loader is not None and val_loader is not None:
            print("INFO: Using pre-computed dataloaders.")
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader if test_loader is not None else val_loader
        else:
            # Fallback for MR-HiSum, which uses its own predefined splits.
            print("INFO: Creating new dataloaders in Solver.")
            if hasattr(config, 'use_hdf5') and config.use_hdf5:
                print("INFO: Using HDF5 format dataset")
                from dataset.loaders.dataset_use_hdf5 import get_dataloader
            else:
                print("INFO: Generating mel-spectrograms on-the-fly.")
                from dataset.loaders.dataset import get_dataloader

            self.train_loader = get_dataloader(config, split='train')
            self.val_loader = get_dataloader(config, split='val')
            self.test_loader = get_dataloader(config, split='test')
        
        self.device = torch.device(config.device)

        self.criterion = nn.MSELoss(reduction='none')
        
        self.writer = SummaryWriter(config.tensorboard_logdir)
        self.config.save_config()
        
        self.best_metrics = {
            'score': {'value': -1, 'epoch': -1},
            'hit1': {'value': -1, 'epoch': -1},
            'f1': {'value': -1, 'epoch': -1},
            'map15': {'value': -1, 'epoch': -1},
            'map50': {'value': -1, 'epoch': -1},
            'kendall_tau': {'value': -2, 'epoch': -1}, # -2 to handle range from -1 to 1
            'spearman_rho': {'value': -2, 'epoch': -1} # -2 to handle range from -1 to 1
        }
        
        self.log_path = os.path.join(self.config.log_dir, 'result_log.txt')
        
        self.global_step = 0

    def build(self):
        self.model = ProposedAVHDNet(config=self.config).to(self.device)
            
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.lr,  
            weight_decay=self.config.l2_reg,  
            betas=(0.9, 0.999)
        )
            
        gamma = 0.99
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        
        self.init_weights(self.model, init_type='xavier')  
        
        return self.model

    def train(self):
        best_map = -1.0
        best_hit1 = -1.0
        best_f1 = -1.0
        best_map15 = -1.0
        best_map50 = -1.0
        best_kendall = -2.0 # Kendall's tau can be negative
        best_spearman = -2.0 # Spearman's rho can be negative
        best_primary_metric = -1.0

        best_map_epoch = 0
        best_hit1_epoch = 0
        best_f1_epoch = 0
        best_map15_epoch = 0
        best_map50_epoch = 0
        best_kendall_epoch = 0
        best_spearman_epoch = 0
        best_primary_metric_epoch = 0

        map_save_path = hit1_save_path = f1_save_path = None
        map15_save_path = map50_save_path = None

        no_improve_epochs = 0
        early_stopping_enabled = getattr(self.config, 'early_stopping', False)
        patience = getattr(self.config, 'patience', 15)

        self.model.train()
        
        if early_stopping_enabled:
            print(f"Early stopping enabled with patience={patience} epochs")
            print(f"Monitoring metric: {self.config.best_metric_name}")

        for epoch in range(self.config.epochs):
            print("[Epoch: {0:6}]".format(str(epoch)+"/"+str(self.config.epochs)))

            loss_history = []
            num_batches = int(len(self.train_loader))
            iterator = iter(self.train_loader)

            for step in tqdm(range(num_batches)):
                self.optimizer.zero_grad()
                batch = next(iterator)

                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                label = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                if torch.isnan(visual).any() or torch.isinf(visual).any():
                    print(f"Warning: NaN/Inf found in visual features, skipping step {step}. Video IDs: {batch.get('video_id', 'N/A')}")
                    continue
                if torch.isnan(audio).any() or torch.isinf(audio).any():
                    print(f"Warning: NaN/Inf found in audio features, skipping step {step}. Video IDs: {batch.get('video_id', 'N/A')}")
                    continue
                
                audio_mel_spec = batch.get('audio_mel_spec', None)
                if audio_mel_spec is not None:
                    audio_mel_spec = audio_mel_spec.to(self.device)
                
                score = self.model(visual_feat=visual, audio_feat=audio,
                                   audio_mel_spec=audio_mel_spec, mask=mask)

                loss_tensor = self.criterion(score, label)

                loss = (loss_tensor * mask.float()).sum() / mask.sum()

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                self.optimizer.step()
                loss_history.append(loss.item())

                self.writer.add_scalar('Train/Loss_step', loss.item(), self.global_step)
                self.global_step += 1


            loss = np.mean(np.array(loss_history))
            self.writer.add_scalar('Train/Loss_epoch', loss, epoch)
            self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            val_metrics = self.evaluate(self.val_loader, epoch)
            
            self.scheduler.step()
            
            primary_metric_name = self.config.best_metric_name
            current_primary_metric_val = val_metrics.get(primary_metric_name, -1)

            if best_primary_metric <= current_primary_metric_val:
                best_primary_metric = current_primary_metric_val
                best_primary_metric_epoch = epoch
                print(f"   => New best model found based on '{primary_metric_name}': {current_primary_metric_val:.5f} @ epoch {epoch}")
                torch.save(self.model.state_dict(), self.config.ckpt_path)
                
                if early_stopping_enabled:
                    no_improve_epochs = 0
                    print(f"   => Early stopping counter reset (patience: {patience})")
            else:
                if early_stopping_enabled:
                    no_improve_epochs += 1
                    print(f"   => No improvement for {no_improve_epochs}/{patience} epochs (best {primary_metric_name}: {best_primary_metric:.5f})")
                    
                    if no_improve_epochs >= patience:
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered at epoch {epoch}")
                        print(f"Best {primary_metric_name}: {best_primary_metric:.5f} @ epoch {best_primary_metric_epoch}")
                        print(f"No improvement for {patience} consecutive epochs")
                        print(f"{'='*60}\n")
                        break

            # The best epoch per secondary metric is always tracked for the logs. Writing a
            # checkpoint for each of them is opt-in: one state dict is ~760 MB, so saving all
            # of them every time a metric improves writes several GB per epoch, which
            # dominates the runtime and hammers network storage. best_model.pt above is the
            # one main.py reloads for the reported test metrics.
            save_all = getattr(self.config, 'save_all_metric_ckpts', False)

            def save_metric_ckpt(filename):
                path = os.path.join(self.config.log_dir, filename)
                if save_all:
                    torch.save(self.model.state_dict(), path)
                return path

            if best_map <= val_metrics['mAP']:
                best_map = val_metrics['mAP']
                best_map_epoch = epoch
                map_save_path = save_metric_ckpt('best_model_score.pt')

            if best_hit1 <= val_metrics['hit_at_1']:
                best_hit1 = val_metrics['hit_at_1']
                best_hit1_epoch = epoch
                hit1_save_path = save_metric_ckpt('best_model_hit1.pt')

            if best_f1 <= val_metrics['F1']:
                best_f1 = val_metrics['F1']
                best_f1_epoch = epoch
                # Note: for both datasets best_metric_name is "F1", so this fires on exactly
                # the same epochs as best_model.pt above and would be a byte-identical copy.
                f1_save_path = save_metric_ckpt('best_model_f1.pt')

            if best_map15 <= val_metrics['mAP@15']:
                best_map15 = val_metrics['mAP@15']
                best_map15_epoch = epoch
                map15_save_path = save_metric_ckpt('best_model_map15.pt')

            if best_map50 <= val_metrics['mAP@50']:
                best_map50 = val_metrics['mAP@50']
                best_map50_epoch = epoch
                map50_save_path = save_metric_ckpt('best_model_map50.pt')

            if best_kendall <= val_metrics['kendall_tau']:
                best_kendall = val_metrics['kendall_tau']
                best_kendall_epoch = epoch
                kendall_save_path = save_metric_ckpt('best_model_kendall.pt')

            if best_spearman <= val_metrics['spearman_rho']:
                best_spearman = val_metrics['spearman_rho']
                best_spearman_epoch = epoch
                spearman_save_path = save_metric_ckpt('best_model_spearman.pt')

            print(f"   [Epoch {epoch}] Train loss: {loss:.5f}")
            print(f"    VAL mAP {val_metrics['mAP']:.5f} | F1 {val_metrics['F1']:.5f} | Kendall's τ {val_metrics['kendall_tau']:.4f} | Spearman's ρ {val_metrics['spearman_rho']:.4f}")
            print(f"    (HIT@1 {val_metrics['hit_at_1']:.5f} | MAP@15 {val_metrics['mAP@15']:.5f} | MAP@50 {val_metrics['mAP@50']:.5f})")
        
        print(f'   Best Val mAP         {best_map:.5f} @ epoch {best_map_epoch}')
        print(f'   Best Val HIT@1  {best_hit1:.5f} @ epoch {best_hit1_epoch}')
        print(f'   Best Val F1     {best_f1:.5f} @ epoch {best_f1_epoch}')
        print(f'   Best Val MAP@15 {best_map15:.5f} @ epoch {best_map15_epoch}')
        print(f'   Best Val MAP@50 {best_map50:.5f} @ epoch {best_map50_epoch}')
        print(f"   Best Val Kendall's τ {best_kendall:.4f} @ epoch {best_kendall_epoch}")
        print(f"   Best Val Spearman's ρ {best_spearman:.4f} @ epoch {best_spearman_epoch}")

        with open(os.path.join(self.config.log_dir, 'results.txt'), 'a') as f:
            f.write(f'   Best Val mAP    {best_map:.5f} @ epoch {best_map_epoch}\n')
            f.write(f'   Best Val HIT@1  {best_hit1:.5f} @ epoch {best_hit1_epoch}\n')
            f.write(f'   Best Val F1     {best_f1:.5f} @ epoch {best_f1_epoch}\n')
            f.write(f'   Best Val MAP@15 {best_map15:.5f} @ epoch {best_map15_epoch}\n')
            f.write(f'   Best Val MAP@50 {best_map50:.5f} @ epoch {best_map50_epoch}\n\n')
            f.write(f"   Best Val Kendall's τ {best_kendall:.4f} @ epoch {best_kendall_epoch}\n")
            f.write(f"   Best Val Spearman's Rho: {best_spearman:.4f} @ epoch {best_spearman_epoch}\n\n")
            f.flush()
        
        self.writer.close()
        
        self.best_metrics = {
            'score': {'value': best_map, 'epoch': best_map_epoch},
            'hit1': {'value': best_hit1, 'epoch': best_hit1_epoch},
            'f1': {'value': best_f1, 'epoch': best_f1_epoch},
            'map15': {'value': best_map15, 'epoch': best_map15_epoch},
            'map50': {'value': best_map50, 'epoch': best_map50_epoch},
            'kendall_tau': {'value': best_kendall, 'epoch': best_kendall_epoch},
            'spearman_rho': {'value': best_spearman, 'epoch': best_spearman_epoch},
        }

        self._save_results_to_log()
        
        return map_save_path, hit1_save_path, f1_save_path, map15_save_path, map50_save_path

    def evaluate(self, loader, epoch=None):
        self.model.eval()
        all_scores = []
        all_labels = []
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                label = batch['label'].to(self.device)
                mask = batch.get('mask').to(self.device)
                
                audio_mel_spec = batch.get('audio_mel_spec', None)
                if audio_mel_spec is not None:
                    audio_mel_spec = audio_mel_spec.to(self.device)
                
                # The mask must be passed exactly as in train(), or the attention blocks
                # attend over the batch's zero padding.
                score = self.model(visual_feat=visual, audio_feat=audio,
                                   audio_mel_spec=audio_mel_spec, mask=mask)

                if torch.isnan(score).any():
                    print("Warning: NaN detected in model output, replacing with zeros")
                    score = torch.zeros_like(score)

                loss_tensor = self.criterion(score, label)
                batch_loss = (loss_tensor * mask.float()).sum() / mask.sum()

                if torch.isnan(batch_loss):
                    print("Warning: NaN detected in batch_loss, setting to 0")
                    batch_loss = torch.tensor(0.0)

                total_loss += batch_loss.item() * audio.size(0)
                total_samples += audio.size(0)

                scores = score.cpu().numpy()
                labels = label.cpu().numpy()

                scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
                labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
                
                for i in range(scores.shape[0]):
                    sample_score = scores[i]
                    sample_label = labels[i]
                    
                    if mask is not None:
                        sample_mask = mask[i].cpu().numpy()
                        sample_score = sample_score[sample_mask]
                        sample_label = sample_label[sample_mask]
                    
                    all_scores.append(sample_score)
                    all_labels.append(sample_label)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        hit_at_1_scores = []
        top5_map_scores = []
        map_scores = []
        f1_scores = []
        map15_scores = []
        map50_scores = []
        kendall_taus = []
        spearman_rhos = []
        
        for sample_score, sample_label in zip(all_scores, all_labels):
            if len(sample_label) == 0:
                continue

            eval_scores = sample_score
            eval_labels = sample_label

            # Correlation/evaluation needs at least 2 data points (frames)
            if len(eval_labels) < 2:
                continue
            
            if np.isnan(eval_scores).any():
                print(f"Warning: NaN detected in eval_scores, replacing with zeros")
                eval_scores = np.nan_to_num(eval_scores)
            
            if np.isnan(eval_labels).any():
                print(f"Warning: NaN detected in eval_labels, replacing with zeros") 
                eval_labels = np.nan_to_num(eval_labels)
            
            hit_at_1_scores.append(compute_hit_at_1([eval_scores], [eval_labels]))
            top5_map_scores.append(compute_top5_map([eval_scores], [eval_labels]))
            map_scores.append(compute_map([eval_scores], [eval_labels]))
            f1_scores.append(compute_f1_score([eval_scores], [eval_labels]))
            map15_scores.append(compute_map_at_threshold([eval_scores], [eval_labels], threshold=0.15))
            map50_scores.append(compute_map_at_threshold([eval_scores], [eval_labels], threshold=0.50))
            
            tau, _ = kendalltau(eval_scores, eval_labels)
            rho, _ = spearmanr(eval_scores, eval_labels)
            kendall_taus.append(0 if np.isnan(tau) else tau)
            spearman_rhos.append(0 if np.isnan(rho) else rho)

        metrics = {
            'loss': avg_loss,
            'hit_at_1': np.mean(hit_at_1_scores) if hit_at_1_scores else 0.0,
            'top5_mAP': np.mean(top5_map_scores) if top5_map_scores else 0.0,
            'mAP': np.mean(map_scores) if map_scores else 0.0,
            'F1': np.mean(f1_scores) if f1_scores else 0.0,
            'mAP@15': np.mean(map15_scores) if map15_scores else 0.0,
            'mAP@50': np.mean(map50_scores) if map50_scores else 0.0,
            'kendall_tau': np.mean(kendall_taus) if kendall_taus else 0.0,
            'spearman_rho': np.mean(spearman_rhos) if spearman_rhos else 0.0,
        }

        if epoch is not None:
            self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
            for k, v in metrics.items():
                if k != 'loss':
                    self.writer.add_scalar(f'Validation/{k}', v, epoch)

        return metrics

    def test(self, ckpt_path=None):
        test_loader = self.test_loader

        if ckpt_path is None:
            ckpt_path = self.config.ckpt_path
            
        if os.path.exists(ckpt_path):
            if ckpt_path:
                state_dict = torch.load(ckpt_path)
                visual_proj_weight = state_dict.get('visual_proj.weight', None)
                if visual_proj_weight is not None:
                    old_visual_dim = visual_proj_weight.shape[1]
                    if old_visual_dim != self.config.model_config['visual_dim']:
                        print(f"Warning: Checkpoint was trained with visual_dim={old_visual_dim}, "
                              f"but current config uses visual_dim={self.config.model_config['visual_dim']}")
                self.model.load_state_dict(state_dict)
            print(f"Loaded model from {ckpt_path}")
        else:
            print(f"Warning: No checkpoint found at {ckpt_path}, using current model state")

        print(f"\nEvaluating model on test set...")
        metrics = self.evaluate(test_loader)
        
        print("\nTest Results:")
        print(f"Test Loss:           {metrics['loss']:.5f}")
        print(f"Test Top-5 mAP:      {metrics['top5_mAP']:.5f}")
        print(f"Test HIT@1:          {metrics['hit_at_1']:.5f}")
        print(f"Test mAP:            {metrics['mAP']:.5f}")
        print(f"Test F1:             {metrics['F1']:.5f}")
        print(f"Test mAP@15:         {metrics['mAP@15']:.5f}")
        print(f"Test mAP@50:         {metrics['mAP@50']:.5f}")
        print(f"Test Kendall's Tau:  {metrics['kendall_tau']:.4f}")
        print(f"Test Spearman's Rho: {metrics['spearman_rho']:.4f}")

        with open(self.log_path, 'a') as f:
            f.write(f"\nTest Results (using {os.path.basename(ckpt_path) if ckpt_path else 'current model'}):\n")
            f.write(f"Test Loss:           {metrics['loss']:.5f}\n")
            f.write(f"Test Top-5 mAP:      {metrics['top5_mAP']:.5f}\n")
            f.write(f"Test HIT@1:          {metrics['hit_at_1']:.5f}\n")
            f.write(f"Test mAP:            {metrics['mAP']:.5f}\n")
            f.write(f"Test F1:             {metrics['F1']:.5f}\n")
            f.write(f"Test mAP@15:         {metrics['mAP@15']:.5f}\n")
            f.write(f"Test mAP@50:         {metrics['mAP@50']:.5f}\n\n")
            f.write(f"Test Kendall's Tau:  {metrics['kendall_tau']:.4f}\n")
            f.write(f"Test Spearman's Rho: {metrics['spearman_rho']:.4f}\n\n")
            f.flush()
            
        return metrics

    def _save_results_to_log(self):
        with open(self.log_path, 'w') as f:
            f.write("Best Validation Results:\n")
            f.write(f"Best Val mAP:         {self.best_metrics['score']['value']:.5f} @ epoch {self.best_metrics['score']['epoch']}\n")
            f.write(f"Best Val HIT@1:       {self.best_metrics['hit1']['value']:.5f} @ epoch {self.best_metrics['hit1']['epoch']}\n")
            f.write(f"Best Val F1:          {self.best_metrics['f1']['value']:.5f} @ epoch {self.best_metrics['f1']['epoch']}\n")
            f.write(f"Best Val MAP@15:      {self.best_metrics['map15']['value']:.5f} @ epoch {self.best_metrics['map15']['epoch']}\n")
            f.write(f"Best Val MAP@50:      {self.best_metrics['map50']['value']:.5f} @ epoch {self.best_metrics['map50']['epoch']}\n")
            f.write(f"Best Val Kendall's Tau:  {self.best_metrics['kendall_tau']['value']:.4f} @ epoch {self.best_metrics['kendall_tau']['epoch']}\n")
            f.write(f"Best Val Spearman's Rho: {self.best_metrics['spearman_rho']['value']:.4f} @ epoch {self.best_metrics['spearman_rho']['epoch']}\n")

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if len(param.shape) >= 2:
                    if init_type == "normal":
                        nn.init.normal_(param, mean=0.0, std=init_gain)
                    elif init_type == "xavier":
                        nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                    elif init_type == "kaiming":
                        nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                    elif init_type == "orthogonal":
                        nn.init.orthogonal_(param, gain=np.sqrt(2.0))
                    else:
                        raise NotImplementedError(f"Initialization method {init_type} is not implemented.")
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)