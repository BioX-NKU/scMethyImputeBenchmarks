import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
from collections import defaultdict
from blocks import BiMamba, CnnL2h128
import torch.optim.lr_scheduler as lr_scheduler


class MambaCpG(pl.LightningModule):
    def __init__(self, ncell, segment_size, dim=32, nmamba=4, lr=3e-3, warmup_steps=1000, 
                 total_epochs=35, epochmax=None, npz_dir=None, y_filename=None):
        super().__init__()
        self.ncell = ncell
        self.segment_size = segment_size
        self.BiMamba = BiMamba(ncell, dim, nmamba)
        self.CNN = nn.Sequential(CnnL2h128(dropout=0, RF=1001), nn.ReLU(), nn.Linear(128,32))
        self.fc = nn.Linear(2 * dim, 1)
        self.hparams.lr = lr
        self.hparams.warmup_steps = warmup_steps
        self.hparams.total_epochs = total_epochs
        self.hparams.epochmax = epochmax if epochmax is not None else total_epochs
        self.save_hyperparameters()
        
        self.npz_dir = npz_dir if npz_dir is not None else os.getcwd()
        self.y_filename = y_filename if y_filename else "unknown_data"
        
        self.val_mask_records = []

    def process_batch(self, batch):
        y, y_masked, mask_indices, cell_indices, x = batch
        y = y.to(torch.float).to(self.device)
        x = x.to(torch.long).to(self.device)
        y_masked = y_masked.to(self.device)
        mask_indices = mask_indices.to(self.device)
        cell_indices = cell_indices.to(self.device)
        return (x, y_masked, cell_indices), y, mask_indices
    
    def forward(self, x, y_masked, cell_indices):
        x = self.CNN(x.view(-1, 1001))
        Mambaout = self.BiMamba(x, y_masked, cell_indices)
        return self.fc(Mambaout).squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        input, y, mask_indices = self.process_batch(batch)
        y_hat = self(*input)
        y_hat_masked = torch.diagonal(y_hat[:, mask_indices[:, :, 0], mask_indices[:, :, 1]]).reshape(-1)
        y_flat = y.reshape(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat_masked, y_flat)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss  

    def validation_step(self, batch, batch_idx):
        input, y, mask_indices = self.process_batch(batch)
        y_hat = self(*input)
        
        segment_idx = batch_idx
        local_rows = mask_indices[:, :, 0].reshape(-1)
        cell_ids = mask_indices[:, :, 1].reshape(-1)
        
        global_row_ids = (segment_idx * self.segment_size + local_rows).detach().cpu().numpy()
        
        y_hat_masked = torch.diagonal(y_hat[:, mask_indices[:, :, 0], mask_indices[:, :, 1]]).reshape(-1)
        y_true = y.reshape(-1).detach().cpu().numpy()
        y_pred_prob = torch.sigmoid(y_hat_masked).detach().cpu().numpy()
        y_pred_bin = (y_pred_prob >= 0.5).astype(np.int32)
        
        for g_row, cid, t, p_prob, p_bin in zip(global_row_ids, cell_ids, y_true, y_pred_prob, y_pred_bin):
            self.val_mask_records.append( (g_row, cid, t, p_prob, p_bin) )
        
        loss = F.binary_cross_entropy_with_logits(y_hat_masked, y.reshape(-1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        if not self.val_mask_records:
            print("\nNo mask site data collected, skipping npz save")
            return
        
        all_global_rows = [rec[0] for rec in self.val_mask_records]
        max_global_row = max(all_global_rows)
        mat_rows = max_global_row + 1
        mat_cols = self.ncell
        
        y_true_mat = np.full((mat_rows, mat_cols), np.nan, dtype=np.float32)
        y_pred_prob_mat = np.full((mat_rows, mat_cols), np.nan, dtype=np.float32)
        y_pred_bin_mat = np.full((mat_rows, mat_cols), np.nan, dtype=np.int32)
        
        for g_row, cid, t, p_prob, p_bin in self.val_mask_records:
            if 0 <= cid < mat_cols:
                y_true_mat[g_row, cid] = t
                y_pred_prob_mat[g_row, cid] = p_prob
                y_pred_bin_mat[g_row, cid] = p_bin
        
        valid_count = np.sum(~np.isnan(y_true_mat))
        
        current_epoch = self.current_epoch
        y_basename = os.path.basename(self.y_filename) if os.path.exists(self.y_filename) else self.y_filename
        npz_filename = f"val_true_pred_{y_basename}_epoch{current_epoch}.npz"
        npz_save_path = os.path.join(self.npz_dir, npz_filename)
        
        if self.trainer.is_global_zero:
            np.savez(
                npz_save_path,
                y_true=y_true_mat,
                y_pred_prob=y_pred_prob_mat,
                y_pred_bin=y_pred_bin_mat,
                segment_size=self.segment_size,
                max_global_row=max_global_row
            )
            
            print(f"\nValidation set matrix saved to: {npz_save_path}")
            print(f"Matrix specifications:")
            print(f"   - Dimensions: [number of global row identifiers, total number of cells] = [{mat_rows}, {mat_cols}]")
            print(f"   - Global row identifier range: 0 ~ {max_global_row} (calculation logic: segment index×{self.segment_size}+local site)")
            print(f"   - Total number of valid mask sites: {valid_count}")
            print(f"   - Column index meaning: directly corresponds to original cell ID (0 ~ {mat_cols-1})")
        
        self.val_mask_records.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.95, 0.9), weight_decay=0.01)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.total_epochs)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        optimizer.step(closure=optimizer_closure)
    
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer