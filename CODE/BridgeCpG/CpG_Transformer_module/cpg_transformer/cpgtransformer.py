import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import auroc, accuracy, matthews_corrcoef, specificity, recall, f1
# Ensure blocks.py path is correct
from cpg_transformer.blocks import (MultiDimWindowTransformerLayer,
                                    CnnL2h128, CnnL3h128, RnnL1, JointL2h512)


class CpGEmbedder(nn.Module):
    """CpG site embedding module (unchanged)"""
    def __init__(self, hidden_size, mode='binary'):
        super().__init__()
        if mode == 'binary':
            self.CpG_embed = nn.Embedding(3, hidden_size)  # 0:mask, 1:unmethylated, 2:methylated
            self.forward = self.forward_binary
        elif mode == 'continuous':
            self.CpG_embed_linear = nn.Linear(1, hidden_size)
            self.mask_embed = self._init_mask(nn.Parameter(torch.Tensor(1, hidden_size)))
            self.forward = self.forward_continuous
        
    def forward_binary(self, y):
        # y: (bsz, seqlen, n_cells) → after embedding: (bsz, seqlen, n_cells, hidden_size)
        return self.CpG_embed(y.long())
    
    def forward_continuous(self, y):
        # Embed normalized continuous values, use dedicated embedding for mask positions
        z = self.CpG_embed_linear(y.unsqueeze(-1).to(self.CpG_embed_linear.weight.dtype) - 1)
        if (y == 0).any():
            z[(y == 0)] = self.mask_embed
        return z
    
    def _init_mask(self, mask):
        # Initialize mask embedding
        bound = 1 / mask.size(1)**0.5
        return nn.init.uniform_(mask, -bound, bound)    


class CpGTransformer(pl.LightningModule):
    def __init__(self, 
                 n_cells=1001,
                 segment_size=1024,  # Segment size (matches data module)
                 RF=1001, 
                 n_conv_layers=2, 
                 CNN_do=.0, 
                 DNA_embed_size=32,
                 cell_embed_size=32, 
                 CpG_embed_size=32, 
                 transf_hsz=64, 
                 transf_do=.20,
                 act='relu', 
                 n_transformers=4, 
                 n_heads=8, 
                 head_dim=8, 
                 window=21,
                 mode='axial', 
                 data_mode='binary', 
                 layernorm=True,
                 lr=5e-4, 
                 lr_decay_factor=.90, 
                 warmup_steps=1000,
                 epochmax=100, 
                 y_filename=None, 
                 csv_dir='./validation_results'):
        super().__init__()
        assert (n_conv_layers == 2) or (n_conv_layers == 3), 'Number of conv layers should be 2 or 3.'
        self.save_hyperparameters()  # Automatically save hyperparameters
        
        # Core parameters
        self.RF = RF
        self.RF2 = int((self.RF - 1) / 2)
        self.n_cols = n_cells  # Total number of cells
        self.segment_size = segment_size  # Segment size (used to calculate global row numbers)
        
        # Validation set mask records: stores (chr_name, global_row, cell_id, true_label, pred_prob, pred_bin, pred_logits)
        self.val_mask_records = []
        
        # Network architecture
        # 1. DNA sequence convolutional embedding
        self.CNN = nn.Sequential(
            CnnL2h128(dropout=CNN_do, RF=RF) if n_conv_layers == 2 else CnnL3h128(dropout=CNN_do, RF=RF),
            nn.ReLU(), 
            nn.Linear(128, DNA_embed_size)
        )
        # 2. Cell embedding
        self.cell_embed = nn.Embedding(n_cells, cell_embed_size)
        # 3. CpG site embedding
        self.CpG_embed = CpGEmbedder(CpG_embed_size, mode=data_mode)
        # 4. Embedding fusion
        self.combine_embeds = nn.Sequential(
            nn.Linear(cell_embed_size + CpG_embed_size + DNA_embed_size, transf_hsz), 
            nn.ReLU()
        )
        # 5. Transformer layers
        TF_layers = [
            MultiDimWindowTransformerLayer(
                transf_hsz, head_dim, n_heads, transf_hsz * 4,
                dropout=transf_do, window=window, activation=act,
                layernorm=layernorm, mode=mode
            ) for _ in range(n_transformers)
        ]
        self.transformer = nn.Sequential(*TF_layers)
        # 6. Output head
        self.output_head = nn.Linear(transf_hsz, 1)
        
        # Result saving configuration
        self.y_filename = y_filename
        self.csv_dir = csv_dir
        os.makedirs(self.csv_dir, exist_ok=True)  # Ensure save directory exists

    def process_batch(self, batch):
        """Process batch data: adapt to chromosome name input for validation set"""
        # Validation set batch: (x_windows, y_orig, y_masked, pos, nonzeros, cell_indices, chr_name)
        # Train/test set batch: (x_windows, y_orig, y_masked, pos, nonzeros, cell_indices)
        if len(batch) == 7:
            # Validation set: includes chromosome name
            x, y_orig, y_masked, pos, ind_train, cell_indices, chr_name = batch
            # Adapt to model input format (add chr_name to inputs)
            inputs = (x, y_masked, pos, cell_indices, chr_name)
        else:
            # Train/test set: no chromosome name
            x, y_orig, y_masked, pos, ind_train, cell_indices = batch
            inputs = (x, y_masked, pos, cell_indices, None)  # Set chr_name to None
        
        # Data type conversion
        x = x.to(torch.long)
        y_orig = y_orig.to(self.dtype) if y_orig.numel() > 0 else torch.tensor([]).to(self.dtype)
        pos = pos.to(torch.long) if pos.numel() > 0 else torch.tensor([]).to(self.dtype)
        
        return inputs, (y_orig, ind_train)

    def forward(self, x, y_masked, pos, cell_indices, chr_name=None):
        """Forward propagation: chr_name only used for validation set, not involved in computation"""
        if x.numel() == 0 or y_masked.numel() == 0:
            return torch.tensor([]).to(self.device)
        
        bsz, seqlen, n_cells = y_masked.shape[:3]
        # 1. DNA sequence embedding: (bsz, seqlen, RF) → (bsz, seqlen, DNA_embed_size)
        DNA_embed = self.CNN(x.view(-1, self.RF)).view(bsz, seqlen, -1)
        # 2. Cell embedding: (n_cells, cell_embed_size) → (bsz, seqlen, n_cells, cell_embed_size)
        cell_embed = self.cell_embed(cell_indices).unsqueeze(1).expand(-1, seqlen, -1, -1)
        # 3. CpG site embedding: (bsz, seqlen, n_cells) → (bsz, seqlen, n_cells, CpG_embed_size)
        CpG_embed = self.CpG_embed(y_masked)
        # 4. Expand DNA embedding dimension: (bsz, seqlen, DNA_embed_size) → (bsz, seqlen, n_cells, DNA_embed_size)
        DNA_embed = DNA_embed.unsqueeze(-2).expand(-1, -1, n_cells, -1)
        # 5. Fusion embedding: (bsz, seqlen, n_cells, transf_hsz)
        x_combined = torch.cat((CpG_embed, cell_embed, DNA_embed), dim=-1)
        x_combined = self.combine_embeds(x_combined)
        # 6. Transformer encoding
        x_transformed, _ = self.transformer((x_combined, pos))
        # 7. Output prediction: (bsz, seqlen, n_cells)
        return self.output_head(x_transformed).squeeze(-1)

    def training_step(self, batch, batch_idx):
        """Training step (no chromosome processing)"""
        inputs, (y_true, ind_train) = self.process_batch(batch)
        y_pred = self(*inputs)

        # Filter invalid data
        if y_pred.numel() == 0 or y_true.numel() == 0 or ind_train.numel() == 0:
            loss = torch.tensor(0.0, device=self.device)
            self.log('train_loss', loss, sync_dist=True)
            return loss

        # Extract predictions and true values for masked sites
        y_pred_masked = torch.diagonal(y_pred[:, ind_train[:, :, 0], ind_train[:, :, 1]]).reshape(-1)
        y_true_masked = torch.diagonal(y_true[:, ind_train[:, :, 0], ind_train[:, :, 1]]).reshape(-1)
        y_true_masked = y_true_masked - 1  # Label normalization: 1→0 (unmethylated), 2→1 (methylated)

        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(y_pred_masked, y_true_masked) if self.hparams.data_mode == 'binary' else F.mse_loss(y_pred_masked, y_true_masked)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step: record chromosome information and masked site results"""
        inputs, (y_true, ind_train) = self.process_batch(batch)
        x, y_masked, pos, cell_indices, chr_name = inputs  # Extract chromosome name
        y_pred = self(*inputs)

        # Filter invalid data
        if y_pred.numel() == 0 or y_true.numel() == 0 or ind_train.numel() == 0 or chr_name is None:
            loss = torch.tensor(0.0, device=self.device)
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return {'val_loss': loss}

        # Core parameter parsing
        bsz, seqlen, n_cells = y_masked.shape[:3]
        segment_idx = batch_idx  # Global index of current segment (validation set segments are ordered)
        local_rows = ind_train[:, :, 0]  # Site indices within segment (local row numbers)
        cell_ids = ind_train[:, :, 1]    # Cell IDs (column numbers)

        # 1. Calculate global row numbers: global_row = segment index × segment size + local row number within segment
        global_row_ids = (segment_idx * self.segment_size + local_rows).detach().cpu().numpy()

        # 2. Extract predictions and true values for masked sites
        y_pred_masked = torch.diagonal(y_pred[:, local_rows, cell_ids]).reshape(-1)
        y_true_masked = torch.diagonal(y_true[:, local_rows, cell_ids]).reshape(-1)
        y_true_masked = y_true_masked - 1  # Label normalization

        # 3. Convert to numpy format (for easy recording)
        y_true_np = y_true_masked.detach().cpu().numpy()
        y_pred_logits_np = y_pred_masked.detach().cpu().numpy()
        y_pred_prob_np = torch.sigmoid(y_pred_masked).detach().cpu().numpy()
        y_pred_bin_np = (y_pred_prob_np >= 0.5).astype(np.int32)

        # 4. Record information for each masked site (including chromosome name)
        batch_size, num_masks, _ = ind_train.shape if ind_train.ndim == 3 else (1, ind_train.shape[0], 2)
        for b in range(batch_size):
            current_chr = chr_name[b] if isinstance(chr_name, (list, torch.Tensor)) else chr_name  # Adapt to batch chromosome names
            for i in range(num_masks):
                # Calculate index of current site in global array
                idx = i if batch_size == 1 else b * num_masks + i
                if idx >= len(y_true_np):
                    continue  # Avoid index out of bounds
                
                # Extract single site information
                g_row = global_row_ids[b, i] if (global_row_ids.ndim == 2 and b < global_row_ids.shape[0] and i < global_row_ids.shape[1]) else global_row_ids[i]
                cid = cell_ids[b, i].item() if (cell_ids.ndim == 3 and b < cell_ids.shape[0] and i < cell_ids.shape[1]) else cell_ids[i].item()
                t = y_true_np[idx]
                p_prob = y_pred_prob_np[idx]
                p_bin = y_pred_bin_np[idx]
                p_logits = y_pred_logits_np[idx]

                # Filter invalid labels (only keep 0/1)
                if not np.isnan(t) and t in [0, 1]:
                    self.val_mask_records.append((current_chr, g_row, cid, t, p_prob, p_bin, p_logits))

        # 5. Calculate validation loss
        loss = F.binary_cross_entropy_with_logits(y_pred_masked, y_true_masked) if self.hparams.data_mode == 'binary' else F.mse_loss(y_pred_masked, y_true_masked)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        """End of validation epoch: split and save results by chromosome"""
        if not self.val_mask_records:
            print("Validation epoch {}: No mask site data collected".format(self.current_epoch))
            return

        current_epoch = self.current_epoch
        data_mode = self.hparams.data_mode

        # 1. Group records by chromosome
        chr_records = {}
        for rec in self.val_mask_records:
            chr_name = rec[0]
            if chr_name not in chr_records:
                chr_records[chr_name] = []
            chr_records[chr_name].append(rec)

        print(f"\nValidation epoch {current_epoch} summary:")
        print(f"  Involved chromosomes: {list(chr_records.keys())}")
        print(f"  Total valid records: {len(self.val_mask_records)}")

        # 2. Split and save NPZ by chromosome
        for chr_name, records in chr_records.items():
            # Filter invalid records for current chromosome
            valid_records = [r for r in records if r[3] in [0, 1]]  # r[3] is true label t
            if not valid_records:
                print(f"  Chromosome {chr_name}: No valid records, skipping save")
                continue

            # Parse site information for current chromosome
            all_g_rows = [r[1] for r in valid_records]  # Global row numbers
            all_cids = [r[2] for r in valid_records]    # Cell IDs
            max_g_row = max(all_g_rows) if all_g_rows else 0
            mat_rows = max_g_row + 1  # Row numbers start from 0, matrix rows = max row number + 1
            mat_cols = self.n_cols     # Matrix columns = total number of cells
            valid_count = len(valid_records)

            print(f"  Chromosome {chr_name}:")
            print(f"    - Valid records: {valid_count}")
            print(f"    - Matrix dimensions: [{mat_rows}, {mat_cols}]")

            # Initialize result matrices (fill empty positions with NaN)
            y_true_mat = np.full((mat_rows, mat_cols), np.nan, dtype=np.float32)       # True labels
            y_pred_prob_mat = np.full((mat_rows, mat_cols), np.nan, dtype=np.float32)  # Predicted probabilities
            y_pred_bin_mat = np.full((mat_rows, mat_cols), np.nan, dtype=np.int32)     # Binarized predictions
            y_pred_logits_mat = np.full((mat_rows, mat_cols), np.nan, dtype=np.float32)# Predicted logits

            # Fill matrices
            for rec in valid_records:
                _, g_row, cid, t, p_prob, p_bin, p_logits = rec
                if 0 <= g_row < mat_rows and 0 <= cid < mat_cols:  # Ensure valid indices
                    y_true_mat[g_row, cid] = t
                    y_pred_prob_mat[g_row, cid] = p_prob
                    y_pred_bin_mat[g_row, cid] = p_bin
                    y_pred_logits_mat[g_row, cid] = p_logits

            # 3. Save NPZ file (including chromosome name and epoch information)
            y_basename = os.path.basename(self.y_filename) if self.y_filename else "unknown_y"
            npz_filename = f"val_{chr_name}_{y_basename}_epoch{current_epoch}.npz"
            npz_save_path = os.path.join(self.csv_dir, npz_filename)

            # Save matrices and meta information
            np.savez(
                npz_save_path,
                chr_name=chr_name,                # Chromosome name
                epoch=current_epoch,              # Current epoch
                segment_size=self.segment_size,   # Segment size
                max_global_row=max_g_row,         # Maximum global row number
                valid_count=valid_count,          # Number of valid sites
                y_true=y_true_mat,                # True label matrix
                y_pred_prob=y_pred_prob_mat,      # Predicted probability matrix
                y_pred_bin=y_pred_bin_mat,        # Binarized prediction matrix
                y_pred_logits=y_pred_logits_mat   # Predicted logits matrix
            )

            print(f"  Saved to: {npz_save_path}")

        # Reset record list (avoid accumulation in next round)
        self.val_mask_records.clear()
        print("\n" + "-"*50 + "\n")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Learning rate decay
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambda epoch: self.hparams.lr_decay_factor
        )
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        """Learning rate warmup"""
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        optimizer.step(closure=optimizer_closure)

    def n_params(self):
        """Calculate number of model parameters"""
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer.append(('total', total_params))
        # Print parameter statistics
        print("\nModel parameter statistics:")
        for name, num in params_per_layer:
            print(f"  {name}: {num:,}")
        print(f"Total parameters: {total_params:,}")
        return params_per_layer