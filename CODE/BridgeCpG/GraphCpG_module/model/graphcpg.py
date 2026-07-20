import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, RGCNConv, global_add_pool
from torch_geometric.utils import dropout_adj
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, MatthewsCorrCoef, Specificity, Recall, F1Score
import numpy as np
from tabulate import tabulate
import os
import csv
import pandas as pd


class GNN(pl.LightningModule):
    def __init__(self, gconv=GCNConv, latent_dim=[32, 32, 32, 1],
                 adj_dropout=0.2, force_undirected=False):
        super(GNN, self).__init__()
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected
        self.num_classes = 2
        self.num_node_features = 4
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(self.num_node_features, latent_dim[0]))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))
        self.lin1 = Linear(sum(latent_dim), 128)
        self.lin2 = Linear(128, self.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type=None, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class CpGGraph(GNN):
    def __init__(self, gconv=RGCNConv, latent_dim=[32, 64, 128, 128, 64, 32],
                 num_relations=2, num_bases=2, adj_dropout=0.2,
                 force_undirected=False,
                 cell_num=25, segment_size=21,
                 multiply_by=1, lr=1e-3, lr_decay_factor=.90, warmup_steps=1000,
                 csv_dir='npz_MK/'):
        super(CpGGraph, self).__init__(
            GCNConv, latent_dim, adj_dropout, force_undirected
        )
        self.segment_size = segment_size
        self.cell_num = cell_num
        self.sum_latent_dim = sum(latent_dim)
        self.save_hyperparameters()
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(self.segment_size + 1, latent_dim[0], num_relations, num_bases))

        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], num_relations, num_bases))

        self.CNN2 = nn.Sequential(
            nn.Conv2d(1, 4, (5, 5), stride=(1, 5), padding=(1, 2)), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 1, (3, 3), stride=(1, 3), padding=(1, 1)), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )
        self.lin1_in_size = self.lin1_param(self.segment_size + self.cell_num, self.sum_latent_dim)
        self.lin1 = Linear(self.lin1_in_size, 40)

        self.hparams.lr = lr
        self.hparams.lr_decay_factor = lr_decay_factor
        self.hparams.warmup_steps = warmup_steps
        self.lin2 = nn.Linear(40, 1)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = torch.tanh(x)
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        x = torch.reshape(concat_states, (-1, self.segment_size + self.cell_num, self.sum_latent_dim)).unsqueeze(1)
        x = self.CNN2(x)
        x = torch.reshape(x.squeeze(), (-1, self.lin1_in_size))
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        return self.lin2(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = F.binary_cross_entropy_with_logits(y_hat, data.y.to(torch.float))
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Modified: Move test logic to validation step
        data = batch
        y_hat = self(data)  
        y = data.y.to(torch.float)  
        device = y_hat.device

        y_hat_sigmoid = torch.sigmoid(y_hat)  

        # Calculate validation loss
        val_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        
        # Calculate metrics
        auroc_metric = AUROC(num_classes=2, multiclass=True).to(device)
        auroc_value = auroc_metric(y_hat_sigmoid, y.long())
        
        acc_metric = Accuracy(task='binary').to(device)
        acc_value = acc_metric(y_hat_sigmoid, y.long())
        
        mcc_metric = MatthewsCorrCoef(num_classes=2).to(device)
        mcc_value = mcc_metric(y_hat_sigmoid, y.long())
        
        tnr_metric = Specificity(task='binary').to(device)
        tnr_value = tnr_metric(y_hat_sigmoid, y.long())
        
        tpr_metric = Recall(task='binary').to(device)
        tpr_value = tpr_metric(y_hat_sigmoid, y.long())
        
        f1_metric = F1Score(task='binary').to(device)
        f1_value = f1_metric(y_hat_sigmoid, y.long())

        # Extract m and n (true positions) of samples in current batch
        if hasattr(data, 'm') and hasattr(data, 'n'):
            m = data.m.detach().cpu()
            n = data.n.detach().cpu()
        else:
            m = torch.full_like(y, -1, dtype=torch.long).detach().cpu()
            n = torch.full_like(y, -1, dtype=torch.long).detach().cpu()

        # Return validation loss and all required information
        return {
            'val_loss': val_loss,
            'AUROC': auroc_value,
            'ACC': acc_value,
            'MCC': mcc_value,
            'TNR': tnr_value,
            'TPR': tpr_value,
            'F1': f1_value,
            'y_hat_logits': y_hat.detach().cpu(),
            'y_hat_sigmoid': y_hat_sigmoid.detach().cpu(),
            'y_true': y.detach().cpu(),
            'm_global': m,
            'n_global': n
        }

    def validation_epoch_end(self, validation_step_outputs):
        # Modified: Move test epoch end logic to validation epoch end
        # Aggregate data from all batches
        all_logits = []    
        all_sigmoid = []   
        all_true = []      
        all_m = []         
        all_n = []         
        
        val_losses = []
        auroc_values = []
        acc_values = []
        mcc_values = []
        tnr_values = []
        tpr_values = []
        f1_values = []

        for step_res in validation_step_outputs:
            all_logits.append(step_res['y_hat_logits'].numpy())    
            all_sigmoid.append(step_res['y_hat_sigmoid'].numpy())
            all_true.append(step_res['y_true'].numpy())
            all_m.append(step_res['m_global'].numpy())
            all_n.append(step_res['n_global'].numpy())
            
            val_losses.append(step_res['val_loss'])
            auroc_values.append(step_res['AUROC'])
            acc_values.append(step_res['ACC'])
            mcc_values.append(step_res['MCC'])
            tnr_values.append(step_res['TNR'])
            tpr_values.append(step_res['TPR'])
            f1_values.append(step_res['F1'])

        # Concatenate data from all batches
        y_pred_logits = np.concatenate(all_logits, axis=0)    
        y_pred_sigmoid = np.concatenate(all_sigmoid, axis=0)  
        y_true = np.concatenate(all_true, axis=0)             
        m_global = np.concatenate(all_m, axis=0)              
        n_global = np.concatenate(all_n, axis=0)              

        # Calculate average validation loss and metrics
        avg_val_loss = torch.stack(val_losses).mean()
        avg_auroc = torch.stack(auroc_values).mean()
        avg_acc = torch.stack(acc_values).mean()
        avg_mcc = torch.stack(mcc_values).mean()
        avg_tnr = torch.stack(tnr_values).mean()
        avg_tpr = torch.stack(tpr_values).mean()
        avg_f1 = torch.stack(f1_values).mean()

        # Log validation metrics
        self.log('val_loss', avg_val_loss, sync_dist=True)
        self.log('val_auroc', avg_auroc, sync_dist=True)
        self.log('val_acc', avg_acc, sync_dist=True)
        self.log('val_mcc', avg_mcc, sync_dist=True)
        self.log('val_tnr', avg_tnr, sync_dist=True)
        self.log('val_tpr', avg_tpr, sync_dist=True)
        self.log('val_f1', avg_f1, sync_dist=True)

        # Binarize prediction results
        y_pred_binary = (y_pred_sigmoid > 0.5).astype(np.float32)

        # Save NPZ file (saved every epoch)
        os.makedirs(self.hparams.csv_dir, exist_ok=True)
        current_epoch = self.current_epoch
        npz_filename = os.path.join(self.hparams.csv_dir, f"val_epoch_{current_epoch}_predictions.npz")
        
        np.savez(
            npz_filename,
            y_pred_logits=y_pred_logits,
            y_pred_sigmoid=y_pred_sigmoid,
            y_pred_binary=y_pred_binary,
            y_true=y_true,
            m_global=m_global,
            n_global=n_global
        )
        
        print(f"\nValidation results for epoch {current_epoch} saved to: {npz_filename}")
        print(f"Data scale: Total samples = {len(y_true)}")
        print(f"True position range: m (row index) = {m_global.min()}~{m_global.max()}, n (column index) = {n_global.min()}~{n_global.max()}")
        print(f"Validation metrics - Loss: {avg_val_loss:.4f}, AUROC: {avg_auroc:.4f}, ACC: {avg_acc:.4f}")

    def test_step(self, batch, batch_idx):
        # Keep the original test step, but it can be simplified or kept as is
        data = batch
        y_hat = self(data)  
        y = data.y.to(torch.float)  
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        
        # Test-specific logic can be added here
        return {'test_loss': loss}

    def test_epoch_end(self, test_step_outputs):
        # Test epoch end logic can be simplified since the main logic is implemented in validation
        test_loss = torch.stack([x['test_loss'] for x in test_step_outputs]).mean()
        self.log('test_loss', test_loss)
        print(f"Test loss: {test_loss:.4f}")
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [lr_scheduler]

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

    def lin1_param(self, H_in, W_in):
        H_out_1conv = math.floor((H_in + 2 * 1 - 1 * (5 - 1) - 1) / 1 + 1)
        H_out_1pool = math.floor((H_out_1conv + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        H_out_2conv = math.floor((H_out_1pool + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1)
        H_out_2pool = math.floor((H_out_2conv + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)

        W_out_1conv = math.floor((W_in + 2 * 2 - 1 * (5 - 1) - 1) / 5 + 1)
        W_out_1pool = math.floor((W_out_1conv + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        W_out_2conv = math.floor((W_out_1pool + 2 * 1 - 1 * (3 - 1) - 1) / 3 + 1)
        W_out_2pool = math.floor((W_out_2conv + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        return H_out_2pool * W_out_2pool


class CpGGraphAnalysis(GNN):
    def __init__(self, gconv=RGCNConv, latent_dim=[32, 64, 128, 128, 64, 32],
                 num_relations=2, num_bases=2, adj_dropout=0.2,
                 force_undirected=False,
                 cell_num=25, segment_size=21,
                 multiply_by=1, lr=1e-3, lr_decay_factor=.90, warmup_steps=1000,
                 csv_dir='./'):
        super(CpGGraphAnalysis, self).__init__(
            GCNConv, latent_dim, adj_dropout, force_undirected
        )
        self.segment_size = segment_size
        self.cell_num = cell_num
        self.sum_latent_dim = sum(latent_dim)
        self.save_hyperparameters()
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(self.segment_size + 1, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], num_relations, num_bases))

        self.CNN2 = nn.Sequential(
            nn.Conv2d(1, 4, (5, 5), stride=(1, 5), padding=(1, 2)), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 1, (3, 3), stride=(1, 3), padding=(1, 1)), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )
        self.lin1_in_size = self.lin1_param(self.segment_size + self.cell_num, self.sum_latent_dim)
        self.lin1 = Linear(self.lin1_in_size, 40)
        self.hparams.lr = lr
        self.hparams.lr_decay_factor = lr_decay_factor
        self.hparams.warmup_steps = warmup_steps
        self.lin2 = nn.Linear(40, 1)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = torch.tanh(x)
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = torch.reshape(concat_states, (-1, self.segment_size + self.cell_num, self.sum_latent_dim)).unsqueeze(1)
        return x

    def lin1_param(self, H_in, W_in):
        H_out_1conv = math.floor((H_in + 2 * 1 - 1 * (5 - 1) - 1) / 1 + 1)
        H_out_1pool = math.floor((H_out_1conv + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        H_out_2conv = math.floor((H_out_1pool + 2 * 1 - 1 * (3 - 1) - 1) / 1 + 1)
        H_out_2pool = math.floor((H_out_2conv + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        W_out_1conv = math.floor((W_in + 2 * 2 - 1 * (5 - 1) - 1) / 5 + 1)
        W_out_1pool = math.floor((W_out_1conv + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        W_out_2conv = math.floor((W_out_1pool + 2 * 1 - 1 * (3 - 1) - 1) / 3 + 1)
        W_out_2pool = math.floor((W_out_2conv + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        return H_out_2pool * W_out_2pool

    # Modified: Move test logic to validation step (same modification as CpGGraph class)
    def training_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = F.binary_cross_entropy_with_logits(y_hat, data.y.to(torch.float))
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Move test logic to validation step
        data = batch
        y_hat = self(data)  
        y = data.y.to(torch.float)  
        device = y_hat.device

        y_hat_sigmoid = torch.sigmoid(y_hat)  

        # Calculate validation loss
        val_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        
        # Calculate metrics
        auroc_metric = AUROC(num_classes=2, multiclass=True).to(device)
        auroc_value = auroc_metric(y_hat_sigmoid, y.long())
        
        acc_metric = Accuracy(task='binary').to(device)
        acc_value = acc_metric(y_hat_sigmoid, y.long())
        
        mcc_metric = MatthewsCorrCoef(num_classes=2).to(device)
        mcc_value = mcc_metric(y_hat_sigmoid, y.long())
        
        tnr_metric = Specificity(task='binary').to(device)
        tnr_value = tnr_metric(y_hat_sigmoid, y.long())
        
        tpr_metric = Recall(task='binary').to(device)
        tpr_value = tpr_metric(y_hat_sigmoid, y.long())
        
        f1_metric = F1Score(task='binary').to(device)
        f1_value = f1_metric(y_hat_sigmoid, y.long())

        # Extract m and n (true positions) of samples in current batch
        if hasattr(data, 'm') and hasattr(data, 'n'):
            m = data.m.detach().cpu()
            n = data.n.detach().cpu()
        else:
            m = torch.full_like(y, -1, dtype=torch.long).detach().cpu()
            n = torch.full_like(y, -1, dtype=torch.long).detach().cpu()

        return {
            'val_loss': val_loss,
            'AUROC': auroc_value,
            'ACC': acc_value,
            'MCC': mcc_value,
            'TNR': tnr_value,
            'TPR': tpr_value,
            'F1': f1_value,
            'y_hat_logits': y_hat.detach().cpu(),
            'y_hat_sigmoid': y_hat_sigmoid.detach().cpu(),
            'y_true': y.detach().cpu(),
            'm_global': m,
            'n_global': n
        }

    def validation_epoch_end(self, validation_step_outputs):
        # Move test epoch end logic to validation epoch end
        all_logits = []    
        all_sigmoid = []   
        all_true = []      
        all_m = []         
        all_n = []         
        
        val_losses = []
        auroc_values = []
        acc_values = []
        mcc_values = []
        tnr_values = []
        tpr_values = []
        f1_values = []

        for step_res in validation_step_outputs:
            all_logits.append(step_res['y_hat_logits'].numpy())    
            all_sigmoid.append(step_res['y_hat_sigmoid'].numpy())
            all_true.append(step_res['y_true'].numpy())
            all_m.append(step_res['m_global'].numpy())
            all_n.append(step_res['n_global'].numpy())
            
            val_losses.append(step_res['val_loss'])
            auroc_values.append(step_res['AUROC'])
            acc_values.append(step_res['ACC'])
            mcc_values.append(step_res['MCC'])
            tnr_values.append(step_res['TNR'])
            tpr_values.append(step_res['TPR'])
            f1_values.append(step_res['F1'])

        # Concatenate data from all batches
        y_pred_logits = np.concatenate(all_logits, axis=0)    
        y_pred_sigmoid = np.concatenate(all_sigmoid, axis=0)  
        y_true = np.concatenate(all_true, axis=0)             
        m_global = np.concatenate(all_m, axis=0)              
        n_global = np.concatenate(all_n, axis=0)              

        # Calculate average validation loss and metrics
        avg_val_loss = torch.stack(val_losses).mean()
        avg_auroc = torch.stack(auroc_values).mean()
        avg_acc = torch.stack(acc_values).mean()
        avg_mcc = torch.stack(mcc_values).mean()
        avg_tnr = torch.stack(tnr_values).mean()
        avg_tpr = torch.stack(tpr_values).mean()
        avg_f1 = torch.stack(f1_values).mean()

        # Log validation metrics
        self.log('val_loss', avg_val_loss, sync_dist=True)
        self.log('val_auroc', avg_auroc, sync_dist=True)
        self.log('val_acc', avg_acc, sync_dist=True)
        self.log('val_mcc', avg_mcc, sync_dist=True)
        self.log('val_tnr', avg_tnr, sync_dist=True)
        self.log('val_tpr', avg_tpr, sync_dist=True)
        self.log('val_f1', avg_f1, sync_dist=True)

        # Binarize prediction results
        y_pred_binary = (y_pred_sigmoid > 0.5).astype(np.float32)

        # Save NPZ file (saved every epoch)
        os.makedirs(self.hparams.csv_dir, exist_ok=True)
        current_epoch = self.current_epoch
        npz_filename = os.path.join(self.hparams.csv_dir, f"val_epoch_{current_epoch}_predictions_mn.npz")
        
        np.savez(
            npz_filename,
            y_pred_logits=y_pred_logits,
            y_pred_sigmoid=y_pred_sigmoid,
            y_pred_binary=y_pred_binary,
            y_true=y_true,
            m_global=m_global,
            n_global=n_global
        )
        
        print(f"\nValidation results for epoch {current_epoch} saved to: {npz_filename}")
        print(f"Data scale: Total samples = {len(y_true)}")
        print(f"True position range: m (row index) = {m_global.min()}~{m_global.max()}, n (column index) = {n_global.min()}~{n_global.max()}")
        print(f"Validation metrics - Loss: {avg_val_loss:.4f}, AUROC: {avg_auroc:.4f}, ACC: {avg_acc:.4f}")

        # Keep the original CSV saving logic (optional)
        all_results = []
        for i, result in enumerate(validation_step_outputs):
            current_results = {
                'Graph Index': i,
                'loss': np.round(result['val_loss'].tolist(), 2),
                'AUROC': np.round(result['AUROC'].tolist() * 100, 2),
                'ACC': np.round(result['ACC'].tolist() * 100, 2),
                'MCC': np.round(result['MCC'].tolist() * 100, 2),
                'TNR': np.round(result['TNR'].tolist() * 100, 2),
                'TPR': np.round(result['TPR'].tolist() * 100, 2),
                'F1': np.round(result['F1'].tolist() * 100, 2)
            }
            all_results.append(current_results)

        # Save metrics CSV
        headers = ['Graph Index', 'loss', 'AUROC', 'ACC', 'MCC', 'TNR', 'TPR', 'F1']
        csv_filename = os.path.join(self.hparams.csv_dir, f'val_epoch_{current_epoch}_metrics.csv')
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows([list(res.values()) for res in all_results])
    
    def test_step(self, batch, batch_idx):
        # Simplify test step
        data = batch
        y_hat = self(data)  
        y = data.y.to(torch.float)  
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return {'test_loss': loss}

    def test_epoch_end(self, test_step_outputs):
        # Simplify test epoch end logic
        test_loss = torch.stack([x['test_loss'] for x in test_step_outputs]).mean()
        self.log('test_loss', test_loss)
        print(f"Final test loss: {test_loss:.4f}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [lr_scheduler]

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