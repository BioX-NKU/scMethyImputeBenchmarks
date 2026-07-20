import os
import csv
import torch
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler
import math
import pandas as pd


def sample_from_cdf(cdf, n):
    return cdf['x'][(torch.rand(n, 1) < cdf['y']).int().argmax(-1)]


def generate_fixed_masks_local(dataset, val_by_chr, val_segment_range, total_mask_count, mask_random_percentage, seed=42):
    """Generate fixed masks with chromosome identifiers (adapted for multiple validation chromosomes)"""
    fixed_masks = []  # Original format: mask list by global segment index (compatible with subsequent use)
    fixed_masks_by_chr = {}  # New format: {chr_name: [masks for each segment of the chromosome], ...}
    original_seed = torch.initial_seed()
    
    # Generate masks for each chromosome separately
    for chr_name in val_by_chr.keys():
        chr_segments = val_by_chr[chr_name]
        chr_mask_list = []
        chr_start_idx, chr_end_idx = val_segment_range[chr_name]
        chr_total_segments = len(chr_segments)
        
        # Count total valid sites for this chromosome
        chr_nonzero_counts = []
        for seg in chr_segments:
            x_seg, y_seg, ind_seg, pos_seg = seg
            y_orig_seg = y_seg + 1
            chr_nonzero_counts.append(y_orig_seg.nonzero(as_tuple=False).size(0))
        chr_total_nonzeros = sum(chr_nonzero_counts)
        total_nonzeros_all = sum([sum((seg[1]+1).nonzero().size(0) for seg in val_by_chr[cn]) for cn in val_by_chr.keys()])
        
        # Allocate mask count proportionally
        chr_mask_count = int(round(chr_total_nonzeros / total_nonzeros_all * total_mask_count))
        print(f"\nChromosome {chr_name}: valid sites {chr_total_nonzeros}, assigned mask count {chr_mask_count}")
        
        # Generate masks for each segment
        for seg_idx in range(chr_total_segments):
            torch.manual_seed(seed + chr_start_idx + seg_idx)
            random.seed(seed + chr_start_idx + seg_idx)
            
            x, y, ind, pos = chr_segments[seg_idx]
            y_orig = y + 1
            nonzeros = y_orig.nonzero(as_tuple=False)
            seg_mask_count = int(round(chr_nonzero_counts[seg_idx] / chr_total_nonzeros * chr_mask_count))
            seg_mask_count = min(seg_mask_count, nonzeros.size(0))
            
            if seg_mask_count > 0:
                perm = torch.randperm(nonzeros.size(0))[:seg_mask_count]
                mask_indices = nonzeros[perm]
            else:
                mask_indices = torch.empty((0, 2), dtype=torch.long)
            
            chr_mask_list.append(mask_indices)
            fixed_masks.append(mask_indices)
        
        fixed_masks_by_chr[chr_name] = chr_mask_list
    
    torch.manual_seed(original_seed)
    random.seed(original_seed)
    return fixed_masks, fixed_masks_by_chr


class CpGTransformerDataModule(pl.LightningDataModule):
    def __init__(self, X, y, pos, segment_size=1024, RF=1001, fracs=[1,0,0],
                 mask_perc=0.25, mask_random_perc=0.2,
                 resample_cells=None, resample_cells_val=None,
                 val_keys=None, test_keys=None,
                 batch_size=1, n_workers=4, fixed_masks_path=None):
        
        assert len(fracs)==3,'length of fractions should be 3 for train/val/test'
        assert sum(fracs)==1, 'Sum of train/val/test fractions should be one.'
        assert val_keys is None or type(val_keys) is list, 'val_keys should be None or list'
        assert test_keys is None or type(test_keys) is list, 'test_keys should be None or list'
        if val_keys is not None and test_keys is not None:
            assert set(val_keys) & set(test_keys) == set(), 'No overlap allowed between val_keys & test_keys'
        super().__init__()
        
        self.X = X
        self.y = y
        self.pos = pos
        self.segment_size = segment_size
        self.RF = RF; self.RF2 = int((RF-1)/2)
        self.fracs = fracs
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.mask_perc = mask_perc
        self.mask_random_perc = mask_random_perc
        self.bsz = batch_size
        self.nw = n_workers
        self.resample = resample_cells
        self.resample_val = resample_cells_val
        self.fixed_masks_path = fixed_masks_path
        
        # Statistical variables
        self.global_stats = {
            "total_original_sites": 0,
            "total_segmented_sites": 0,
            "chromosome_stats": {}
        }
        
    def setup(self, stage=None):
        train = []; test = []
        # Key modification 1: Store validation set grouped by chromosome (record segment-chromosome mapping)
        val_by_chr = {}  # {chr_name: [list of segments for the chromosome (each segment: (X,y,ind,pos))], ...}
        val_segment_range = {}  # {chr_name: (start global index of segments, end global index of segments), ...}
        global_val_index = 0  # Global index of validation set segments (cross chromosomes)

        for chr_name in self.y.keys():
            # Record original site count
            original_pos = self.pos[chr_name]
            original_pos_count = len(original_pos) if 'numpy' in str(type(original_pos)) else original_pos.size(0)
            self.global_stats["total_original_sites"] += original_pos_count
            self.global_stats["chromosome_stats"][chr_name] = {
                "original_sites": original_pos_count,
                "filtered_sites": 0,
                "segments": 0,
                "segmented_sites": 0
            }
            print(f"\nProcessing chromosome {chr_name}:")
            print(f"  Original site count before segmentation: {original_pos_count}")
            
            # Load data and preprocess
            y_temp = self.y[chr_name]
            X_temp = self.X[chr_name]
            pos_temp = self.pos[chr_name]
            
            if 'numpy' in str(type(X_temp)):
                X_temp = torch.from_numpy(X_temp)
                y_temp = torch.from_numpy(y_temp)
                pos_temp = torch.from_numpy(pos_temp)
                
            # Pad edges
            X_temp = torch.cat((torch.full((self.RF2,),4, dtype=torch.int8), X_temp,
                                torch.full((self.RF2,),4, dtype=torch.int8)))
            pos_temp = pos_temp.clone() + self.RF2

            # Filter gaps
            mask = torch.ones_like(X_temp, dtype=torch.bool)
            for e, b in zip(pos_temp[1:][pos_temp[1:] - pos_temp[:-1] > self.RF],
                            pos_temp[:-1][pos_temp[1:] - pos_temp[:-1] > self.RF]):
                T = e - self.RF2
                mask[torch.arange(b + self.RF2 + 1, T)] = False

            # Mark valid CpG sites
            tmp = torch.zeros_like(X_temp, dtype=torch.int8)
            tmp[pos_temp.to(torch.long)] = 1
            tmp = tmp[mask]
            indices = torch.where(tmp)[0]
            X_temp = X_temp[mask]
            
            # Record filtered site count
            filtered_pos_count = len(pos_temp)
            self.global_stats["chromosome_stats"][chr_name]["filtered_sites"] = filtered_pos_count
            print(f"  Filtered site count (after gap removal): {filtered_pos_count} (removed {original_pos_count - filtered_pos_count} gap sites)")

            # Skip chromosomes with insufficient sites
            if filtered_pos_count < self.segment_size:
                print(f"  Skipping chromosome {chr_name}: filtered site count {filtered_pos_count} < segment size {self.segment_size}")
                continue
            
            # Core segmentation logic
            cuts_ = torch.arange(0, filtered_pos_count - self.segment_size + 1, self.segment_size)
            cuts = torch.tensor([(indices[i], indices[i + self.segment_size - 1]) for i in cuts_])
            # Add last segment
            cuts_ = torch.cat((cuts_, torch.tensor([filtered_pos_count - self.segment_size])))
            cuts = torch.cat((cuts, torch.tensor([(indices[-self.segment_size], indices[-1])])))

            # Generate segment list (each segment: (X segment, y segment, ind segment, pos segment))
            batched_temp = [
                (X_temp[max(srt - self.RF2, 0):stp + 1 + self.RF2],
                 y_temp[i:i + self.segment_size],
                 indices[i:i + self.segment_size] - indices[i] + self.RF2, 
                 pos_temp[i:i + self.segment_size] - pos_temp[i]) 
                for i, (srt, stp) in zip(cuts_, cuts)
            ]
            
            # Record post-segmentation statistics
            segment_count = len(batched_temp)
            segmented_sites = sum(len(seg[3]) for seg in batched_temp)
            self.global_stats["chromosome_stats"][chr_name]["segments"] = segment_count
            self.global_stats["chromosome_stats"][chr_name]["segmented_sites"] = segmented_sites
            self.global_stats["total_segmented_sites"] += segmented_sites
            print(f"  After segmentation: generated {segment_count} segments, total segmented sites {segmented_sites}")

            # Key modification 2: Store validation set by chromosome and record global index range
            if self.val_keys is not None and chr_name in self.val_keys:
                val_by_chr[chr_name] = batched_temp  # Store segments for this chromosome
                val_segment_range[chr_name] = (global_val_index, global_val_index + segment_count)  # Record index range
                global_val_index += segment_count  # Update global index
                print(f"  Assigned to validation set: {segment_count} segments, global index range {val_segment_range[chr_name]}")
            
            elif self.test_keys is not None and chr_name in self.test_keys:
                test += batched_temp
                print(f"  Assigned to test set: {len(batched_temp)} segments")
            
            elif self.fracs != [1,0,0]:
                random.shuffle(batched_temp)
                splits = np.cumsum(np.round(np.array(self.fracs)*len(batched_temp)).astype('int'))
                train += batched_temp[:splits[0]]
                test += batched_temp[splits[1]:]
                print(f"  Assigned proportionally: train set {len(batched_temp[:splits[0]])} | test set {len(batched_temp[splits[1]:])}")
            
            else:
                train += batched_temp
                print(f"  Assigned to train set: {len(batched_temp)} segments")

        # Key modification 3: Build global validation set list (add chromosome label to each segment: (X,y,ind,pos,chr_name))
        val = []
        if self.val_keys is not None:
            for chr_name in self.val_keys:
                if chr_name in val_by_chr:
                    # Add chromosome name to each segment, format: (X,y,ind,pos,chr_name)
                    labeled_segments = [(*seg, chr_name) for seg in val_by_chr[chr_name]]
                    val.extend(labeled_segments)
        print(f"\nTotal validation set segments after merging: {len(val)} (from chromosomes: {list(val_by_chr.keys())})")

        # Generate fixed masks (with chromosome distinction)
        fixed_masks = None
        fixed_masks_by_chr = None
        if self.val_keys is not None and len(val) > 0:
            val_dataset_temp = CpGTransformerDataset_fix(
                val, RF=self.RF, mask_percentage=self.mask_perc, 
                mask_random_percentage=self.mask_random_perc, resample_cells=self.resample_val
            )

            # Calculate total mask count
            total_nonzeros_all = sum([sum((seg[1]+1).nonzero().size(0) for seg in val_by_chr[cn]) for cn in val_by_chr.keys()])
            total_mask_count = int(round(total_nonzeros_all * self.mask_perc))

            if self.fixed_masks_path:
                fixed_masks_file = os.path.join(self.fixed_masks_path, "fixed_masks_with_chr.pt")
                if os.path.exists(fixed_masks_file):
                    loaded_data = torch.load(fixed_masks_file)
                    fixed_masks = loaded_data["fixed_masks"]
                    fixed_masks_by_chr = loaded_data["fixed_masks_by_chr"]
                    print(f"\nLoaded masks with chromosome identifiers: {list(fixed_masks_by_chr.keys())}")
                    # Validate mask count matching
                    for chr_name in self.val_keys:
                        assert len(fixed_masks_by_chr[chr_name]) == len(val_by_chr[chr_name]), \
                            f"Mask count for chromosome {chr_name} does not match segment count"
                else:
                    fixed_masks, fixed_masks_by_chr = generate_fixed_masks_local(
                        val_dataset_temp, val_by_chr, val_segment_range, 
                        total_mask_count, self.mask_random_perc
                    )
                    os.makedirs(os.path.dirname(fixed_masks_file), exist_ok=True)
                    torch.save(
                        {"fixed_masks": fixed_masks, "fixed_masks_by_chr": fixed_masks_by_chr},
                        fixed_masks_file
                    )
                    print(f"\nSaved masks with chromosome identifiers to {fixed_masks_file}")
            else:
                fixed_masks, fixed_masks_by_chr = generate_fixed_masks_local(
                    val_dataset_temp, val_by_chr, val_segment_range, 
                    total_mask_count, self.mask_random_perc
                )
                print(f"\nGenerated masks with chromosome identifiers, total count: {total_mask_count}")

        # Initialize datasets
        self.train = CpGTransformerDataset(train, RF=self.RF,
                                           mask_percentage=self.mask_perc, 
                                           mask_random_percentage=self.mask_random_perc,
                                           resample_cells=self.resample)

        self.val = CpGTransformerDataset_fix(val, RF=self.RF,
                                           mask_percentage=self.mask_perc, 
                                           mask_random_percentage=self.mask_random_perc,
                                           resample_cells=self.resample_val,
                                           fixed_mask_indices=fixed_masks)

        self.test = CpGTransformerDataset(test, RF=self.RF,
                                          mask_percentage=0.0,  # No masking for test set
                                          mask_random_percentage=0.0,
                                          resample_cells=None)

        # Print global statistics
        print("\n" + "="*50)
        print("Global site statistics summary:")
        print(f"Total original sites across all chromosomes: {self.global_stats['total_original_sites']}")
        print(f"Total filtered sites after gap removal: {sum(stats['filtered_sites'] for stats in self.global_stats['chromosome_stats'].values())}")
        print(f"Total segmented sites: {self.global_stats['total_segmented_sites']}")
        print(f"Segmented sites ratio to filtered sites: {self.global_stats['total_segmented_sites'] / sum(stats['filtered_sites'] for stats in self.global_stats['chromosome_stats'].values()):.2%}")
        print("="*50 + "\n")
        
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, num_workers=self.nw,
                                           batch_size=self.bsz, shuffle=True,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, num_workers=self.nw,
                                           batch_size=self.bsz, shuffle=False,
                                           pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, num_workers=self.nw,
                                           batch_size=self.bsz, shuffle=False,
                                           pin_memory=True)


class CpGTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, split, RF=1001, mask_percentage=0.25,
                 mask_random_percentage=0.20, resample_cells=None):
        self.split = split  # Train/test set segments: (X,y,ind,pos) (no chromosome name)
        RF2 = int((RF-1)/2)
        self.r = torch.arange(-RF2, RF2+1)
        self.k = RF

        # Build CDF
        s = torch.stack([s[1] for s in split]) if len(split) > 0 else torch.tensor([])
        s = s[s != -1] if s.numel() > 0 else torch.tensor([])
        indices = torch.randperm(s.shape[0])[:2500] if s.shape[0] > 2500 else torch.arange(s.shape[0])
        self.cdf = {'x': s[indices].sort().values, 'y': torch.linspace(0, 1, len(indices))} if len(indices) > 0 else {'x': torch.tensor([]), 'y': torch.tensor([])}
        
        self.mp = mask_percentage
        self.mrp = mask_random_percentage
        self.resample = resample_cells
        
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, ind, pos = self.split[index]  # Train/test set segments have no chromosome name
        x_windows = x[ind.unsqueeze(1).repeat(1,self.k)+self.r]
        cell_indices = torch.arange(y.shape[1])
            
        if self.resample and self.resample < cell_indices.shape[0]:
            sample_indices = torch.randperm(cell_indices.shape[0])[:self.resample]
            cell_indices = cell_indices[sample_indices]
            y = y[:,sample_indices]
        
        y_orig = y+1
        seqlen, n_rep = y_orig.size()
        y_masked = y_orig.clone()
        
        nonzeros = y_masked.nonzero(as_tuple=False)
        n_permute = min(int(seqlen*self.mp), nonzeros.size(0)) if seqlen > 0 else 0
        
        if self.mrp and n_permute > 0 and len(self.cdf['x']) > 0:
            n_mask, n_random = int(n_permute*(1-self.mrp)), math.ceil(n_permute*self.mrp)
            perm = torch.randperm(nonzeros.size(0))[:n_permute]
            nonzeros = nonzeros[perm]
            mask, rand = torch.split(nonzeros,[n_mask,n_random]) if n_mask > 0 else (nonzeros, torch.empty((0,2)))
            
            y_masked[mask[:,0],mask[:,1]] = 0 if mask.numel() > 0 else y_masked
            if rand.numel() > 0:
                y_masked[rand[:,0],rand[:,1]] = sample_from_cdf(self.cdf, n_random)+1
        elif n_permute > 0:
            perm = torch.randperm(nonzeros.size(0))[:n_permute]
            nonzeros = nonzeros[perm]
            y_masked[nonzeros[:,0],nonzeros[:,1]] = 0
        
        # Train/test set return: no chromosome name (maintain original compatibility)
        return x_windows, y_orig, y_masked, pos, nonzeros, cell_indices


class CpGTransformerDataset_fix(torch.utils.data.Dataset):
    """Validation set specific dataset: returns chromosome name, supports fixed masks"""
    def __init__(self, split, RF=1001, mask_percentage=0.25,
                 mask_random_percentage=0.20, resample_cells=None, fixed_mask_indices=None):
        self.split = split  # Validation set segments: (X,y,ind,pos,chr_name) (includes chromosome name)
        RF2 = int((RF-1)/2)
        self.r = torch.arange(-RF2, RF2+1)
        self.k = RF

        # Build CDF
        s = torch.stack([s[1] for s in split]) if len(split) > 0 else torch.tensor([])
        s = s[s != -1] if s.numel() > 0 else torch.tensor([])
        indices = torch.randperm(s.shape[0])[:2500] if s.shape[0] > 2500 else torch.arange(s.shape[0])
        self.cdf = {'x': s[indices].sort().values, 'y': torch.linspace(0, 1, len(indices))} if len(indices) > 0 else {'x': torch.tensor([]), 'y': torch.tensor([])}
        
        self.mp = mask_percentage
        self.mrp = mask_random_percentage
        self.resample = resample_cells
        self.fixed_mask_indices = fixed_mask_indices  # Fixed mask list (by global segment index)
        
        print(f"Initialized validation set with fixed masks: {len(self.split)} segments, {len(fixed_mask_indices) if fixed_mask_indices is not None else 0} fixed masks")
    
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        # Key modification: parse chromosome name from segment (split element format: (X,y,ind,pos,chr_name))
        x, y, ind, pos, chr_name = self.split[index]
        x_windows = x[ind.unsqueeze(1).repeat(1,self.k)+self.r]
        cell_indices = torch.arange(y.shape[1])
            
        # Resample cells (if needed)
        if self.resample and self.resample < cell_indices.shape[0]:
            sample_indices = torch.randperm(cell_indices.shape[0])[:self.resample]
            cell_indices = cell_indices[sample_indices]
            y = y[:,sample_indices]
        
        y_orig = y+1
        seqlen, n_rep = y_orig.size() if y_orig.numel() > 0 else (0, 0)
        y_masked = y_orig.clone() if y_orig.numel() > 0 else torch.tensor([])
        nonzeros = torch.empty((0,2), dtype=torch.long)

        # Apply fixed masks
        if self.fixed_mask_indices is not None and index < len(self.fixed_mask_indices):
            local_mask = self.fixed_mask_indices[index]  # Fixed mask for current segment
            if local_mask.numel() > 0 and seqlen > 0 and n_rep > 0:
                # Filter invalid indices
                valid_row = (local_mask[:, 0] >= 0) & (local_mask[:, 0] < seqlen)
                valid_col = (local_mask[:, 1] >= 0) & (local_mask[:, 1] < n_rep)
                valid_mask = valid_row & valid_col
                nonzeros = local_mask[valid_mask]
                n_permute = nonzeros.size(0)

                if n_permute > 0:
                    # Allocate mask and random replacement proportionally
                    if self.mrp and len(self.cdf['x']) > 0:
                        n_mask = int(n_permute * (1 - self.mrp))
                        n_random = n_permute - n_mask
                        mask, rand = torch.split(nonzeros, [n_mask, n_random]) if n_mask > 0 else (nonzeros, torch.empty((0,2)))
                        
                        y_masked[mask[:,0], mask[:,1]] = 0 if mask.numel() > 0 else y_masked
                        if rand.numel() > 0:
                            y_masked[rand[:,0], rand[:,1]] = sample_from_cdf(self.cdf, n_random) + 1
                    else:
                        y_masked[nonzeros[:,0], nonzeros[:,1]] = 0
        # Fallback to random masking (if fixed masks are invalid)
        elif seqlen > 0 and self.mp > 0:
            nonzeros = y_masked.nonzero(as_tuple=False) if y_masked.numel() > 0 else torch.empty((0,2))
            n_permute = min(int(seqlen*self.mp), nonzeros.size(0))
            if n_permute > 0:
                perm = torch.randperm(nonzeros.size(0))[:n_permute]
                nonzeros = nonzeros[perm]
                y_masked[nonzeros[:,0], nonzeros[:,1]] = 0

        return x_windows, y_orig, y_masked, pos, nonzeros, cell_indices, chr_name