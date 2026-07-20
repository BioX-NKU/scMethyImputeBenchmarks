import torch
import torch.utils.data
import pytorch_lightning as pl
import os

class MambaCpGDataModule(pl.LightningDataModule):
    def __init__(self, x, y, pos, DNA_window=1001, segment_size=1024, batch_size=1, 
                 n_workers=4, mask_percentage=0.15, masked_replace_percentage=0.2, val_keys=None, test_keys=None, fixed_mask_path=None):
        super().__init__()
        self.x = x
        self.y = y
        self.pos = pos
        self.DNA_win = DNA_window
        self.hDNA_win = int((DNA_window-1)/2)                     
        self.segsz = segment_size
        self.bsz = batch_size
        self.n_wor = n_workers
        self.maskp = mask_percentage
        self.replacep = masked_replace_percentage
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.prepare_data_per_node = True
        self.fixed_mask_path = fixed_mask_path
        
    def setup(self, stage):
        train = []; val = []; test = []
        
        for chr_name in self.y.keys():
            x = self.x[chr_name]
            y = self.y[chr_name]
            pos = self.pos[chr_name]
            if 'numpy' in str(type(x)):
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                pos = torch.from_numpy(pos)
            
            DNAx = torch.cat((torch.full((self.hDNA_win,), 4, dtype=torch.int8), 
                             x, 
                             torch.full((self.hDNA_win,), 4, dtype=torch.int8)), dim=0)
            posx = pos + self.hDNA_win
            start = posx - self.hDNA_win
            end = posx + self.hDNA_win + 1
            
            DNA_seg = []  
            for s, e in zip(start, end):
                segment = DNAx[s:e]
                if segment.size(0) == self.DNA_win:
                    DNA_seg.append(segment)
            
            if not DNA_seg:
                continue
                
            DNA_seg = torch.stack(DNA_seg)

            n_pos = len(DNA_seg)
            if n_pos < self.segsz:
                continue
            
            batch_ = [(DNA_seg[i:i + self.segsz], y[i:i + self.segsz], pos[i:i + self.segsz] - pos[i])
                       for i in range(0, n_pos - self.segsz + 1, self.segsz)]
       
            a = n_pos % self.segsz
            if a > 0:
                _, ncell = y.shape
                end_DNA_padding = torch.full((self.segsz - a, self.DNA_win), 4, dtype=torch.int8)
                end_DNA_seg = torch.cat((DNA_seg[-a:], end_DNA_padding), dim=0)
                end_pos_padding = torch.full((self.segsz - a,), -1, dtype=torch.int32)
                end_pos_seg = torch.cat((pos[-a:], end_pos_padding), dim=0)
                end_CpG_padding = torch.full((self.segsz - a, ncell), -1, dtype=torch.int8)
                end_CpG_seg = torch.cat((y[-a:], end_CpG_padding), dim=0)
                batch_.append((end_DNA_seg, end_CpG_seg, end_pos_seg))               

            if self.val_keys is not None and chr_name in self.val_keys:
                val += batch_
            elif self.test_keys is not None and chr_name in self.test_keys:
                test += batch_
            else:
                train += batch_
        
        self.train = MambaCpGDataset(train, mask_percentage=self.maskp, masked_replace_percentage=self.replacep)
        self.val = MambaCpGDataset_fix(val, mask_percentage=self.maskp, masked_replace_percentage=self.replacep, fixed_masks_path=self.fixed_mask_path)
        self.test = test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, num_workers=self.n_wor, batch_size=self.bsz, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, num_workers=self.n_wor, batch_size=self.bsz, shuffle=False, pin_memory=True)


class MambaCpGDataset(torch.utils.data.Dataset):
    def __init__(self, split, mask_percentage, masked_replace_percentage): 
        self.split = split
        self.maskp = mask_percentage
        self.replacep = masked_replace_percentage

    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, _ = self.split[index]
        cell_indices = torch.arange(y.shape[1])
        y = y + 1

        y_known = y.nonzero(as_tuple=False)
        num_to_mask = int(y_known.size(0) * self.maskp)

        mask_indices = y_known[torch.randperm(y_known.size(0))[:num_to_mask]]

        y_masked = y.clone()
        y_masked[mask_indices[:, 0], mask_indices[:, 1]] = 0

        num_to_replace = int(num_to_mask * self.replacep)
        replace_indices = mask_indices[torch.randperm(mask_indices.size(0))[:num_to_replace]]

        replacements = torch.randint(1, 3, (num_to_replace,))
        for i, (row, col) in enumerate(replace_indices):
            y_masked[row, col] = replacements[i]
        
        y_orig = []
        for i in range(num_to_mask):
            y_orig.append(y[mask_indices[i, 0], mask_indices[i, 1]])

        y_orig = torch.tensor(y_orig) - 1

        return y_orig, y_masked, mask_indices, cell_indices, x


class MambaCpGDataset_fix(torch.utils.data.Dataset):
    def __init__(self, split, mask_percentage, masked_replace_percentage, fixed_masks_path=None): 
        self.split = split
        self.maskp = mask_percentage
        self.replacep = masked_replace_percentage
        self.fixed_masks_path = fixed_masks_path
        self.fixed_masks = None
        
        if self.fixed_masks_path and os.path.exists(self.fixed_masks_path):
            self.fixed_masks = torch.load(self.fixed_masks_path)
            print(f"Loaded fixed mask sites from {self.fixed_masks_path}, total {len(self.fixed_masks)} segments (one-to-one correspondence with data segments)")
            
            for i, mask in enumerate(self.fixed_masks):
                if not (isinstance(mask, torch.Tensor) and mask.dim() == 2 and mask.size(1) == 2):
                    raise ValueError(f"Mask {i} format error! Expected Tensor with shape [n, 2], got {type(mask).__name__}/{getattr(mask, 'shape', 'no shape')}")
                if (mask < 0).any():
                    print(f"Warning: Mask {i} contains negative indices, invalid indices will be filtered automatically")
        else:
            print("Fixed mask file path not provided or file does not exist, random mask will be used")

    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, _ = self.split[index]
        cell_indices = torch.arange(y.shape[1])
        y_orig = y + 1
        nsite, ncell = y_orig.shape
        mask_indices = torch.empty((0, 2), dtype=torch.long)

        if self.fixed_masks is not None:
            if index < len(self.fixed_masks):
                local_mask = self.fixed_masks[index]
                
                valid_row = (local_mask[:, 0] >= 0) & (local_mask[:, 0] < nsite)
                valid_col = (local_mask[:, 1] >= 0) & (local_mask[:, 1] < ncell)
                valid_mask = valid_row & valid_col
                mask_indices = local_mask[valid_mask]

                num_to_mask = mask_indices.size(0)
                if num_to_mask > 0:
                    print(f"Segment {index}: Using fixed mask, valid site count: {num_to_mask} (original mask site count: {local_mask.size(0)}, filtered invalid site count: {local_mask.size(0)-num_to_mask})")
                else:
                    print(f"Segment {index}: Fixed mask has no valid indices (possible index out of segment range), falling back to random mask")
                    mask_indices = self._generate_random_mask(y_orig)
                    num_to_mask = mask_indices.size(0)
            else:
                print(f"Segment {index}: Exceeds fixed mask range (total {len(self.fixed_masks)} masks, current segment index {index}), falling back to random mask")
                mask_indices = self._generate_random_mask(y_orig)
                num_to_mask = mask_indices.size(0)
        else:
            mask_indices = self._generate_random_mask(y_orig)
            num_to_mask = mask_indices.size(0)
            print(f"Segment {index}: Using random mask, site count: {num_to_mask}")

        y_masked = y_orig.clone()
        if num_to_mask > 0:
            y_masked[mask_indices[:, 0], mask_indices[:, 1]] = 0

            if self.replacep > 0:
                num_to_replace = int(num_to_mask * self.replacep)
                if num_to_replace > 0:
                    replace_idx = mask_indices[torch.randperm(num_to_mask)[:num_to_replace]]
                    replacements = torch.randint(1, 3, (num_to_replace,), device=replace_idx.device)
                    y_masked[replace_idx[:, 0], replace_idx[:, 1]] = replacements

        y_true = y_orig[mask_indices[:, 0], mask_indices[:, 1]] - 1 if num_to_mask > 0 else torch.tensor([], dtype=torch.long)

        return y_true, y_masked, mask_indices, cell_indices, x

    def _generate_random_mask(self, y_orig):
        y_known = y_orig.nonzero(as_tuple=False)
        num_known = y_known.size(0)
        if num_known == 0:
            return torch.empty((0, 2), dtype=torch.long)
        
        num_to_mask = min(int(num_known * self.maskp), num_known)
        return y_known[torch.randperm(num_known)[:num_to_mask]]


class MambaCpGImputingDataModule(pl.LightningDataModule):
    def __init__(self, x, y, pos, DNA_window=1001, segment_size=1024, keys=None, n_workers=4):
        super().__init__()
        self.x = x
        self.y = y
        self.pos = pos
        self.DNA_win = DNA_window
        self.hDNA_win = int((DNA_window - 1) / 2)                     
        self.segsz = segment_size
        self.keys = keys
        self.n_wor = n_workers
        
    def setup(self, stage):
        self.datasets_per_chr = dict()
        
        iterate = self.keys if self.keys is not None else self.y.keys()
        for chr_name in iterate:
            x = self.x[chr_name]
            y = self.y[chr_name]
            pos = self.pos[chr_name]
            if 'numpy' in str(type(x)):
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                pos = torch.from_numpy(pos)

            DNAx = torch.cat((torch.full((self.hDNA_win,), 4, dtype=torch.int8), 
                             x, 
                             torch.full((self.hDNA_win,), 4, dtype=torch.int8)), dim=0)
            posx = pos + self.hDNA_win
            start = posx - self.hDNA_win
            end = posx + self.hDNA_win + 1
            
            DNA_seg = []  
            for s, e in zip(start, end):
                segment = DNAx[s:e]
                if segment.size(0) == self.DNA_win:
                    DNA_seg.append(segment)
            
            if not DNA_seg:
                continue
                
            DNA_seg = torch.stack(DNA_seg)

            n_pos = len(DNA_seg)
            batch_ = [(DNA_seg[i:i + self.segsz], y[i:i + self.segsz], pos[i:i + self.segsz] - pos[i])
                       for i in range(0, n_pos - self.segsz + 1, self.segsz)]
       
            a = n_pos % self.segsz
            if a > 0:
                _, ncell = y.shape
                end_DNA_padding = torch.full((self.segsz - a, self.DNA_win), 4, dtype=torch.int8)
                end_DNA_seg = torch.cat((DNA_seg[-a:], end_DNA_padding), dim=0)
                end_pos_padding = torch.full((self.segsz - a,), -1, dtype=torch.int32)
                end_pos_seg = torch.cat((pos[-a:], end_pos_padding), dim=0)
                end_CpG_padding = torch.full((self.segsz - a, ncell), -1, dtype=torch.int8)
                end_CpG_seg = torch.cat((y[-a:], end_CpG_padding), dim=0)
                batch_.append((end_DNA_seg, end_CpG_seg, end_pos_seg))               

            self.datasets_per_chr[chr_name] = torch.utils.data.DataLoader(
                ImputingDataset(batch_), num_workers = self.n_wor, shuffle=False, pin_memory=True)
            
class ImputingDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, _ = self.split[index] 
        y_orig = y + 1
        cell_indices = torch.arange(y.shape[1])
        
        return x, y_orig, cell_indices