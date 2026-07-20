from time import time
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import scipy.sparse as ssp
import bisect
from tqdm import tqdm


class CpGGraphDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for CpG methylation graph data processing
    Splits data into train/val/test sets by chromosome and creates graph datasets
    
    Args:
        y (dict): Dictionary of methylation matrices with chromosome names as keys
        segment_size (int): Size of the sliding window for subgraph extraction (default: 21)
        val_keys (list): List of chromosome names for validation set
        test_keys (list): List of chromosome names for test set
        batch_size (int): Batch size for data loaders (default: 2)
        n_workers (int): Number of workers for data loading (default: 4)
        cell_nums (int): Number of cells to use (False to use all cells from chr1)
        save_cell_id (bool): Whether to save cell indices (m/n) in Data objects
        mask_file_path (str): Path to mask indices file for fixed dataset
    """
    def __init__(self, y, segment_size=21, 
                 val_keys=None, test_keys=None,
                 batch_size=2, n_workers=4, cell_nums=False, save_cell_id=False, mask_file_path=False):
        super().__init__()
        self.save_hyperparameters(ignore=['y'])
        self.y = y
        self.segment_size = segment_size
        self.half_seg = int((segment_size - 1)/2)
        self.val_keys = val_keys if val_keys else []
        self.test_keys = test_keys if test_keys else []
        self.batch_size = batch_size
        self.nw = n_workers
        self.mask_file_path = mask_file_path
        
        # Set number of cells
        if cell_nums:
            self.cell_num = cell_nums
        else:
            self.cell_num = y["chr1"].shape[1]
        self.save_cell_id = save_cell_id
    
    def setup(self):
        """Prepare train/val/test datasets by extracting subgraphs from chromosomes"""
        time_start = time()
        
        # Split chromosomes into train/val/test sets
        all_chrs = list(self.y.keys())
        train_chrs = [chr_name for chr_name in all_chrs 
                      if chr_name not in self.val_keys and chr_name not in self.test_keys]
        
        # Extract subgraphs for each dataset split
        val_graphs_temp, val_chrs_length_dict = self.extract_subgraphs(self.val_keys, "val", self.cell_num)
        train_graphs_temp, train_chrs_length_dict = self.extract_subgraphs(train_chrs, "train", self.cell_num)
        test_graphs_temp, test_chrs_length_dict = self.extract_subgraphs(self.test_keys, "test", self.cell_num)
        
        # Create datasets
        self.val = CpGGraphDataset_fix(
            val_graphs_temp, val_chrs_length_dict, 
            self.segment_size, True, self.mask_file_path
        )
        self.train = CpGGraphDataset(
            train_graphs_temp, train_chrs_length_dict, 
            self.segment_size, self.save_cell_id
        )
        self.test = CpGGraphDataset_fix(
            test_graphs_temp, test_chrs_length_dict, 
            self.segment_size, True, self.mask_file_path
        )
        
        time_end = time()
        print(f"Data preparation time: {time_end - time_start:.2f} seconds")

    def extract_subgraphs(self, chr_names, dataset_type, cell_nums):
        """
        Extract subgraph data from specified chromosomes
        
        Args:
            chr_names (list): List of chromosome names to process
            dataset_type (str): Dataset type (train/val/test) for logging
            cell_nums (int): Number of cells to include
        
        Returns:
            tuple: (list of CSR matrices, length dictionary for chromosome indices)
        """
        chrs_graphs_temp = []
        chrs_length_dict = [0]
        chr_length = 0
        
        for chr_name in tqdm(chr_names, desc=f"Processing {dataset_type} chromosomes"):
            # Get methylation matrix and pad edges to handle window boundaries
            y_temp = self.y[chr_name][:, :cell_nums]
            padding = np.full((self.half_seg, y_temp.shape[1]), -1)
            y_temp = np.concatenate((padding, y_temp, padding))
            
            # Convert to CSR matrix (efficient for row-based operations)
            y_temp_csr = ssp.csr_matrix(y_temp + 1)  # Shift values: -1→0, 0→1, 1→2
            chr_length += len(y_temp_csr.data)
            chrs_length_dict.append(chr_length)
            chrs_graphs_temp.append(y_temp_csr)
        
        return chrs_graphs_temp, chrs_length_dict

    def train_dataloader(self):
        """Return training DataLoader with shuffling"""
        return DataLoader(
            self.train, 
            num_workers=self.nw,
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=10
        )

    def val_dataloader(self):
        """Return validation DataLoader without shuffling"""
        return DataLoader(
            self.val, 
            num_workers=self.nw,
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=10
        )

    def test_dataloader(self):
        """Return test DataLoader without shuffling"""
        return DataLoader(
            self.test, 
            num_workers=self.nw,
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=10
        )


class CpGGraphDataset(Dataset):
    """
    Base Dataset class for CpG methylation graph data
    Creates graph representations for each methylation site using sliding window
    
    Args:
        chrs_graphs_temp (list): List of CSR matrices for each chromosome
        chrs_length_dict (list): Cumulative length dictionary for chromosome indices
        segment_size (int): Size of the sliding window
        save_cell_id (bool): Whether to save cell position indices (m/n)
    """
    def __init__(self, chrs_graphs_temp, chrs_length_dict, segment_size, save_cell_id):
        super().__init__()
        self.chrs_graphs_temp = chrs_graphs_temp
        self.chrs_length_dict = chrs_length_dict
        self.link_num = chrs_length_dict[-1]
        self.segment_size = segment_size
        self.cell_num = chrs_graphs_temp[0].shape[1]
        self.half_seg = int((segment_size - 1)/2)
        self.save_cell_id = save_cell_id
        
        # Create node features (locus-aware encoding)
        # Segment nodes (0-segment_size-1) get unique IDs, cell nodes get 0
        self.raw_x = F.one_hot(
            torch.cat((torch.arange(0, self.segment_size)+1, torch.zeros(self.cell_num))).long()
        ).to(torch.float)

    def len(self):
        """Return total number of methylation sites"""
        return self.link_num

    def get(self, index): 
        """
        Get graph data for a single methylation site
        
        Args:
            index (int): Global index of the methylation site
        
        Returns:
            Data: PyTorch Geometric Data object with graph information
        """
        # Find corresponding chromosome and local index
        chr_ind = bisect.bisect_left(self.chrs_length_dict, index + 1) - 1
        chr_graph_csr = self.chrs_graphs_temp[chr_ind]
        rela_index = index - self.chrs_length_dict[chr_ind]
        
        # Get position of the methylation site
        col_ind = chr_graph_csr.indices[rela_index]
        row_ind = bisect.bisect_left(chr_graph_csr.indptr, rela_index + 1)
        
        # Define window boundaries for subgraph extraction
        window_start = (row_ind - 1) - self.half_seg
        window_end = row_ind + self.half_seg
        
        # Extract subgraph from window
        subgraph_csr = chr_graph_csr[window_start : window_end, :]
        
        # Get target value (shift back from CSR encoding)
        y = torch.tensor(subgraph_csr[self.half_seg, col_ind] - 1).to(torch.bool)
        
        # Remove target information from subgraph (mask it)
        subgraph_csr[self.half_seg, col_ind] = 1
        
        # Convert to COO format for edge creation
        subgraph_coo = subgraph_csr.tocoo()
        
        # Create edge indices (segment nodes: 0-segment_size-1, cell nodes: segment_size+)
        chr_u = torch.from_numpy(subgraph_coo.row).to(torch.long)
        chr_v = torch.from_numpy(subgraph_coo.col + self.segment_size).to(torch.int16)
        chr_r = torch.from_numpy(subgraph_coo.data - 1).to(torch.bool)
        
        # Node features
        x = self.raw_x.clone()
        
        # Create bidirectional edges
        edge_index = torch.stack([
            torch.cat([chr_u, chr_v]), 
            torch.cat([chr_v, chr_u])
        ], 0)
        
        # Edge types (methylation status)
        edge_type = torch.cat([chr_r, chr_r])
        
        # Create Data object with optional position information
        if self.save_cell_id:
            subgraph_data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_type=edge_type, 
                y=y, 
                chr=torch.tensor(chr_ind, dtype=torch.long),
                m=torch.tensor(row_ind, dtype=torch.long),
                n=torch.tensor(col_ind, dtype=torch.long)
            )
        else:
            subgraph_data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_type=edge_type, 
                y=y
            )
        
        return subgraph_data


class CpGGraphImputationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for CpG methylation imputation task
    Focuses on missing value imputation across all chromosomes
    
    Args:
        y (dict): Dictionary of methylation matrices with chromosome names as keys
        segment_size (int): Size of the sliding window for subgraph extraction (default: 21)
        batch_size (int): Batch size for data loaders (default: 2)
        n_workers (int): Number of workers for data loading (default: 4)
        cell_nums (int): Number of cells to use (False to use all cells from chr1)
    """
    def __init__(self, y, segment_size=21, 
                 batch_size=2, n_workers=4, cell_nums=False):
        super().__init__()
        self.save_hyperparameters(ignore=['y'])
        self.y = y
        self.segment_size = segment_size
        self.half_seg = int((segment_size - 1)/2)
        self.batch_size = batch_size
        self.nw = n_workers
        
        # Set number of cells
        if cell_nums:
            self.cell_num = cell_nums
        else:
            self.cell_num = y["chr1"].shape[1]
        self.save_cell_id = True
    
    def setup(self):
        """Prepare imputation dataset by identifying missing values"""
        time_start = time()
        
        # Extract subgraphs for imputation (focus on missing values)
        graphs_temp, chrs_missing_length_dict, y_missing_temp = self.extract_imputation_subgraphs(self.cell_num)
        
        # Create imputation dataset
        self.all = CpGGraphImputationDataset(
            graphs_temp, 
            chrs_missing_length_dict, 
            y_missing_temp, 
            self.segment_size, 
            self.save_cell_id
        )
        
        time_end = time()
        print(f"Imputation data preparation time: {time_end - time_start:.2f} seconds")

    def extract_imputation_subgraphs(self, cell_nums):
        """
        Extract subgraph data for missing value imputation
        
        Args:
            cell_nums (int): Number of cells to include
        
        Returns:
            tuple: (list of CSR matrices, missing length dict, missing value positions)
        """
        y_missing_temp = []
        chrs_graphs_temp = []
        chrs_missing_length_dict = [0]
        chr_missing_length = 0
        
        # Process all chromosomes in the dataset
        used_chrs = self.y.files if hasattr(self.y, 'files') else list(self.y.keys())
        
        for chr_name in tqdm(used_chrs, desc="Processing chromosomes for imputation"):
            # Get methylation matrix
            y_temp = self.y[chr_name][:, :cell_nums]
            
            # Identify missing values (-1) with at least 2 valid values in the row
            y_missing = []
            valid_rows = list(set(np.argwhere(y_temp > -1)[:, 0]))
            
            for row in valid_rows:
                missing_cols = np.argwhere(y_temp[row, :] == -1)
                # Only include rows with at least 2 valid values
                if y_temp.shape[1] - len(missing_cols) > 1:
                    y_missing.extend([[row, col[0]] for col in missing_cols])
            
            # Convert to array and update counters
            y_missing = np.array(y_missing)
            y_missing_temp.append(y_missing)
            chr_missing_length += len(y_missing)
            chrs_missing_length_dict.append(chr_missing_length)
            
            # Pad matrix and convert to CSR
            padding = np.full((self.half_seg, y_temp.shape[1]), -1)
            y_temp = np.concatenate((padding, y_temp, padding))
            y_temp_csr = ssp.csr_matrix(y_temp + 1)
            chrs_graphs_temp.append(y_temp_csr)
        
        return chrs_graphs_temp, chrs_missing_length_dict, y_missing_temp


class CpGGraphImputationDataset(Dataset):
    """
    Dataset class for CpG methylation imputation task
    Focuses on missing value positions and creates graph representations
    
    Args:
        chrs_graphs_temp (list): List of CSR matrices for each chromosome
        chrs_missing_length_dict (list): Cumulative length dict for missing positions
        y_missing_temp (list): Missing value positions for each chromosome
        segment_size (int): Size of the sliding window
        save_cell_id (bool): Whether to save cell position indices (m/n)
    """
    def __init__(self, chrs_graphs_temp, chrs_missing_length_dict, y_missing_temp, segment_size, save_cell_id):
        super().__init__()
        self.chrs_graphs_temp = chrs_graphs_temp
        self.cell_num = chrs_graphs_temp[0].shape[1]
        self.half_seg = int((segment_size - 1)/2)
        self.chrs_missing_length_dict = chrs_missing_length_dict
        self.y_missing_temp = y_missing_temp
        self.link_num = chrs_missing_length_dict[-1]
        self.segment_size = segment_size
        self.save_cell_id = save_cell_id
        
        # Create node features (locus-aware encoding)
        self.raw_x = F.one_hot(
            torch.cat((torch.arange(0, self.segment_size)+1, torch.zeros(self.cell_num))).long()
        ).to(torch.float)

    def len(self):
        """Return total number of missing values"""
        return self.link_num

    def get(self, index): 
        """
        Get graph data for a single missing value position
        
        Args:
            index (int): Global index of the missing value
        
        Returns:
            Data: PyTorch Geometric Data object with graph information
        """
        # Find corresponding chromosome and local index
        chr_ind = bisect.bisect_left(self.chrs_missing_length_dict, index + 1) - 1
        chr_graph_csr = self.chrs_graphs_temp[chr_ind]
        y_missing = self.y_missing_temp[chr_ind]
        rela_index = index - self.chrs_missing_length_dict[chr_ind]
        
        # Get position of the missing value
        row_ind, col_ind = y_missing[rela_index]
        row_ind = row_ind + self.half_seg + 1  # Adjust for padding
        
        # Define window boundaries
        window_start = (row_ind - 1) - self.half_seg
        window_end = row_ind + self.half_seg
        
        # Extract subgraph
        subgraph_csr = chr_graph_csr[window_start : window_end, :]
        
        # Get target value (missing value placeholder)
        y = torch.tensor(subgraph_csr[self.half_seg, col_ind] - 1).to(torch.bool)
        
        # Mask target information
        subgraph_csr[self.half_seg, col_ind] = 1
        
        # Convert to COO format
        subgraph_coo = subgraph_csr.tocoo()
        
        # Create edge indices
        chr_u = torch.from_numpy(subgraph_coo.row).to(torch.long)
        chr_v = torch.from_numpy(subgraph_coo.col + self.segment_size).to(torch.int16)
        chr_r = torch.from_numpy(subgraph_coo.data - 1).to(torch.bool)
        
        # Node features
        x = self.raw_x.clone()
        
        # Create bidirectional edges
        edge_index = torch.stack([
            torch.cat([chr_u, chr_v]), 
            torch.cat([chr_v, chr_u])
        ], 0)
        
        # Edge types
        edge_type = torch.cat([chr_r, chr_r])
        
        # Create Data object with position information
        if self.save_cell_id:
            subgraph_data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_type=edge_type,
                chr=torch.tensor(chr_ind, dtype=torch.long),
                m=torch.tensor(row_ind - self.half_seg - 1, dtype=torch.long),  # Adjust back to original row
                n=torch.tensor(col_ind, dtype=torch.long)
            )
        else:
            subgraph_data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_type=edge_type
            )
        
        return subgraph_data


class CpGGraphDataset_fix(Dataset):
    """
    Fixed Dataset class for CpG methylation graph data
    Addresses CSR matrix write protection by using LIL format for mask application
    
    Args:
        chrs_graphs_temp (list): List of CSR matrices for each chromosome
        chrs_length_dict (list): Cumulative length dictionary for chromosome indices
        segment_size (int): Size of the sliding window
        save_cell_id (bool): Whether to save cell position indices (m/n)
        mask_file_path (str): Path to mask indices file (N x 2 array of [row, col] indices)
    """
    def __init__(self, chrs_graphs_temp, chrs_length_dict, segment_size, save_cell_id, mask_file_path):
        super().__init__()
        self.chrs_graphs_temp = chrs_graphs_temp
        self.chrs_length_dict = chrs_length_dict
        self.link_num = chrs_length_dict[-1]
        self.segment_size = segment_size
        self.cell_num = chrs_graphs_temp[0].shape[1]
        self.half_seg = int((segment_size - 1) / 2)
        self.save_cell_id = save_cell_id
        
        # Load and validate mask indices
        if mask_file_path and mask_file_path != 'False':
            self.mask_indices = torch.load(mask_file_path)
            # Validate mask format (N x 2 array of indices)
            if self.mask_indices.ndim != 2 or self.mask_indices.shape[1] != 2:
                raise ValueError("Mask file must be a 2D array with shape [N, 2] (row, column indices)")
        else:
            self.mask_indices = None
        
        # Create node features
        self.raw_x = F.one_hot(
            torch.cat((torch.arange(0, self.segment_size) + 1, torch.zeros(self.cell_num))).long()
        ).to(torch.float)

    def len(self):
        """Return total number of methylation sites"""
        return self.link_num

    def get(self, index):
        """
        Get graph data for a single methylation site with mask application
        
        Args:
            index (int): Global index of the methylation site
        
        Returns:
            Data: PyTorch Geometric Data object with graph information
        """
        # Find corresponding chromosome and local index
        chr_ind = bisect.bisect_left(self.chrs_length_dict, index + 1) - 1
        chr_graph_csr = self.chrs_graphs_temp[chr_ind]
        rela_index = index - self.chrs_length_dict[chr_ind]
        
        # Get position of the methylation site
        col_ind = chr_graph_csr.indices[rela_index]
        row_ind = bisect.bisect_left(chr_graph_csr.indptr, rela_index + 1)
        
        # Define window boundaries
        window_start = (row_ind - 1) - self.half_seg
        window_end = row_ind + self.half_seg
        
        # Step 1: Extract subgraph and convert to LIL format (writable)
        subgraph_csr = chr_graph_csr[window_start : window_end, :]
        subgraph_lil = subgraph_csr.tolil()  # LIL format supports in-place modification
        
        # Get target value
        y = torch.tensor(subgraph_lil[self.half_seg, col_ind] - 1).to(torch.bool)
        
        # Step 2: Apply mask if provided (modify LIL matrix)
        if self.mask_indices is not None:
            # Filter mask indices within current window
            current_global_row_start = window_start
            current_global_row_end = window_end - 1
            mask_in_window = (self.mask_indices[:, 0] >= current_global_row_start) & \
                             (self.mask_indices[:, 0] <= current_global_row_end)
            relevant_indices = self.mask_indices[mask_in_window]
            
            if len(relevant_indices) > 0:
                # Convert to local indices within subgraph
                local_rows = relevant_indices[:, 0] - window_start
                local_cols = relevant_indices[:, 1]
                
                # Validate indices to prevent out-of-bounds errors
                valid_mask = (local_rows >= 0) & (local_rows < subgraph_lil.shape[0]) & \
                             (local_cols >= 0) & (local_cols < subgraph_lil.shape[1])
                
                if torch.any(valid_mask):
                    # Apply mask (set to 1) - LIL format supports direct assignment
                    valid_rows = local_rows[valid_mask].numpy().astype(int)
                    valid_cols = local_cols[valid_mask].numpy().astype(int)
                    subgraph_lil[valid_rows, valid_cols] = 1
        
        # Step 3: Convert back to CSR for COO conversion
        subgraph_csr = subgraph_lil.tocsr()
        
        # Step 4: Create graph structure (same as base dataset)
        subgraph_coo = subgraph_csr.tocoo()
        chr_u = torch.from_numpy(subgraph_coo.row).to(torch.long)
        chr_v = torch.from_numpy(subgraph_coo.col + self.segment_size).to(torch.int16)
        chr_r = torch.from_numpy(subgraph_coo.data - 1).to(torch.bool)
        
        # Node features
        x = self.raw_x.clone()
        
        # Bidirectional edge index
        edge_index = torch.stack([
            torch.cat([chr_u, chr_v]), 
            torch.cat([chr_v, chr_u])
        ], 0)
        
        # Edge types
        edge_type = torch.cat([chr_r, chr_r])
        
        # Create Data object with position information
        if self.save_cell_id:
            subgraph_data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_type=edge_type, 
                y=y, 
                chr=torch.tensor(chr_ind, dtype=torch.long),
                m=torch.tensor(row_ind, dtype=torch.long),
                n=torch.tensor(col_ind, dtype=torch.long)
            )
        else:
            subgraph_data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_type=edge_type, 
                y=y
            )
        
        return subgraph_data