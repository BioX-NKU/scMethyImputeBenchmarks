# -*- coding: utf-8 -*-
"""
Process only chr10 and chr11, split into 4 blocks (Block0-3) by halving the original row count (without sorting)
Completely resolve empty DataFrame issues during merging (enhanced logging + filter validation + logic optimization)
"""
from __future__ import division
from sys import argv
import math
import pandas as pd
import numpy as np
import os
import time

from numba import jit
     
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
warnings.filterwarnings('ignore')

@jit
def chu(a, n):
    return round(a/(2*n), 4)

@jit
def corre(a, b):
    return round(sum(a==b)/len(a), 4)   

@jit
def getlog2(a):    
    return round(np.log2(a+1.01), 4)

def reduce_mem(df):
    """Reduce DataFrame memory usage (with defensive logic retained)"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    if df.empty:
        print(f"reduce_mem received empty DataFrame (shape: {df.shape}), return directly")
        return df
    
    for col in df.columns:
        if col not in df.columns:
            print(f"Column {col} does not exist in DataFrame, skip")
            continue
        
        col_type = df[col].dtypes
        if not isinstance(col_type, np.dtype):
            print(f"Column {col} type {type(col_type)} is abnormal (not np.dtype), skip")
            continue
        
        if str(col_type) not in numerics:
            continue
        
        c_min = df[col].min()
        c_max = df[col].max()
        if pd.isnull(c_min) or pd.isnull(c_max):
            continue
        
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    return df

def unionlargefile(i1, gse, region, chh, cell_num): 
    """Merge large block data (core optimization: full-link data volume logging + filter rationality verification)"""
    print(f"\n=== Block{i1} start merging large block data ===")
    file_dir_Block = r'%s/Forimputation/local_methFeature_cellbycell/Block%d' % (gse, i1)
    if not os.path.exists(file_dir_Block):
        os.makedirs(file_dir_Block)    
        print(f"Created Block output directory: {file_dir_Block}")
    
    # Confirm merge source directory and file list
    meragefiledir = r'%s/Forimputation/local_methFeature_for_imputation/localRegion_%d' % (gse, region)
    if not os.path.exists(meragefiledir):
        print(f"Block{i1} merge source directory does not exist: {meragefiledir}, skip")
        return f'Block{i1}: Source directory does not exist, skip!'
    
    filenames = os.listdir(meragefiledir)
    if not filenames:
        print(f"Block{i1} merge source directory {meragefiledir} has no files, skip")
        return f'Block{i1}: No files, skip!'
    print(f"File list under merge source directory {meragefiledir}: {filenames}")

    # Read first file (core: print data volume before/after filtering to locate empty data reasons)
    first_file = filenames[0]
    path = r'%s/%s' % (meragefiledir, first_file)
    if not os.path.exists(path):
        print(f"First file {path} does not exist, skip")
        return f'Block{i1}: First file does not exist, skip!'
    
    data = pd.read_csv(path, header=0, sep='\t')
    print(f"Read first file {first_file}: original data volume {len(data)} rows, column names {list(data.columns[:5])}...")
    
    # Preprocessing (retain original data volume logging)
    data[list(data)[2:]] = data[list(data)[2:]].astype('float16')
    data[list(data)[2:]] = data[list(data)[2:]].round(4)
    print(f"Data volume after preprocessing: {len(data)} rows (no data loss)")
    
    # Filter chromosome and row range for current block (core verification: avoid empty after filtering)
    current_block_info = chh[i1]
    current_chrom = current_block_info['chrom']
    row_min = current_block_info['row_min']
    row_max = current_block_info['row_max']
    print(f"Current block filter conditions: chromosome={current_chrom}, row range [{row_min}:{row_max}]")
    
    # Chromosome filter verification
    before_chrom_filter = len(data)
    data = data[data['chrom'] == current_chrom]
    after_chrom_filter = len(data)
    print(f"Chromosome filter: {before_chrom_filter} rows before filtering → {after_chrom_filter} rows after filtering")
    if after_chrom_filter == 0:
        print(f"No data left after chromosome {current_chrom} filtering, check if file contains this chromosome")
        return f'Block{i1}: No data after chromosome filtering, skip!'
    
    # Row range filter verification (avoid index out of bounds)
    before_row_filter = len(data)
    if row_min is not None:
        row_min = max(0, row_min)  
        data = data.iloc[row_min:]
    if row_max is not None:
        row_max = min(len(data), row_max)  
        data = data.iloc[:row_max]
    after_row_filter = len(data)
    print(f"Row range filter: {before_row_filter} rows before filtering → {after_row_filter} rows after filtering")
    if after_row_filter == 0:
        print(f"No data left after row range filtering, adjust row_min/row_max (current [{row_min}:{row_max}])")
        return f'Block{i1}: No data after row range filtering, skip!'
    
    # Memory optimization
    data = reduce_mem(data)
    print(f"Data volume after memory optimization: {len(data)} rows (no data loss)")

    # Merge subsequent files (print filter log for each file)
    for i in range(1, len(filenames)):
        curr_file = filenames[i]
        path = r'%s/%s' % (meragefiledir, curr_file)
        if not os.path.exists(path):
            print(f"File {curr_file} does not exist, skip")
            continue
        
        df = pd.read_csv(path, header=0, sep='\t')
        print(f"\nRead file {curr_file}: original data volume {len(df)} rows")
        
        # Preprocessing + filtering (same logic as first file)
        df[list(df)[2:]] = df[list(df)[2:]].astype('float16')
        df[list(df)[2:]] = df[list(df)[2:]].round(4)
        
        # Chromosome filtering
        df_chrom = df[df['chrom'] == current_chrom]
        print(f"{curr_file} chromosome filter: {len(df)} rows → {len(df_chrom)} rows")
        if len(df_chrom) == 0:
            print(f"{curr_file} has no {current_chrom} data, skip merging")
            continue
        
        # Row range filtering
        if row_min is not None:
            df_row = df_chrom.iloc[max(0, row_min):]
        else:
            df_row = df_chrom.copy()
        if row_max is not None:
            df_row = df_row.iloc[:min(len(df_row), row_max)]
        print(f"{curr_file} row range filter: {len(df_chrom)} rows → {len(df_row)} rows")
        if len(df_row) == 0:
            print(f"{curr_file} has no data after row range filtering, skip merging")
            continue
        
        # Merge (outer join to ensure no data loss)
        df_row = reduce_mem(df_row)
        before_merge = len(data)
        data = pd.merge(data, df_row, how='outer', on=['chrom', 'location'])
        after_merge = len(data)
        print(f"{curr_file} merge completed: {before_merge} rows before merging → {after_merge} rows after merging")

    # Split by cell and save (only save if there is data)
    if len(data) == 0:
        print(f"No data left after merging all files, skip cell splitting")
        return f'Block{i1}: No data after merging, skip!'
    
    file_dir_1 = r'%s/region%d/corr' % (file_dir_Block, region)
    file_dir_2 = r'%s/region%d/methy' % (file_dir_Block, region)
    os.makedirs(file_dir_1, exist_ok=True)
    os.makedirs(file_dir_2, exist_ok=True)
    print(f"\nCreated cell split directories: corr={file_dir_1}, methy={file_dir_2}")
    
    a = list(data)
    valid_cell_count = 0
    for cell in cell_num:
        # Filter r and methy columns for current cell
        cell_r = [col for col in a[2:] if (col.split('_')[0] == cell or col.split('_')[1] == cell) and col.split('_')[-1] == 'r']
        cell_meth = [col for col in a[2:] if (col.split('_')[0] == cell or col.split('_')[1] == cell) and col.split('_')[-1] == 'methy']
        
        if not cell_r and not cell_meth:
            print(f"Cell {cell} has no corresponding r/methy columns, skip")
            continue
        
        # Generate split data
        df_r = data[['chrom', 'location'] + cell_r].dropna(axis=0, subset=cell_r, how='all') if cell_r else pd.DataFrame()
        df_m = data[['chrom', 'location'] + cell_meth].dropna(axis=0, subset=cell_meth, how='all') if cell_meth else pd.DataFrame()
        
        # Save valid data
        if len(df_r) > 0:
            df_r.to_csv(r'%s/%s_r.txt' % (file_dir_1, cell), sep='\t', header=True, index=False)
            print(f"Saved corr data for cell {cell}: {len(df_r)} rows")
        if len(df_m) > 0:
            df_m.to_csv(r'%s/%s_m.txt' % (file_dir_2, cell), sep='\t', header=True, index=False)
            print(f"Saved methy data for cell {cell}: {len(df_m)} rows")
        valid_cell_count += 1
    
    print(f"\n=== Block{i1} merge completed: processed {valid_cell_count} valid cells ===")
    return f'Block{i1}: Success (processed {valid_cell_count} cells)!'
        
def unionfinalfile(file_dir, meragefiledir, filenames, filenames1, filenames2, i1, region, file_dir_1, file_dir_2):
    """Merge final files (optimization: data volume tracking + skip abnormal files)"""
    print(f"\n=== Start merging final files (range {i1}-{i1+4}) ===")
    if i1 + 4 >= len(filenames1):
        i2 = len(filenames1)
    else:
        i2 = i1 + 4   
    print(f"Merge file range: {i1}→{i2}, total {len(filenames1)} target files")

    for j in range(i1, i2):
        if j >= len(filenames1) or j >= len(filenames2):
            print(f"Index j={j} exceeds file list length, skip")
            continue
        
        corr_file = filenames1[j]
        methy_file = filenames2[j]
        print(f"\n--- Processing file pair: corr={corr_file}, methy={methy_file} ---")
        
        # Read corr file of first block
        path_r = r'%s/%s/region%d/corr/%s' % (meragefiledir, filenames[0], region, corr_file)
        if not os.path.exists(path_r):
            print(f"corr file {path_r} does not exist, skip this file pair")
            continue
        df_r = pd.read_csv(path_r, header=0, sep='\t')
        print(f"Read first block corr file: {len(df_r)} rows")
        
        # Read methy file of first block
        path_m = r'%s/%s/region%d/methy/%s' % (meragefiledir, filenames[0], region, methy_file)
        if not os.path.exists(path_m):
            print(f"methy file {path_m} does not exist, skip this file pair")
            continue
        df_m = pd.read_csv(path_m, header=0, sep='\t')
        print(f"Read first block methy file: {len(df_m)} rows")
        
        # Merge files of subsequent blocks
        for block in filenames[1:]:
            block_corr_path = r'%s/%s/region%d/corr/%s' % (meragefiledir, block, region, corr_file)
            block_methy_path = r'%s/%s/region%d/methy/%s' % (meragefiledir, block, region, methy_file)
            
            # Merge corr
            if os.path.exists(block_corr_path):
                block_corr = pd.read_csv(block_corr_path, header=0, sep='\t')
                before_corr_merge = len(df_r)
                df_r = pd.concat([df_r, block_corr], ignore_index=True)
                after_corr_merge = len(df_r)
                print(f"Merged corr of block {block}: {before_corr_merge} rows → {after_corr_merge} rows")
            else:
                print(f"corr file of block {block} does not exist, skip")
            
            # Merge methy
            if os.path.exists(block_methy_path):
                block_methy = pd.read_csv(block_methy_path, header=0, sep='\t')
                before_methy_merge = len(df_m)
                df_m = pd.concat([df_m, block_methy], ignore_index=True)
                after_methy_merge = len(df_m)
                print(f"Merged methy of block {block}: {before_methy_merge} rows → {after_methy_merge} rows")
            else:
                print(f"methy file of block {block} does not exist, skip")
        
        # Deduplicate and save (only save if there is data)
        # Deduplicate corr
        df_r = reduce_mem(df_r)
        before_corr_drop = len(df_r)
        df_r = df_r.drop_duplicates(['chrom', 'location'], keep='first')
        after_corr_drop = len(df_r)
        if after_corr_drop > 0:
            df_r.to_csv(r'%s/%s' % (file_dir_1, corr_file), sep='\t', header=True, index=False)
            print(f"Saved corr file: {before_corr_drop} rows before deduplication → {after_corr_drop} rows after deduplication")
        else:
            print(f"No data left after corr file deduplication, skip saving")
        
        # Deduplicate methy
        df_m = reduce_mem(df_m)
        before_methy_drop = len(df_m)
        df_m = df_m.drop_duplicates(['chrom', 'location'], keep='first')
        after_methy_drop = len(df_m)
        if after_methy_drop > 0:
            df_m.to_csv(r'%s/%s' % (file_dir_2, methy_file), sep='\t', header=True, index=False)
            print(f"Saved methy file: {before_methy_drop} rows before deduplication → {after_methy_drop} rows after deduplication")
        else:
            print(f"No data left after methy file deduplication, skip saving")

    print(f"\n=== Merge of range {i1}-{i2} completed ===")
    return f'Block{i1}: Final merge completed!'

def unionfile(file_dir, meragefiledir, filenames, filenames1, i1):
    """Merge local feature files generated by test (Block0-3→localRegion)"""
    print(f"\n=== Start merging local feature files (process {i1}) ===")
    if i1 + 2 >= len(filenames1):
        i2 = len(filenames1)
    else:
        i2 = i1 + 2   
    print(f"Merge file range: {i1}→{i2}, block list: {filenames}")

    for j in range(i1, i2):
        if j >= len(filenames1):
            print(f"Index j={j} exceeds file list length, skip")
            continue
        
        target_file = filenames1[j]
        df_merged = pd.DataFrame(columns=['chrom', 'location'])  # Initialize non-empty DataFrame
        print(f"\n--- Processing target file: {target_file} ---")

        for block in filenames:
            block_path = r'%s/%s/%s' % (meragefiledir, block, target_file)
            if not os.path.exists(block_path):
                print(f"File {target_file} of block {block} does not exist (path: {block_path}), skip")
                continue
            
            # Read block file
            block_data = pd.read_csv(block_path, header=0, sep='\t')
            print(f"Read file of block {block}: {len(block_data)} rows, column names {list(block_data.columns[:5])}...")
            
            # Preprocessing (same logic as test generation)
            block_data[list(block_data)[2:]] = block_data[list(block_data)[2:]].astype('float16')
            block_data[list(block_data)[2:]] = block_data[list(block_data)[2:]].round(4)
            block_data = reduce_mem(block_data)
            
            # Merge (use merge for first time, concat for subsequent to avoid empty DataFrame issue)
            if df_merged.empty:
                df_merged = block_data.copy()
                print(f"First merge: block {block} data as initial data ({len(df_merged)} rows)")
            else:
                before_merge = len(df_merged)
                df_merged = pd.concat([df_merged, block_data], ignore_index=True)
                after_merge = len(df_merged)
                print(f"Merged block {block}: {before_merge} rows → {after_merge} rows")
        
        # Deduplicate and save (only save if there is data)
        if df_merged.empty:
            print(f"No data left after merging all blocks, skip saving {target_file}")
            continue
        
        before_drop = len(df_merged)
        df_merged = df_merged.drop_duplicates(['chrom', 'location'], keep='first')
        after_drop = len(df_merged)
        if after_drop > 0:
            save_path = r'%s/%s' % (file_dir, target_file)
            df_merged.to_csv(save_path, sep='\t', header=True, index=False)
            print(f"Saved merged file: {save_path} ({before_drop} rows before deduplication → {after_drop} rows after deduplication)")
        else:
            print(f"No data left after merged file deduplication, skip saving {target_file}")

    print(f"\n=== Process {i1} merge completed ===")
    return f'Process {i1}: Local feature merge completed!'
    
def test(data, block_info, neighbor_region, file_dir, gse, bocknum, i1):    
    current_chrom = block_info['chrom']
    row_min = block_info['row_min']
    row_max = block_info['row_max']
    print(f"\n=== Block{bocknum} process {i1} start processing: {current_chrom}[{row_min}:{row_max}] ===")
    
    if i1 + 2 >= (len(list(data)) - 1):
        i2 = len(list(data)) - 1 
    else:
        i2 = i1 + 2
        
    for i in range(i1, i2):
        current_col = list(data)[i]
        print(f"\n--- Processing cell column: {current_col} ---")
        data_all = pd.DataFrame(columns=['chrom', 'location'])  # Initialize non-empty DataFrame

        # Read r value file
        path_r_impu = r'%s/for_impu_r/Block%d/%s_local_r.txt' % (gse, bocknum, current_col) 
        if not os.path.exists(path_r_impu):
            print(f"r value file does not exist: {path_r_impu}, skip current column")
            continue
        data_r_for_imputation = pd.read_csv(path_r_impu, header=0, sep='\t')
        data_r_for_imputation = reduce_mem(data_r_for_imputation)
        if data_r_for_imputation.empty:
            print(f"r value file is empty: {path_r_impu}, skip current column")
            continue
        print(f"Read r value file: {len(data_r_for_imputation)} rows")
        
        # Process cell pairs
        valid_pair_count = 0
        for k in range(i + 1, len(list(data))):
            pair_col = list(data)[k]
            r_col = f'%s_%s_r' % (current_col, pair_col)
            
            # Check if r column exists
            if r_col not in data_r_for_imputation.columns:
                print(f"r column {r_col} of cell pair {current_col}-{pair_col} does not exist, skip")
                continue
            
            # Filter cell pair data
            data1_cols = [list(data)[0], list(data)[1], current_col, pair_col]
            data1 = data[data1_cols].copy()
            data1 = reduce_mem(data1)
            if data1.empty:
                print(f"Original data of cell pair {current_col}-{pair_col} is empty, skip")
                continue
            
            # Filter chromosome and row range for current block
            data2 = data1[data1['chrom'] == current_chrom].copy()
            if row_min is not None:
                data2 = data2.iloc[max(0, row_min):]
            if row_max is not None:
                data2 = data2.iloc[:min(len(data2), row_max)]
            if data2.empty:
                print(f"No data left after filtering cell pair {current_col}-{pair_col}, skip")
                continue
            print(f"Data after filtering cell pair {current_col}-{pair_col}: {len(data2)} rows")
            
            # Subsequent preprocessing logic (retain original)
            data2 = data2.dropna(subset=[pair_col])
            data2_sorted = data2.sort_values(by='location')
            serchtable = list(data2_sorted[current_col])
            serchtable_m = list(data2_sorted[pair_col])
            serchtable_l = list(data2_sorted['location'])
            
            data2_sorted = data2_sorted.dropna(subset=[current_col])
            data2_sorted = pd.merge(data2_sorted, data_r_for_imputation[['chrom', 'location', r_col]], 
                                    how='left', on=['chrom', 'location'])
            if data2_sorted.empty:
                print(f"No data left after merging r values for cell pair {current_col}-{pair_col}, skip")
                continue
            
            # Neighbor feature calculation (retain original)
            data2_value = data2_sorted.values
            loca_for_imputation = []
            imputation_r = []
            imputation_m = []
            
            s = -1
            ss = []
            for l in range(len(serchtable)):                    
                if not(np.isnan(serchtable[l])):
                    s += 1
                ss.append(s)
                
                if len(ss) >= 2 and ss[-1] < len(data2_value) - neighbor_region:
                    if (ss[-1] == ss[-2]) and (ss[-1] != -1) and (ss[-1] >= neighbor_region):
                        loca_for_imputation.append(serchtable_l[l])
                        imputation_r.append(data2_value[ss[-1], -1])  
                        imputation_m.append(serchtable_m[l])    
            
            # Organize results
            if len(loca_for_imputation) == 0:
                print(f"No valid neighbor data for cell pair {current_col}-{pair_col}, skip")
                continue
            
            data_for_impu = pd.DataFrame({
                'chrom': current_chrom,
                'location': loca_for_imputation,
                r_col: imputation_r,
                f'%s_%s_methy' % (current_col, pair_col): imputation_m
            })
            data_for_impu[f'%s_%s_methy' % (current_col, pair_col)] = np.log2(
                data_for_impu[f'%s_%s_methy' % (current_col, pair_col)] + 1.01
            ).round(4)
            data_for_impu = reduce_mem(data_for_impu)
            print(f"Generated neighbor data for cell pair {current_col}-{pair_col}: {len(data_for_impu)} rows")
            
            # Merge to total results
            if data_all.empty:
                data_all = data_for_impu.copy()
            else:
                data_all = pd.merge(data_all, data_for_impu, how='outer', on=['chrom', 'location'])
            valid_pair_count += 1
        
        # Save results
        if data_all.empty:
            print(f"No valid data for cell column {current_col}, skip saving")
        else:
            save_path = r'%s/%s_local_methFeature.txt' % (file_dir, current_col)
            data_all.to_csv(save_path, sep='\t', header=True, index=False)
            print(f"Saved results for cell column {current_col}: {save_path} ({len(data_all)} rows)")
    
    print(f"\n=== Block{bocknum} process {i1} processing completed ===")
    return f'Process {i1}: Block{bocknum} processing completed!'

if __name__ == '__main__':
    # Command line parameter verification
    if len(argv) != 5:
        print("Parameter error! Correct format: python script_name data_path input_filename neighbor_region correlation_threshold")
        exit(1)
    DataPath = r'%s' % argv[1]
    InputDataName = '%s' % argv[2]
    neighbor_region = int(argv[3])
    threshold = float(argv[4])
    print(f"=== Program startup parameters ===")
    print(f"Data path: {DataPath}")
    print(f"Input file: {InputDataName}")
    print(f"Neighbor region: {neighbor_region}")
    print(f"Correlation coefficient threshold: {threshold}")
    
    gse = DataPath
    ff = InputDataName	    
    start = time.clock()

    # Read and preprocess original data
    print(f"\n=== Read original data ===")
    path_raw = r'%s/%s' % (gse, ff)
    if not os.path.exists(path_raw):
        print(f"Original data file does not exist: {path_raw}, exit program")
        exit(1)
    data = pd.read_csv(path_raw, header=0, sep='\t')
    print(f"Read original data: {len(data)} rows, {len(data.columns)} columns, column names {list(data.columns[:5])}...")
    
    # Memory optimization
    data = reduce_mem(data)
    print(f"Original data after memory optimization: {len(data)} rows (no data loss)")
    
    # Ensure chrom column is the first column
    if list(data)[0] != 'chrom':
        if 'chrom' in data.columns:
            data = data[['chrom'] + [col for col in data.columns if col != 'chrom']]
            print(f"Adjusted chrom column to be the first column")
        else:
            print(f"No chrom column in data, exit program")
            exit(1)

    # Filter target chromosomes (chr10/chr11)
    print(f"\n=== Filter target chromosomes (chr10/chr11) ===")
    target_chroms = ['chr10', 'chr11']
    original_data = data.copy()
    data = data[data['chrom'].isin(target_chroms)]  
    data = data.drop_duplicates(['chrom', 'location'], keep='first')
    
    if data.empty:
        print(f"No data left after filtering chr10/chr11 from original data, exit program")
        exit(1)
    print(f"Data after filtering: {len(data)} rows, chromosome distribution: {data['chrom'].value_counts().to_dict()}")

    # Group by chromosome and split into blocks (4 blocks)
    print(f"\n=== Chromosome grouping and block splitting ===")
    chrom_data_dict = {}
    split_row = {}
    for chrom in target_chroms:
        # Group and reset index (keep original order)
        chrom_data = original_data[original_data['chrom'] == chrom].copy()
        chrom_data = chrom_data.reset_index(drop=True)
        chrom_data_dict[chrom] = chrom_data
        
        # Calculate split point (halve the row count)
        total_rows = len(chrom_data)
        split_row[chrom] = total_rows // 2
        print(f"- {chrom}: total rows {total_rows}, split point {split_row[chrom]} (first half 0-{split_row[chrom]}, second half {split_row[chrom]}-{total_rows})")
    
    # Define 4 blocks (Block0-3)
    chh = [
        {'chrom': 'chr10', 'row_min': None, 'row_max': split_row['chr10']},    # Block0: first half of chr10
        {'chrom': 'chr10', 'row_min': split_row['chr10'], 'row_max': None},    # Block1: second half of chr10
        {'chrom': 'chr11', 'row_min': None, 'row_max': split_row['chr11']},    # Block2: first half of chr11
        {'chrom': 'chr11', 'row_min': split_row['chr11'], 'row_max': None}     # Block3: second half of chr11
    ]
    # Build block description list first
    blocks_description = []
    for i in range(4):
        chrom = chh[i]["chrom"]
        row_min = chh[i]["row_min"]
        row_max = chh[i]["row_max"]
        blocks_description.append(f'Block{i}({chrom}[{row_min}:{row_max}])')

    # Print block description
    print(f"4 blocks defined: {blocks_description}")

    # Extract cell numbers
    cell_num = list(data.columns[2:-1]) if len(data.columns) > 3 else []
    cell_num = list(set(cell_num))  # Deduplicate
    print(f"\n=== Extract cell numbers ===")
    if not cell_num:
        print(f"No cell numbers extracted (check data column structure)")
    else:
        print(f"Extracted {len(cell_num)} cells in total: {cell_num[:5]}...")  # Print first 5 to avoid long output

    # Calculate column block indexes (control number of columns processed in parallel)
    a = len(data.columns) - 3  # Number of cell columns (exclude chrom, location, last column)
    ll = [0]
    k = 0
    for i in range(int(math.floor(a / 2))):
        k += 2
        ll.append(k)
    if a % 2 != 0:
        ll.append(k + 1)
    print(f"\n=== Column block indexes ===")
    print(f"Total number of cell columns: {a}, block indexes: {ll}")

    # Multi-process processing for each Block (local feature generation)
    print(f"\n=== Start processing 4 Blocks (local feature generation) ===")
    for block_idx in range(len(chh)):
        current_block = chh[block_idx]
        current_chrom = current_block['chrom']
        current_data = chrom_data_dict[current_chrom]
        
        print(f"\n=== Block{block_idx} ({current_chrom}[{current_block['row_min']}:{current_block['row_max']}]) ===")
        print(f"Block data volume: {len(current_data)} rows")
        
        # Create Block directory
        file_dir_Block = r'%s/Forimputation/local_methFeature/Block%d' % (gse, block_idx)
        os.makedirs(file_dir_Block, exist_ok=True)
        print(f"Block directory: {file_dir_Block}")
        
        # Multi-process processing (max_workers = 1/2 of CPU cores to avoid resource competition)
        max_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 2
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    test, 
                    current_data,
                    current_block,
                    neighbor_region,
                    file_dir_Block,
                    gse,
                    block_idx,
                    i
                ) for i in ll
            ]
            for j in as_completed(futures):
                try:
                    result = j.result()
                    print(f"{result}")
                except Exception as e:
                    print(f"Process execution exception: {str(e)}")
    
    # Merge local feature files (Block0-3→localRegion)
    print(f"\n=== Merge local feature files ===")
    meragefiledir = r'%s/Forimputation/local_methFeature' % gse
    filenames = [f'Block{i}' for i in range(4)]  # Process only Block0-3
    filenames = [f for f in filenames if os.path.exists(os.path.join(meragefiledir, f))]
    if not filenames:
        print(f"No valid Block directories, skip local feature merging")
    else:
        # Get target file list (based on Block0)
        first_block = filenames[0]
        first_block_dir = os.path.join(meragefiledir, first_block)
        filenames1 = os.listdir(first_block_dir) if os.path.exists(first_block_dir) else []
        if not filenames1:
            print(f"No files in Block{first_block}, skip local feature merging")
        else:
            # Create merge directory
            file_dir_union = r'%s/Forimputation/local_methFeature_for_imputation/localRegion_%d' % (gse, neighbor_region)
            os.makedirs(file_dir_union, exist_ok=True)
            print(f"Local feature merge directory: {file_dir_union}")
            
            # Multi-process merge
            with ProcessPoolExecutor(max_workers=min(4, os.cpu_count()//2)) as pool:
                futures = [
                    pool.submit(unionfile, file_dir_union, meragefiledir, filenames, filenames1, i) 
                    for i in ll
                ]
                for j in as_completed(futures):
                    try:
                        print(f"{j.result()}")
                    except Exception as e:
                        print(f"Merge process exception: {str(e)}")
            print(f"Local feature merge completed")

    # Merge large block files (localRegion→cellbycell Block)
    print(f"\n=== Merge large block files (generate cellbycell Block) ===")
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count()//2)) as pool:
        futures = [
            pool.submit(unionlargefile, i, gse, neighbor_region, chh, cell_num) 
            for i in range(len(chh))
        ]
        for j in as_completed(futures):
            try:
                print(f"{j.result()}")
            except Exception as e:
                print(f"Large block merge process exception: {str(e)}")
    print(f"Large block file merge completed")

    # Merge final files (cellbycell Block→final corr/methy)
    print(f"\n=== Merge final files (generate global corr/methy) ===")
    file_dir_cell = r'%s/Forimputation/local_methFeature_cellbycell' % gse
    if not os.path.exists(file_dir_cell):
        print(f"cellbycell directory does not exist, skip final merge")
    else:
        meragefiledir = file_dir_cell
        filenames = [f'Block{i}' for i in range(4)]
        filenames = [f for f in filenames if os.path.exists(os.path.join(meragefiledir, f))]
        if not filenames:
            print(f"No valid cellbycell Block directories, skip final merge")
        else:
            # Get file list (based on first Block)
            first_block = filenames[0]
            corr_dir = os.path.join(meragefiledir, first_block, f'region{neighbor_region}', 'corr')
            methy_dir = os.path.join(meragefiledir, first_block, f'region{neighbor_region}', 'methy')
            if not os.path.exists(corr_dir) or not os.path.exists(methy_dir):
                print(f"corr/methy directories do not exist, skip final merge")
            else:
                filenames1 = os.listdir(corr_dir)
                filenames2 = os.listdir(methy_dir)
                if not filenames1 or not filenames2:
                    print(f"No files in corr/methy directories, skip final merge")
                else:
                    # Create final merge directory
                    file_dir_1 = r'%s/Forimputation/local_methFeature_cellbycell/region%d/corr' % (gse, neighbor_region)
                    file_dir_2 = r'%s/Forimputation/local_methFeature_cellbycell/region%d/methy' % (gse, neighbor_region)
                    os.makedirs(file_dir_1, exist_ok=True)
                    os.makedirs(file_dir_2, exist_ok=True)
                    print(f"Final merge directories: corr={file_dir_1}, methy={file_dir_2}")
                    
                    # Calculate merge block indexes
                    a = len(filenames1)
                    ll_final = [0]
                    k = 0
                    for i in range(int(math.floor(a / 4))):
                        k += 4
                        ll_final.append(k)
                    if a % 2 != 0:
                        ll_final.append(k + 1)
                    
                    # Multi-process merge
                    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count()//2)) as pool:
                        futures = [
                            pool.submit(
                                unionfinalfile, 
                                file_dir_cell, 
                                meragefiledir, 
                                filenames, 
                                filenames1, 
                                filenames2, 
                                i, 
                                neighbor_region, 
                                file_dir_1, 
                                file_dir_2
                            ) for i in ll_final
                        ]
                        for j in as_completed(futures):
                            try:
                                print(f"{j.result()}")
                            except Exception as e:
                                print(f"Final merge process exception: {str(e)}")
                    print(f"Final file merge completed")

    # Split by cell and generate matching results
    print(f"\n=== Split by cell and generate matching results ===")
    if not cell_num:
        print(f"No cell numbers, skip splitting")
    else:
        file_dir_1 = r'%s/Forimputation/local_methFeature_cellbycell/region%d/corr' % (gse, neighbor_region)
        file_dir_2 = r'%s/Forimputation/local_methFeature_cellbycell/region%d/methy' % (gse, neighbor_region)
        if not os.path.exists(file_dir_1) or not os.path.exists(file_dir_2):
            print(f"corr/methy directories do not exist, skip splitting")
        else:
            file_dir_final = r'%s/Forimputation/region%d_localmatched_morethan08' % (gse, neighbor_region)
            os.makedirs(file_dir_final, exist_ok=True)
            print(f"Split result directory: {file_dir_final}")
            
            valid_cell_count = 0
            for cell in cell_num:
                path_r = r'%s/%s_r.txt' % (file_dir_1, cell)
                path_m = r'%s/%s_m.txt' % (file_dir_2, cell)
                if not os.path.exists(path_r) or not os.path.exists(path_m):
                    print(f"corr/methy files of cell {cell} do not exist, skip")
                    continue
                
                # Read data
                data_r = pd.read_csv(path_r, header=0, sep='\t')
                data_m = pd.read_csv(path_m, header=0, sep='\t')
                if data_r.empty or data_m.empty:
                    print(f"corr/methy files of cell {cell} are empty, skip")
                    continue
                
                # Calculate matching results (retain original logic)
                df_r = data_r[['chrom', 'location']].copy()
                name = list(data_r.columns[2:])
                data_r.fillna(0, inplace=True)
                data_m.fillna(0, inplace=True)
                
                r_value = data_r.values
                m_value = data_m.values    
                col_ind = []
                col_ind_m = []
                
                for j in range(len(r_value)):       
                    a_arr = np.array(r_value[j, 2:])
                    b_arr = np.array(m_value[j, 2:])
                    c = np.where(a_arr >= threshold)
                    
                    if len(c[0]) != 0:
                        ss = np.sum(a_arr[c[0]] * b_arr[c[0]])
                        col_ind_m.append(round(ss / len(c[0]), 4))    
                        if len(c[0]) == 1 :
                            col_name = name[c[0][0]]
                            if col_name.split('_')[-2] == cell:
                                col_ind.append(col_name.split('_')[0])
                            else:
                                col_ind.append(col_name.split('_')[-2])
                        else:
                            col_ind.append(len(c[0]))
                    else:
                        col_ind.append(np.nan)
                        col_ind_m.append(np.nan)
                
                # Save results
                df_r['aver_meth'] = col_ind_m
                df_r['matched_cell'] = col_ind
                save_path = r'%s/%s.txt' % (file_dir_final, cell)
                df_r.to_csv(save_path, sep='\t', header=True, index=False)
                print(f"Saved split results for cell {cell}: {save_path} ({len(df_r)} rows)")
                valid_cell_count += 1
            
            print(f"Cell splitting completed: processed {valid_cell_count} valid cells in total")

    # Program end
    elapsed = (time.clock() - start)
    print(f"\n=== All processes completed ===")
    print(f"Total time consumed: {round(elapsed / 60, 2)} minutes ({round(elapsed, 2)} seconds)")
    print(f"Result directory: {gse}/Forimputation")