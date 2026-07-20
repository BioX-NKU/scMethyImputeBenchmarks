# -*- coding: utf-8 -*-
"""
Process only chr10 and chr11, split into 4 blocks (Block0-3) by halving the original row count (without sorting)
Fix column index matching, variable initialization, and data null value issues to ensure file generation
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

##############################################################
def reduce_mem(df):
    """Reduce DataFrame memory usage"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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

@jit
def chu(a, n):
    return round(a/(2*n), 4)

@jit
def corre(a, b):
    return round(sum(a==b)/len(a), 4)   

@jit
def getlog2(a):    
    return round(np.log2(a+1.01), 4)

@jit
def hah(a, b, c, d):
    return round(np.log2(a+1.01), 4) * (1 - (abs(b - c)/d)) 

##############################################################
# Merge files (keep original logic, adapt to block results)
def unionfile(file_dir, meragefiledir, filenames, filenames1, i1):
    if i1 + 2 >= len(filenames1):
        i2 = len(filenames1)
    else:
        i2 = i1 + 2   
    for j in range(i1, i2):
        df = pd.DataFrame(np.random.randn(0, 2), columns=['chrom', 'location'])       
        for i in range(len(filenames)):
            path = r'%s/%s/%s' % (meragefiledir, filenames[i], filenames1[j])
            if not os.path.exists(path):
                print(f"Merged file {path} does not exist, skip")
                continue
            data = pd.read_csv(path, header=0, sep='\t')
            data = data[data['chrom'].isin(['chr10', 'chr11'])]
            if i == 0:
                df = pd.merge(df, data, how='outer')
            else:
                df = pd.concat([df, data], ignore_index=True)
        if not df.empty:
            df = df.drop_duplicates(['chrom', 'location'])
            df[list(df)[2:]] = df[list(df)[2:]].astype('float16').round(4)       
            df.to_csv(r'%s/%s' % (file_dir, filenames1[j]), sep='\t', header=True, index=False)
            print(f"Merged file generated: {filenames1[j]} ({len(df)} rows)")
        else:
            print(f"No data after merging, skip generating {filenames1[j]}")
    return (f'Process {i1}:Done!')

# Core fix: correct column index matching, variable initialization, log monitoring
def test(data, block_info, neighbor_region, file_dir, colname, cell_columns, i1):
    """
    cell_columns: List of cell columns (e.g., ['cell1','cell2',...'bulk']), solves column index matching issues
    """
    current_chrom = block_info['chrom']
    row_min = block_info['row_min']
    row_max = block_info['row_max']
    
    # 1. Fix: process index range of cell columns (i1 is relative index of cell columns)
    i2 = min(i1 + 2, len(cell_columns))  # Process 2 cell columns each time to avoid out of range
    print(f"Process {i1}: Processing cell column indexes {i1}~{i2-1} (corresponding columns: {cell_columns[i1:i2]})")
    
    # 2. Fix: initialize d_c outside loop to avoid undefined error
    d_c = pd.DataFrame(
        np.random.randn(0, 2 + 2*neighbor_region), 
        columns=['chrom', 'location'] + colname
    )  

    # 3. Handle duplicate indexes (precaution)
    if data.index.duplicated().any():
        data = data.reset_index(drop=True)

    # Iterate over each cell column (get column name from cell column list instead of list(data)[i])
    for cell_idx in range(i1, i2):
        current_cell_col = cell_columns[cell_idx]  # Correctly get cell column name (e.g., 'cell1')
        print(f"\n{current_cell_col} (chrom {current_chrom}, row range: {row_min}-{row_max}): start!")
        
        # 4. Correctly filter columns: chrom + location + current cell column (no duplicates)
        try:
            data3 = data[['chrom', 'location', current_cell_col]].copy()
        except KeyError:
            print(f"Cell column {current_cell_col} does not exist in data, skip")
            continue
        
        # Filter data of current chromosome + row range
        data2 = data3.loc[data3['chrom'] == current_chrom].copy()
        if row_min is not None:
            data2 = data2.iloc[row_min:]
        if row_max is not None:
            data2 = data2.iloc[:row_max]
        print(f"Filtered data volume: {len(data2)} rows (chrom {current_chrom}, column {current_cell_col})")
        
        # 5. Data preprocessing: delete null values, check if data volume is sufficient
        data2 = data2.dropna(subset=[current_cell_col])
        if len(data2) <= 2 * neighbor_region:  # Neighborhood calculation requires at least 2*neighbor_region+1 rows
            print(f"Insufficient valid data ({len(data2)} rows ≤ {2*neighbor_region} rows), cannot calculate neighborhood features, skip")
            continue
        
        # Neighborhood calculation (keep original logic, add logs)
        data2_sorted = data2.sort_values(by='location')
        data2_value = data2_sorted.values
        loca = list(data2_sorted['location'])
        serchtable = list(data2_sorted[current_cell_col])
        serchtable_l = list(data2_sorted['location'])
        
        # Initialize neighborhood results
        data_neigh = pd.DataFrame(
            np.random.randn(len(serchtable_l), 2 + 2*neighbor_region), 
            columns=['chrom', 'location'] + colname
        )
        data_neigh['location'] = serchtable_l
        data_neigh['chrom'] = current_chrom
        
        # Track non-null value indexes
        s = -1
        ss = []
        for l in range(len(serchtable)):                    
            if not(np.isnan(serchtable[l])):
                s += 1
            ss.append(s)
        
        # Sliding window calculation
        valid_neigh_count = 0  # Count number of valid neighborhood features
        for l in range(len(serchtable)):
            if len(ss) >= 2 and ss[-1] < len(data2_value) - neighbor_region:
                if (ss[l] == ss[l-1] if l > 0 else False) and ss[l] != -1:
                    if ss[l] >= neighbor_region and ss[l] < len(data2_value) - neighbor_region:
                        location = serchtable_l[l]
                        win_start = ss[l] - neighbor_region
                        win_end = ss[l] + neighbor_region + 1
                        location_1 = list(data2_value[win_start:ss[l]+1, 1])
                        location_2 = list(data2_value[ss[l]+1:win_end, 1])
                        
                        # Calculate maximum distance
                        maxdis = max(location_2[-1] - location, location - location_1[0]) if (location_1 and location_2) else 1
                        # Calculate neighborhood features
                        loca_win = location_1 + location_2
                        ind_win = list(range(win_start, ss[l]+1)) + list(range(ss[l]+1, win_end))
                        mmm_win = list(data2_value[ind_win, 2])
                        
                        for p in range(len(mmm_win)):
                            m4 = hah(mmm_win[p], loca_win[p], location, maxdis)
                            data_neigh.iloc[l, p+2] = round(m4, 4)
                        valid_neigh_count += 1
        
        # 6. Process neighborhood results (delete rows with all null values)
        data_neigh = data_neigh.dropna(axis=0, how='any')
        print(f"Neighborhood calculation completed: Generated {len(data_neigh)} rows of valid features (original {valid_neigh_count} neighborhood points)")
        
        # Merge to global result
        if not data_neigh.empty:
            d_c = pd.concat([d_c, data_neigh], ignore_index=True)
            print(f"Merged global result: {len(d_c)} rows")

    # 7. Save files (key: clearly print save status and path)
    if not d_c.empty:
        # Split and save by cell column (one file per cell column)
        for cell_idx in range(i1, i2):
            current_cell_col = cell_columns[cell_idx]
            # Filter neighborhood feature columns of current cell column (neighbor_0~neighbor_N)
            cell_neigh_cols = ['chrom', 'location'] + [col for col in colname]
            cell_dc = d_c[cell_neigh_cols].copy()
            # Keep only rows with data in current cell column (avoid empty rows)
            cell_dc = cell_dc.dropna(axis=0, how='any')
            if not cell_dc.empty:
                save_path = r'%s/%s_neighbor_methFeature.txt' % (file_dir, current_cell_col)
                cell_dc.to_csv(save_path, sep='\t', header=True, index=False)
                print(f"\nFile generated: {save_path} ({len(cell_dc)} rows)")
            else:
                print(f"{current_cell_col} has no valid neighborhood features, skip file generation")
    else:
        print(f"\nProcess {i1} has no valid neighborhood features, skip file generation")
    
    return (f'Process {i1}:Done!')

##############################################################
if __name__ == '__main__':
    # Command line parameters
    if len(argv) != 4:
        print("Parameter error! Correct format: python script_name data_path input_filename neighbor_region")
        exit(1)
    DataPath = r'%s' % argv[1]
    InputDataName = '%s' % argv[2]
    neighbor_region = int(argv[3])

    gse = DataPath
    ff = InputDataName	
    start = time.clock()   

    # Create main output directory
    file_dir = r'%s/Forimputation/neighbor_methFeature_%d' % (gse, neighbor_region)
    os.makedirs(file_dir, exist_ok=True) 
    print(f"Main output directory: {file_dir}")

    # Read original data (core: separate cell column list)
    path = r'%s/%s' % (gse, ff)
    if not os.path.exists(path):
        print(f"Original data file {path} does not exist, exit")
        exit(1)
    data = pd.read_csv(path, header=0, sep='\t')
    
    # Handle duplicate columns/indexes
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()]
        print("Original data has duplicate columns, duplicates removed")
    if data.index.duplicated().any():
        data = data.reset_index(drop=True)
        print("Original data has duplicate indexes, indexes reset")
    
    data = reduce_mem(data)
    print(f"Original data: {data.shape} (rows × columns), column names: {list(data.columns)}")
    
    # Separate cell column list (key: start from 3rd column, exclude chrom and location)
    if list(data)[0] != 'chrom' or list(data)[1] != 'location':
        print(f"First two columns are not chrom/location, format error, exit")
        exit(1)
    cell_columns = list(data.columns[2:])  # Correct cell column list (e.g., ['cell1','cell2',...'bulk'])
    print(f"Cell column list: {cell_columns} (total {len(cell_columns)} cells)")
    if len(cell_columns) == 0:
        print(f"No cell columns found, exit")
        exit(1)

    # Define target chromosomes and 4 blocks (keep original logic)
    target_chroms = ['chr10', 'chr11']  
    original_data = data.copy()
    data = data[data['chrom'].isin(target_chroms)].drop_duplicates(['chrom', 'location'])
    print(f"Data after filtering chr10/chr11: {data.shape} rows")
    
    # Group by chromosome
    chrom_data_dict = {}
    split_row = {}
    for chrom in target_chroms:
        chrom_data = original_data[original_data['chrom'] == chrom].copy().reset_index(drop=True)
        chrom_data_dict[chrom] = chrom_data
        total_rows = len(chrom_data)
        split_row[chrom] = total_rows // 2
        print(f"{chrom}: {total_rows} rows, split point at row {split_row[chrom]}")
    
    chh = [
        {'chrom': 'chr10', 'row_min': None, 'row_max': split_row['chr10']},
        {'chrom': 'chr10', 'row_min': split_row['chr10'], 'row_max': None},
        {'chrom': 'chr11', 'row_min': None, 'row_max': split_row['chr11']},
        {'chrom': 'chr11', 'row_min': split_row['chr11'], 'row_max': None}
    ]

    # Generate neighborhood column names
    colname = [f'neighbor_{i}' for i in range(2 * neighbor_region)]  
    print(f"Neighborhood feature columns: {colname} (total {len(colname)} columns)")

    # Calculate cell column block indexes (based on cell column list, correct matching)
    ll = list(range(0, len(cell_columns), 2))  # Start from 0, step 2, cover all cell columns
    print(f"Cell column block indexes: {ll}")

    # Process each Block
    for block_idx in range(len(chh)):
        current_block = chh[block_idx]
        current_chrom = current_block['chrom']
        file_dir_Block = r'%s/Block%d' % (file_dir, block_idx)
        os.makedirs(file_dir_Block, exist_ok=True)
        print(f"\n=== Block{block_idx} (chrom {current_chrom}, {current_block['row_min']}-{current_block['row_max']}) ===")
        
        data1 = chrom_data_dict[current_chrom]
        print(f"Block{block_idx} data: {data1.shape} rows")
        
        # Multi-process: pass cell column list to solve index matching issue
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    test, 
                    data1,
                    current_block,
                    neighbor_region,
                    file_dir_Block,
                    colname,
                    cell_columns,  # Key: pass cell column list to test function
                    i
                ) for i in ll
            ]
            for j in as_completed(futures):
                try:
                    print(j.result())
                except Exception as e:
                    print(f"Process error: {str(e)}")    

    # Merge neighborhood feature results
    print("\n=== Merge neighborhood feature results ===")
    meragefiledir = file_dir
    filenames = [f'Block{i}' for i in range(len(chh)) if os.path.exists(r'%s/Block%d' % (file_dir, i))]
    if not filenames:
        print("No Block directories to merge")
    else:
        meragefiledir1 = r'%s/%s' % (file_dir, filenames[0])
        filenames1 = [f for f in os.listdir(meragefiledir1) if f.endswith('_neighbor_methFeature.txt')]
        if not filenames1:
            print("No neighborhood feature files to merge")
        else:
            file_dir_union = r'%s/localRegion_%d' % (file_dir, neighbor_region)
            os.makedirs(file_dir_union, exist_ok=True)
            print(f"Merge directory: {file_dir_union}")
            
            # Merge block indexes
            ll_union = list(range(0, len(filenames1), 2))
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(unionfile, file_dir_union, meragefiledir, filenames, filenames1, i) 
                    for i in ll_union
                ]
                for j in as_completed(futures):
                    try:
                        print(j.result())
                    except Exception as e:
                        print(f"Merge process error: {str(e)}")   
             
    # End
    elapsed = (time.clock() - start)
    print(f"\nAll processes completed, total time: {round(elapsed/60,2)} minutes ({round(elapsed,2)} seconds)")