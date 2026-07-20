# -*- coding: utf-8 -*-
"""
Process chr10 and chr11 by splitting into half according to original order and row count without sorting
Each chromosome is divided into 2 blocks, total 4 blocks (Block0-3)
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

####################################
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
####################################    
@jit
def chu(a, n):
    """Calculate and return rounded value of a/(2*n)"""
    return round(a/(2*n), 4)

@jit
def corre(a, b):
    """Calculate and return correlation coefficient of a and b"""
    return round(sum(a==b)/len(a), 4)   

@jit
def getlog2(a):    
    """Calculate and return rounded value of log2(a+1.01)"""
    return round(np.log2(a+1.01), 4)

####################################
# Merge files
def unionfile(file_dir, meragefiledir, filenames, filenames1, i1):
    """Merge multiple files and remove duplicates"""
    if i1 + 2 >= len(filenames1):
        i2 = len(filenames1)
    else:
        i2 = i1 + 2   
    for j in range(i1, i2):
        df = pd.DataFrame(np.random.randn(0, 2), columns=['chrom','location'])       
        for i in range(len(filenames)):
            path = r'%s/%s/%s' %(meragefiledir, filenames[i], filenames1[j])
            data = pd.read_csv(path, header=0, sep='\t')
            data = reduce_mem(data)       
            df = pd.concat([df, data], axis=0, join='outer', ignore_index=False)
                    
        df = df.drop_duplicates(['chrom', 'location'])
        df.to_csv(r'%s/%s' % (file_dir, filenames1[j]), sep='\t', header=True, index=False)     
    return (f'Process {i1}: Completed!')

# Get local methylation features
def test(data, block_info, neighbor_region, file_dir_Block, file_dir_for_impu, i1):    
    """Process data of a single Block to calculate local methylation features (fixed version)"""
    # Get chromosome and row range info of current block
    current_chrom = block_info['chrom']  # Chromosome to process (chr10 or chr11)
    row_min = block_info['row_min']      # Min row index (None = no lower limit)
    row_max = block_info['row_max']      # Max row index (None = no upper limit)
    
    # Determine column index range to process
    if i1 + 2 >= (len(list(data)) - 1):
        i2 = len(list(data)) - 1 
    else:
        i2 = i1 + 2
        
    for i in range(i1, i2):
        current_col = list(data)[i]
        print(f'{current_col} (chrom {current_chrom}, row range: {row_min}-{row_max}): Start processing!')       
        # Initialize result DataFrame (ensure chrom and location are first columns)
        data_all = pd.DataFrame(columns=['chrom', 'location'])  
        data_all_r_for_impu = pd.DataFrame(columns=['chrom', 'location'])      
        
        for k in range(i + 1, len(list(data))):
            # Define cell pair name
            pair_col = list(data)[k]
            r_col_name = f'{current_col}_{pair_col}_r'
            methy_col_name = f'{current_col}_{pair_col}_methy'
            
            # Select columns to process (use column names directly to avoid index errors)
            data1 = data[['chrom', 'location', current_col, pair_col]]
            data1 = reduce_mem(data1) 
            
            # Initialize result DataFrame (explicit column order)
            data_chr = pd.DataFrame(
                columns=['chrom', 'location', r_col_name, methy_col_name]
            )
            data_r_for_impu = pd.DataFrame(
                columns=['chrom', 'location', r_col_name]
            )
            
            # Filter data of current chromosome
            data2 = data1[data1['chrom'] == current_chrom].copy() 
            
            # Filter by row range (keep original order)
            if row_min is not None:
                data2 = data2.iloc[row_min:]  # Start from row_min (inclusive)
            if row_max is not None:
                data2 = data2.iloc[:row_max]  # End at row_max (exclusive)
                
            # Data preprocessing (only drop null values in target columns)
            data2 = data2.dropna(subset=[current_col, pair_col])
            if len(data2) == 0:
                print(f"Cell pair {current_col}-{pair_col} has no valid data (too many null values), skip")
                continue
                
            # Sort by location for neighborhood calculation (internal use only)
            data2_sorted = data2.sort_values(by='location')

            # Calculate neighborhood features (only if data volume is sufficient)
            if len(data2_sorted) > 2 * neighbor_region:
                loca = list(data2_sorted['location'])
                data2_value = data2_sorted.values
                
                # Extract basic location info (chrom and location)
                base_data = pd.DataFrame(
                    data2_value[neighbor_region:len(loca)-neighbor_region, 0:2],
                    columns=['chrom', 'location']
                )
                
                # Initialize calculation result lists
                r_values = []
                methy_values = []
                                        
                # Initialize neighborhood indexes
                ind_l = list(range(0, neighbor_region))
                ind_r = list(range(neighbor_region + 1, 2 * neighbor_region + 1))                    
                ind = ind_l + ind_r                    
                
                # Calculate match count of initial window
                cell_a = pd.Series(data2_value[ind, 2])  # Values of current_col
                cell_b = pd.Series(data2_value[ind, 3])  # Values of pair_col
                rr = sum(cell_a == cell_b)
                
                # Calculate result of first window
                r_values.append(chu(rr, neighbor_region))
                methy_values.append(getlog2(data2_value[neighbor_region, 3]))
                
                # Sliding window calculation for subsequent results
                cell_a_l = list(data2_value[ind_l, 2])
                cell_a_r = list(data2_value[ind_r, 2]) 
                cell_b_l = list(data2_value[ind_l, 3])
                cell_b_r = list(data2_value[ind_r, 3])     
                
                for l in range(neighbor_region + 1, len(loca) - neighbor_region):
                    # Update match count
                    if cell_a_l[0] == cell_b_l[0]:
                        rr -= 1
                    if data2_value[l-1, 2] == data2_value[l-1, 3]:
                        rr += 1    
                    if cell_a_r[0] == cell_b_r[0]:
                        rr -= 1   
                    if data2_value[l + neighbor_region, 2] == data2_value[l + neighbor_region, 3]:
                        rr += 1
                   
                    # Update window
                    ind_l.pop(0)
                    ind_l.append(l - 1)
                    cell_a_l.pop(0)
                    cell_a_l.append(data2_value[l - 1, 2])
                    cell_a_r.pop(0)
                    cell_a_r.append(data2_value[l + neighbor_region, 2]) 
                    ind_r.pop(0)
                    ind_r.append(l + neighbor_region)
                    cell_b_l.pop(0)
                    cell_b_l.append(data2_value[l - 1, 3])
                    cell_b_r.pop(0)
                    cell_b_r.append(data2_value[l + neighbor_region, 3])  
                    
                    # Calculate result of current window
                    r_for2 = chu(rr, neighbor_region)
                    r_values.append(r_for2)
                    methy_values.append(getlog2(data2_value[l, 3]))
                                                                                   
                # Fill results to DataFrame (ensure correct column order)
                base_data[r_col_name] = r_values
                base_data[methy_col_name] = methy_values      
                
                # Merge to result
                data_chr = pd.concat([data_chr, base_data], ignore_index=True)
                data_r_for_impu = pd.concat([data_r_for_impu, base_data[['chrom', 'location', r_col_name]]], ignore_index=True)

            # Merge to global result
            if not data_chr.empty:
                data_chr = reduce_mem(data_chr)
                if data_all.empty:
                    data_all = data_chr.copy()
                else:
                    data_all = pd.merge(data_all, data_chr, how='outer', on=['chrom', 'location'])
                                
            if not data_r_for_impu.empty:
                data_r_for_impu = reduce_mem(data_r_for_impu)            	
                if data_all_r_for_impu.empty:
                    data_all_r_for_impu = data_r_for_impu.copy()
                else:
                    data_all_r_for_impu = pd.merge(
                        data_all_r_for_impu, 
                        data_r_for_impu, 
                        how='outer', 
                        on=['chrom', 'location']
                    )
                                
            print(f'{current_col}-{pair_col}: Completed (generated {len(r_values)} valid _r values)')
            
        # Save local methylation feature file
        if not data_all.empty:
            data_all.to_csv(
                r'%s/%s_local_methFeature.txt' % (file_dir_Block, current_col),
                sep='\t', header=True, index=False
            ) 
            print(f'Saved {current_col}_local_methFeature.txt ({len(data_all)} rows)')
        else:
            print(f'{current_col} has no valid methylation feature data, skip saving')
                
        # Save _r file (key fix: ensure correct column order and values)
        if not data_all_r_for_impu.empty:
            # Force column order: chrom → location → all _r columns
            core_cols = ['chrom', 'location']
            r_cols = [col for col in data_all_r_for_impu.columns if col not in core_cols]
            data_all_r_for_impu = data_all_r_for_impu[core_cols + r_cols]
            
            data_all_r_for_impu.to_csv(
                r'%s/%s_local_r.txt' % (file_dir_for_impu, current_col),
                sep='\t', header=True, index=False
            ) 
            print(f'Saved {current_col}_local_r.txt ({len(data_all_r_for_impu)} rows, contains {len(r_cols)} _r columns)')
        else:
            print(f'{current_col} has no valid _r data, skip saving _local_r.txt')
        
        print(f'{current_col} (chrom {current_chrom}, row range: {row_min}-{row_max}): Processing completed!')
    return (f'Block{i1}: Completed!')

if __name__ == '__main__': 
    # Command line parameters
    DataPath = r'%s' % argv[1]          # Data path
    InputDataName = '%s' % argv[2]      # Input file name
    neighbor_region = int(argv[3])      # Neighborhood range
    threshold = float(argv[4])          # Correlation coefficient threshold
   
    gse = DataPath
    ff = InputDataName	
	   
    start = time.clock()
    
    # Create output directory
    file_dir = r'%s/local_methFeature_spares' % gse
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # Read data (keep original order)
    path = r'%s/%s' % (gse, ff)
    data = pd.read_csv(path, header=0, sep='\t')
    data = reduce_mem(data)
    
    # Ensure first column is chrom
    if list(data)[0] != 'chrom':
        del data['%s' % list(data)[0]]
    
    # Process only chr10 and chr11, keep original order
    target_chroms = ['chr10', 'chr11']  
    
    # Save original data (no sorting)
    original_data = data.copy()
    
    # Group by chromosome and save original row indexes (key modification: no sorting, keep original order)
    chrom_data_dict = {}
    for chrom in target_chroms:
        # Filter data of the chromosome, keep original order
        chrom_data = original_data[original_data['chrom'] == chrom].copy()
        # Keep original row indexes
        chrom_data = chrom_data.reset_index(drop=True)  # Reset index to 0,1,2...
        chrom_data_dict[chrom] = chrom_data
        print(f"Total rows of {chrom}: {len(chrom_data)}")
    
    # Automatically calculate row split point for each chromosome (split into half by row count, no sorting)
    split_row = {}
    for chrom in target_chroms:
        chrom_data = chrom_data_dict[chrom]
        total_rows = len(chrom_data)
        if total_rows <= 1:
            # Too few data rows, no split
            split_row[chrom] = 0
        else:
            # Split into half by total rows (integer division)
            split_row[chrom] = total_rows // 2
        print(f"Row split point of {chrom}: Row {split_row[chrom]} (count from 0)")
    
    # Build 4 Blocks (Block0-3), use row index range instead of position range
    chh = [
        # Block0: First half of chr10 (row 0 to split_row['chr10'])
        {'chrom': 'chr10', 'row_min': None, 'row_max': split_row['chr10']},
        # Block1: Second half of chr10 (row split_row['chr10'] to end)
        {'chrom': 'chr10', 'row_min': split_row['chr10'], 'row_max': None},
        # Block2: First half of chr11 (row 0 to split_row['chr11'])
        {'chrom': 'chr11', 'row_min': None, 'row_max': split_row['chr11']},
        # Block3: Second half of chr11 (row split_row['chr11'] to end)
        {'chrom': 'chr11', 'row_min': split_row['chr11'], 'row_max': None}
    ]
              
    # Extract cell numbers
    cell_num = []
    for i in range(2, len(list(data)) - 1):
        cell_num.append(list(data)[i])
    cell_num = list(set(cell_num))  
              
    print('Getting local methylation features: Start!')

    # Calculate column block indexes
    a = len(list(data)) - 3
    ll = []
    k = 0
    for i in range(int(math.floor(a / 2))):
        k += 2
        ll.append(k)
    if a % 2 != 0:
        ll.append(k + 1)
    print("Cell column block indexes ll：", ll)
          
    # Process each Block
    for block_idx in range(len(chh)):
        current_block = chh[block_idx]
        chrom = current_block['chrom']
        print(f'Block{block_idx} (chrom {chrom}, row range: {current_block["row_min"]}-{current_block["row_max"]}): Start processing!')
        
        # Create output directory for current Block
        file_dir_Block = r'%s/Block%d' % (file_dir, block_idx)
        if not os.path.exists(file_dir_Block):
            os.makedirs(file_dir_Block)
        
        # Create impu directory for current Block
        file_dir_for_impu = r'%s/for_impu_r/Block%d' % (gse, block_idx)
        if not os.path.exists(file_dir_for_impu):
            os.makedirs(file_dir_for_impu)                    
        
        # Get original data of current chromosome (no sorting)
        data1 = chrom_data_dict[chrom]
        
        # Verify data volume of current Block
        if current_block['row_min'] is not None and current_block['row_max'] is not None:
            temp_data = data1.iloc[current_block['row_min']:current_block['row_max']]
        elif current_block['row_min'] is not None:
            temp_data = data1.iloc[current_block['row_min']:]
        elif current_block['row_max'] is not None:
            temp_data = data1.iloc[:current_block['row_max']]
        else:
            temp_data = data1.copy()
        print(f'Data volume of Block{block_idx}: {len(temp_data)} rows')
        
        # Multi-process processing for current Block
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    test, 
                    data1, 
                    current_block, 
                    neighbor_region, 
                    file_dir_Block, 
                    file_dir_for_impu, 
                    i
                ) for i in ll
            ]
            for j in as_completed(futures):
                print(j.result()) 
    
    elapsed = (time.clock() - start)
    print(f"Processing time: {round(elapsed, 4)} seconds")
    print("Getting local methylation features: Completed!" )

    # Release memory
    del data
    del data1

    ############################################################################
    # Merge local methylation features
    print("Merging local methylation features: Start!" )

    meragefiledir = r'%s' % file_dir
    filenames = os.listdir(meragefiledir)
    
    meragefiledir1 = r'%s/%s' % (file_dir, filenames[0])
    filenames1 = os.listdir(meragefiledir1)

    file_dir = r'%s/local_methFeature/localRegion_%d' % (gse, neighbor_region)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)  
  
    a = len(filenames1)
    ll = [0]
    k = 0
    for i in range(int(math.floor(a / 2))):
        k += 2
        ll.append(k)
    if a % 2 != 0:
        ll.append(k + 1)
    
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(unionfile, file_dir, meragefiledir, filenames, filenames1, i) 
            for i in ll
        ]
        for j in as_completed(futures):
            print(j.result())   
      
    print("Merging local methylation features: Completed!" )
    ############################################################################
    
    # Further merge all results
    region = neighbor_region
    meragefiledir = r'%s/local_methFeature/localRegion_%d' % (gse, region)
    filenames = os.listdir(meragefiledir)

    path = r'%s/local_methFeature/localRegion_%d/%s' % (gse, region, filenames[0])
    data = pd.read_csv(path, header=0, sep='\t')
    data = reduce_mem(data)	
    for i in range(1, len(filenames)):
        path = r'%s/%s' % (meragefiledir, filenames[i])
        df = pd.read_csv(path, header=0, sep='\t')
        df = reduce_mem(df)
        data = pd.merge(data, df, how='outer', on=['chrom', 'location'])
    print("Final merge completed" )  
     
    ############################################################################    
    # Split results by cell
    file_dir_1 = r'%s/local_methFeature_cellbycell/region%d/corr' % (gse, region)
    if not os.path.exists(file_dir_1):
        os.makedirs(file_dir_1)

    file_dir_2 = r'%s/local_methFeature_cellbycell/region%d/methy' % (gse, region)
    if not os.path.exists(file_dir_2):
        os.makedirs(file_dir_2)
		    
    for i in range(len(cell_num)): 
        cell_r = []
        cell_meth = []
        for j in range(2, len(list(data))):
            col_name = list(data)[j]
            parts = col_name.split('_')
        
            # Filter columns related to current cell
            if len(parts) > 1:
                if (parts[0] == cell_num[i]) or (parts[1] == cell_num[i]):
                    if parts[-1] == 'r':
                        cell_r.append(col_name)
                    if parts[-1] == 'methy':
                        cell_meth.append(col_name)
            else:
                print(f"Invalid column name format: {col_name}")
         
        
        df_r = data[['chrom', 'location'] + cell_r] 
        df_m = data[['chrom', 'location'] + cell_meth] 
        
        # Remove rows with all null values
        df_r = df_r.dropna(axis=0, subset=list(df_r)[2:], how='all') 
        df_m = df_m.dropna(axis=0, subset=list(df_m)[2:], how='all') 
        
        # Save results
        df_r.to_csv(
            r'%s/%s_r.txt' % (file_dir_1, cell_num[i]),
            sep='\t', header=True, index=False
        )    
        df_m.to_csv(
            r'%s/%s_m.txt' % (file_dir_2, cell_num[i]),
            sep='\t', header=True, index=False
        )    
    print("Splitting by cell completed" )     
    ############################################################################
    
    # Calculate matching results
    for i in range(len(cell_num)):
        # Read correlation data
        path = r'%s/%s_r.txt' % (file_dir_1, cell_num[i])
        data_r = pd.read_csv(path, header=0, sep='\t')

        # Read methylation data
        path = r'%s/%s_m.txt' % (file_dir_2, cell_num[i])
        data_m = pd.read_csv(path, header=0, sep='\t')

        df_r = data_r[['chrom', 'location']]
    
        name = list(data_r)[2:]
               
        # Calculate matching results
        data_r.fillna(0, inplace=True)
        data_m.fillna(0, inplace=True)
    
        r_value = data_r.values
        m_value = data_m.values    
        
        col_ind = []
        col_ind_m = []
        for j in range(len(r_value)):       
            a = np.array(r_value[j, 2:])
            b = np.array(m_value[j, 2:])
            c = np.where(a >= threshold)
            
            if len(c[0]) != 0:
                ss = 0
                for k in range(len(c[0])):           
                    ss += a[c[0][k]] * b[c[0][k]]
                col_ind_m.append(round(ss / len(c[0]), 4))    
                if len(c[0]) == 1 :
                    if name[c[0][0]].split('_')[-2] == cell_num[i]:
                        col_ind.append(name[c[0][0]].split('_')[0])
                    else:
                        col_ind.append(name[c[0][0]].split('_')[-2])
                else:
                    col_ind.append(len(c[0]))
            else:
                col_ind.append(np.nan)
                col_ind_m.append(np.nan)

        df_r['aver_meth'] = col_ind_m
        df_r['matched_cell'] = col_ind
                
        # Save final results
        file_dir_final = r'%s/region%d_localmatched_morethan08' % (gse, region)
        if not os.path.exists(file_dir_final):
            os.makedirs(file_dir_final)
        
        path = r'%s/%s.txt' % (file_dir_final, cell_num[i])
        df_r.to_csv(path, sep='\t', header=True, index=False)   
    print("All processing completed!" )