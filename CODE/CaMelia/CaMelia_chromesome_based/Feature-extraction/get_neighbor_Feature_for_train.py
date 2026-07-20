# -*- coding: utf-8 -*-
"""
Process only chr10 and chr11, split each chromosome into half according to original row count (without sorting)
Divide into a total of 4 blocks (Block0-3) for processing
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

##############################################################
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
# Merge files
def unionfile(file_dir, meragefiledir, filenames, filenames1, i1):
    if i1 + 2 >= len(filenames1):
        i2 = len(filenames1)
    else:
        i2 = i1 + 2   
    for j in range(i1, i2):
        df = pd.DataFrame(np.random.randn(0, 2), columns=['chrom', 'location'])       
        for i in range(len(filenames)):
            path = r'%s/%s/%s' %(meragefiledir, filenames[i], filenames1[j])
            data = pd.read_csv(path, header=0, sep='\t')
            df = pd.concat([df, data], axis=0, join='outer', ignore_index=False)
        df = df.drop_duplicates(['chrom', 'location'])
        df[list(df)[2:]] = df[list(df)[2:]].astype('float16')
        df[list(df)[2:]] = df[list(df)[2:]].round(4)        
        df.to_csv(r'%s/%s' % (file_dir, filenames1[j]), sep='\t', header=True, index=False)     
    return (f'Process {i1}: Completed!')


def test(data, block_info, neighbor_region, file_dir, colname, i1):
    """
    Process data of a single block
    block_info: Contains information of chromosome (chrom), start row (row_min) and end row (row_max)
    """
    current_chrom = block_info['chrom']  # Chromosome to process currently
    row_min = block_info['row_min']      # Start row index
    row_max = block_info['row_max']      # End row index
    
    if i1 + 2 >= (len(list(data)) - 1):
        i2 = len(list(data)) - 1 
    else:
        i2 = i1 + 2
        
    for i in range(i1, i2): 
        print(f"{list(data)[i]} (chrom {current_chrom}, row range: {row_min}-{row_max}): Start processing!")
        # Filter target column data of current chromosome (maintain original order)
        data3 = data[['chrom', 'location', '%s' % list(data)[i]]]               
        d_c = pd.DataFrame(
            np.random.randn(0, 2 + 2*neighbor_region), 
            columns=['chrom', 'location'] + colname
        ) 
        
        # Process only the specified chromosome of current block
        data2 = data3[data3['chrom'] == current_chrom].copy()
        # Filter by row index range (Core Modification: No sorting, split by original row index directly)
        if row_min is not None:
            data2 = data2.iloc[row_min:]  # Start from row_min (inclusive)
        if row_max is not None:
            data2 = data2.iloc[:row_max]  # End at row_max (exclusive)
        
        # Data preprocessing (only delete null values, keep order unchanged)
        data2 = data2.dropna(subset=['%s' % list(data3)[2]])
        data2 = data2.reindex()  # Reset index but keep original order
        
        # Neighborhood calculation requires sorting by location (local sorting, no impact on block division)
        data2_sorted = data2.sort_values(by='location')
        
        if len(data2_sorted) > 2 * neighbor_region:
            loca = list(data2_sorted['location'])
            data2_value = data2_sorted.values
            
            data_r = pd.DataFrame(
                np.random.randn(0, 2 + 2*neighbor_region), 
                columns=['chrom', 'location'] + colname
            )
            data_r['chrom'] = data2_value[neighbor_region:len(loca)-neighbor_region, 0]
            data_r['location'] = data2_value[neighbor_region:len(loca)-neighbor_region, 1]
            
            data_r_value = data_r.values
                   
            for l in range(neighbor_region, len(loca) - neighbor_region):
                ind = list(range(l - neighbor_region, l + neighbor_region + 1))
                ind.remove(l)
                location_1 = list(data2_value[l - neighbor_region:l, 1])
                location = data2_value[l, 1]
                location_2 = list(data2_value[l + 1:l + neighbor_region + 1, 1])
                
                if (location_2[-1] - location) >= (location - location_1[0]):
                    maxdis = location_2[-1] - location 
                else:
                    maxdis = location - location_1[0] 
                            
                loca_comb = location_1 + location_2
                mmm = list(data2_value[ind, 2])
                            
                for p in range(len(mmm)):
                   m4 = hah(mmm[p], loca_comb[p], location, maxdis)
                   data_r_value[l - neighbor_region, p + 2] = round(m4, 4)
                                                                                                                        
            data_r = pd.DataFrame(data_r_value, columns=['chrom', 'location'] + colname)
            d_c = pd.merge(d_c, data_r, how='outer')  
      
        d_c.to_csv(
            r'%s/%s_neighbor_methFeature.txt' % (file_dir, list(data)[i]),
            sep='\t', header=True, index=False
        )   
        print(f"{list(data)[i]} (chrom {current_chrom}, row range: {row_min}-{row_max}): Processing completed!")
    return (f'Process {i1}: Completed!')

##############################################################

if __name__ == '__main__':
    # Command line parameters
    DataPath = r'%s' % argv[1]          # Data path
    InputDataName = '%s' % argv[2]      # Input file name
    neighbor_region = int(argv[3])      # Neighborhood range    

    gse = DataPath
    ff = InputDataName	
    
    start = time.clock()    
   
    # Create output directory
    file_dir = r'%s/neighbor_methFeature_%d' % (gse, neighbor_region)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir) 

    # Read data (maintain original order)
    path = r'%s/%s' % (gse, ff)
    data = pd.read_csv(path, header=0, sep='\t')
    data = reduce_mem(data)
    
    # Ensure the first column is chrom
    if list(data)[0] != 'chrom':
        del data['%s' % list(data)[0]]
   
    # Process only chr10 and chr11 (Core Modification 1: Specify target chromosomes)
    target_chroms = ['chr10', 'chr11']
    
    # Filter data to keep only target chromosomes (maintain original order)
    original_data = data.copy()  # Save original data
    data = data[data.chrom.isin(target_chroms)]          
    data = data.drop_duplicates(['chrom', 'location'])
    
    # Group by chromosome and save original row indexes (Core Modification 2: No sorting, maintain original order)
    chrom_data_dict = {}
    for chrom in target_chroms:
        # Filter data of the chromosome, maintain original order
        chrom_data = original_data[original_data['chrom'] == chrom].copy()
        # Reset index to 0,1,2... (maintain original order)
        chrom_data = chrom_data.reset_index(drop=True)
        chrom_data_dict[chrom] = chrom_data
        print(f"Total rows of {chrom}: {len(chrom_data)}")
    
    # Automatically calculate row split point for each chromosome (split into half by row count)
    split_row = {}
    for chrom in target_chroms:
        chrom_data = chrom_data_dict[chrom]
        total_rows = len(chrom_data)
        if total_rows <= 1:
            split_row[chrom] = 0  # No split if too few data rows
        else:
            split_row[chrom] = total_rows // 2  # Integer division to get middle row
        print(f"Row split point of {chrom}: Row {split_row[chrom]} (counting from 0)")
    
    # Build 4 Blocks (Block0-3) (Core Modification 3: Define 4 processing blocks)
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
                            
    # Generate neighborhood column names
    colname = []
    for i in range(2 * neighbor_region):
        colname.append('neighbor_%d' % i)
                            
    # Calculate column block indexes
    a = len(list(data)) - 3
    ll = []
    k = 0
    for i in range(int(math.floor(a / 2))):
        k += 2
        ll.append(k)
    if a % 2 != 0:
        ll.append(k + 1) 
    
    # Process each Block
    for block_idx in range(len(chh)):
        current_block = chh[block_idx]
        chrom = current_block['chrom']
        print(f'Block{block_idx} (chrom {chrom}, row range: {current_block["row_min"]}-{current_block["row_max"]}): Start!')
        
        # Create output directory for current Block
        file_dir_Block = r'%s/Block%d' % (file_dir, block_idx)
        if not os.path.exists(file_dir_Block):
            os.makedirs(file_dir_Block)    
        
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
                    current_block,  # Pass block info (chromosome + row range)
                    neighbor_region, 
                    file_dir_Block, 
                    colname, 
                    i
                ) for i in ll
            ]
            for j in as_completed(futures):
                print(j.result())     

    elapsed = (time.clock() - start)
    print(f"Processing time: {round(elapsed, 4)} seconds")
    
    # Release memory
    del data
    del data1

    ####################################    
    # Merge neighborhood methylation features
    print("Merging neighborhood methylation features: Start!" )

    meragefiledir = r'%s' % file_dir
    filenames = os.listdir(meragefiledir)
    
    meragefiledir1 = r'%s/%s' % (file_dir, filenames[0])
    filenames1 = os.listdir(meragefiledir1)

    file_dir_union = r'%s/neighbor_methFeature_%d/localRegion_%d' % (gse, neighbor_region, neighbor_region)
    if not os.path.exists(file_dir_union):
        os.makedirs(file_dir_union)  
  
    a = len(filenames1)
    ll = [0]
    k = 0
    for i in range(int(math.floor(a / 2))):
        k += 2
        ll.append(k)
    if a % 2 != 0:
        ll.append(k + 1)
    
    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(unionfile, file_dir_union, meragefiledir, filenames, filenames1, i) 
            for i in ll
        ]
        for j in as_completed(futures):
            print(j.result())   
         
    print("Merging neighborhood methylation features: Completed!" )