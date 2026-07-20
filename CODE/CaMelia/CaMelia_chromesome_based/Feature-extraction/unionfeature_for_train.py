# -*- coding: utf-8 -*-
from __future__ import division
from sys import argv
import pandas as pd
import numpy as np
import os, time
import warnings
warnings.filterwarnings('ignore')

###########################################
def reduce_mem(df):
    """Reduce DataFrame memory usage by downcasting numeric data types to smallest possible size"""
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
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
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction_percent = 100 * (start_mem - end_mem) / start_mem
    time_spent = (time.time() - starttime) / 60
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction), time spent: {:2.2f} min'.format(
        end_mem, reduction_percent, time_spent
    ))
    return df

###########################################
if __name__ == '__main__': 
    # All data located in the same directory
    DataPath = r'%s' % argv[1]
    # Input data filename
    InputDataName = '%s' % argv[2]  
    # Neighborhood range for feature calculation
    neighbor_region = int(argv[3])
    
    gse = DataPath
    ff  = InputDataName	    
    region = neighbor_region 
    
    name = ff.split('.')[0].split('_')[-1]
    
    # Load main dataset
    path = r'%s/%s' % (gse, ff)
    data = pd.read_csv(path, header=0, sep='\t')
    data = reduce_mem(data)
    
    # Ensure first column is 'chrom' (remove invalid first column if needed)
    if list(data)[0] != 'chrom':
        del data['%s' % list(data)[0]]
  
    # Extract cell column names (from 3rd column to second last column)
    cell_num = list(data)[2:(len(list(data))-1)]
    
    # Create output directory
    file_dir = r'%s/Available_Train_dataset/region%d' % (gse, region)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)   
        
    # Process each cell column
    for i in range(len(cell_num)):            
        # Load local methylation features
        path = r'%s/region10_localmatched_morethan08/%s.txt' % (gse, cell_num[i])
        local_methFeature = pd.read_csv(path, header=0, sep='\t')
        local_methFeature = reduce_mem(local_methFeature)
        local_methFeature = local_methFeature.rename(columns={'aver_meth':'local_methFeature'})
        local_methFeature = local_methFeature[['chrom','location','local_methFeature']]

        # Load neighborhood methylation features
        path = r'%s/neighbor_methFeature_%d/localRegion_%s/%s_neighbor_methFeature.txt' % (gse, region, region, cell_num[i])
        neighbor_methFeature = pd.read_csv(path, header=0, sep='\t')
        neighbor_methFeature = reduce_mem(neighbor_methFeature)

        # Merge main data with local and neighborhood features (inner join on chrom/location)
        data_all = pd.merge(data[['chrom','location','%s' % cell_num[i]]], local_methFeature)
        data_all = pd.merge(data_all, neighbor_methFeature, how='inner', on=['chrom','location'])
        
        # Save merged dataset
        data_all.to_csv(r'%s/%s.txt' % (file_dir, cell_num[i]), sep='\t', header=True, index=False)    
        del data_all