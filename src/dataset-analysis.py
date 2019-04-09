import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")


import numpy as np
import pandas as pd
from src.pipeline.Pipeline import Pipeline

pipObj = Pipeline()

data = [
    # '../data/temp/shower_atle.7z/shower_atle',
    # '../data/temp/nonshower_paul.7z/nonshower_paul',
    # '../data/temp/Thomas.7z/Thomas',
    # '../data/temp/Thomas2.7z/Thomas2',
    '../data/temp/Sigve.7z/Sigve'
]

dataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    data,
    merge_column='time',
    master_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz'],
    slave_columns=['time', 'bx1', 'by1', 'bz1', 'btemp'],
    slave2_columns=['time', 'tx1', 'ty1', 'tz1', 'ttemp'],
    rearrange_columns_to=[
                    'time',
                    'bx',
                    'by',
                    'bz',
                    'tx',
                    'ty',
                    'tz',
                    'btemp',
                    'ttemp'
                ],
    save=False,
    added_columns_name=['labels']
)

# print(dataframe.describe())
# print(dataframe.labels)

labels = {}
for row in dataframe['labels']:
    labels[str(row[0])] = labels.get(str(row[0]), 0) + 1
    
print("Label distribution: ", labels)