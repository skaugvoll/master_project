import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")


import numpy as np
import pandas as pd
from src.pipeline.Pipeline import Pipeline

pipObj = Pipeline()



list_with_subjects = [
    '../data/input/Sigve.7z',
    '../data/input/Sigve2.7z',
    '../data/input/Thomas.7z',
    '../data/input/Thomas2.7z',
    '../data/input/shower_atle.7z',
    '../data/input/nonshower_paul.7z',

]
#
# data = pipObj.unzip_multiple_directories(list_with_subjects, zip_to="../data/temp/")
# print(unzipped_paths)



data = [
    # '../data/temp/shower_atle.7z/shower_atle',
    # '../data/temp/nonshower_paul.7z/nonshower_paul',
    # '../data/temp/Thomas.7z/Thomas',
    # '../data/temp/Thomas2.7z/Thomas2',
    # '../data/temp/Thomas3.7z/Thomas3',
    # '../data/temp/Sigve.7z/Sigve',
    # '../data/temp/Sigve2.7z/Sigve2'
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
    added_columns_name=['labels'],
    # files=True,
    drop_non_labels=False
)

print(dataframe.describe())
# print(dataframe.labels)

# import matplotlib.pyplot as plt
# dataframe[['btemp', 'ttemp']].plot(style=['r-', 'g--'])
# plt.savefig('Temp.png')

rows = dataframe.shape[0]

print('Dataframe sneakpeak')
print(dataframe.head(5))


nanDF = pd.DataFrame(dataframe['labels'].astype(float))
nan = list(nanDF['labels'].index[nanDF['labels'].apply(np.isnan)])
print('Nan: {}'.format(len(nan)))


dataframe.dropna(subset=['labels'], inplace=True)

print('Rows {}'.format(rows))
print('Nan + Nrows {}'.format(len(nan) + dataframe.shape[0]))

labels = {}
for row in dataframe['labels']:
    labels[str(row[0])] = labels.get(str(row[0]), 0) + 1

print("Label distribution: ", labels)