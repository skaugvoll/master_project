import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.resampler import main as resampler
import pandas as pd

pipObj = Pipeline()

# list_with_subjects = [
#     '../data/input/4000181.7z'
#     # '../data/input/training_data/006'
# ]
#
# ###unzip all data
# unzipped_paths = pipObj.unzip_multiple_directories(list_with_subjects, zip_to="../data/temp/")
# print(unzipped_paths)

subject = "022"
#
# resample = [
#     '../data/input/training_data/'+subject
#     ]
#
# trainDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
#     resample,
#     merge_column='time',
#     rearrange_columns_to=[
#         'time',
#         'bx',
#         'by',
#         'bz',
#         'tx',
#         'ty',
#         'tz',
#         'btemp',
#         'ttemp'
#     ],
#     save=True,
#     out_path='../data/temp/merged/res'+subject+'.csv',
#     # added_columns_name=['labels'],m
#     list=False
# )

# csv = pd.read_csv('../data/temp/merged/res'+subject+'.csv', header=0)
#
# csv.index = pd.date_range( start=pd.Timestamp.now(), periods=len(csv), freq=pd.Timedelta( seconds=1/100))
#
# csv.to_csv('../data/temp/merged/res'+subject+'.csv')

resampler("fourier", 100, 50, 20000, '../data/temp/merged/res'+subject+'.csv', '../data/temp/merged/resampled'+subject+'.csv', ['label'], save=True)