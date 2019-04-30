import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline

pipObj = Pipeline()

pipObj.downsampleData(
    input_csv_path="../data/temp/merged/res006.csv",
    out_csv_path="../data/temp/merged/resampled006.csv",
    discrete_columns=['label']
)