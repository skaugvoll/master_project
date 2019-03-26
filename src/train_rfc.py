import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src import models


pipObj = Pipeline()


list_with_subjects = [
            '../data/input/nonshower_paul.7z',
            '../data/input/shower_atle.7z'
        ]


# unzip all data
# pipObj.unzip_multiple_directories(list_with_subjects, zip_to="../data/temp/")


list_with_subjects = [
            '../data/temp/nonshower_paul.7z',
            '../data/temp/shower_atle.7z'
        ]

testDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    list_with_subjects,
    back_keywords=['Back', "B"],
    thigh_keywords = ['Thigh', "T"],
    label_keywords = ['GoPro', "Labels", "intervals", "interval", "json"],
    out_path=None,
    merge_column = None,
    master_columns = ['bx', 'by', 'bz'],
    slave_columns = ['tx', 'ty', 'tz'],
    rearrange_columns_to = None,
    save=False,
    added_columns_name=["label"]
)