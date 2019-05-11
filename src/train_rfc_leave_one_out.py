import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")


import datetime
import numpy as np
from matplotlib import pyplot as plt
from src.pipeline.DataHandler import DataHandler
from src.pipeline.Pipeline import Pipeline

now = datetime.datetime.now()

pipObj = Pipeline()

list_with_subjects = [
    '../data/input/shower_atle.7z',
    '../data/input/nonshower_paul.7z',
    '../data/input/Thomas.7z',
    '../data/input/Thomas2.7z',
    '../data/input/Thomas3.7z',
    '../data/input/Sigve.7z',
    '../data/input/Sigve2.7z',
    '../data/input/Vegard.7z',
    '../data/input/Eivind.7z'
]

# ###unzip all data
unzipped_paths = pipObj.unzip_multiple_directories(list_with_subjects, zip_to="../data/temp/")


# unzipped_paths = [
#     '../data/temp/Thomas.7z/Thomas',
#     '../data/temp/Sigve2.7z/Sigve2',
# ]


# Trenger ikke downsample, da data er recorded i 50Hz
dataframes = pipObj.create_large_dataframe_from_multiple_input_directories(
    unzipped_paths,
    merge_column=None,
    save=False,
    added_columns_name=['label'],
    list=True
)


_, run_history = pipObj.train_RFC_model_leave_one_out(
    training_dataframe=dataframes,
    back_cols=['bx','by','bz'],
    thigh_cols=['tx', 'ty', 'tx'],
    back_temp_col='btemp',
    thigh_temp_col='ttemp',
    label_col='label',
    save_weights=True,
    save_to_path="trained_models/LOO_RFC_" + str(now.day) + "_" + str(now.month),
)

print("---------------------------------------------")
pipObj.plot_run_history(run_history, 3, 3, unzipped_paths, img_title="LOO_RFC_RUN_HISTORY.png")

print("AVG ACCURACY: ", run_history['AVG_ACCURACY'])

pipObj.calculate_avg_prec_recall_f1(run_history)

# write RUN_HISTORY to JSON FILE
pipObj.save_run_history_to_file(run_history, "RFC_RUN_HISTORY_LOO.json")

