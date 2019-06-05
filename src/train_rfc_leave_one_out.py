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
    '../data/input/xxx_x.7z',
    '../data/input/xxx_x.7z',
]

# ###unzip all data
# unzipped_paths = pipObj.unzip_multiple_directories(list_with_subjects, zip_to="../data/temp/")


unzipped_paths = [
    '../data/temp/xxx_x.7z/xxx_x',
    '../data/temp/xxx_x.7z/xxx_x',
]

# Trenger ikke downsample, da data er recorded i 50Hz
dataframes = pipObj.create_large_dataframe_from_multiple_input_directories(
    unzipped_paths,
    merge_column=None,
    save=False,
    added_columns_name=['label'],
    list=True,
    files=True
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
    data_names=unzipped_paths,
    rfc_memory_in_seconds=600,
    rfc_use_acc_data = True
)

print("---------------------------------------------")
pipObj.plot_run_history(run_history, 2, 2, unzipped_paths, img_title="RFC_RUN_LOO_WD_WM_P_HISTORY.png")

print("AVG ACCURACY: ", run_history['AVG_ACCURACY'])

pipObj.calculate_avg_prec_recall_f1(run_history, add_to_history=True)

# write RUN_HISTORY to JSON FILE
pipObj.save_run_history_to_file(run_history, "RFC_RUN_LOO_WD_WM_P.json")

print("AVG PRECISION: ", run_history['AVG_PRECISION'])

