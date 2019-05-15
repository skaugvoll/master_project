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
# unzipped_paths = pipObj.unzip_multiple_directories(list_with_subjects, zip_to="../data/temp/")


unzipped_paths = [
    '../data/temp/P1_atle.7z/P1_atle',
    '../data/temp/P1_vegar.7z/P1_vegar',
    '../data/temp/P2_atle.7z/P2_atle',
    '../data/temp/P2_vegar.7z/P2_vegar',
    # '../data/temp/Sigve.7z/Sigve',
    # '../data/temp/Thomas.7z/Thomas',
    # '../data/temp/Thomas2.7z/Thomas2',
    # '../data/temp/shower_atle.7z/shower_atle',
    # '../data/temp/nonshower_paul.7z/nonshower_paul',
    # '../data/temp/Sigve2.7z/Sigve2',
    # '../data/temp/Thomas3.7z/Thomas3',
    # '../data/temp/Vegard.7z/Vegard',
    # '../data/temp/Eivind.7z/Eivind',
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

