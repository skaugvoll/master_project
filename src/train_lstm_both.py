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

train_list_with_subjects = [
    '../data/input/training_data/006',  #001
    '../data/input/training_data/008',  #002
    '../data/input/training_data/009',  #003
    '../data/input/training_data/010',  #004
    '../data/input/training_data/011',  #005
    '../data/input/training_data/012',  #006
    '../data/input/training_data/013',  #007
    '../data/input/training_data/014',  #008
    '../data/input/training_data/015',  #009
    '../data/input/training_data/016',  #010
    '../data/input/training_data/017',  #011
    '../data/input/training_data/018',  #012
    '../data/input/training_data/019',  #013
    '../data/input/training_data/020',  #014
    '../data/input/training_data/021',  #015
    '../data/input/training_data/022'   #016
]




dataframes = pipObj.create_large_dataframe_from_multiple_input_directories(
    train_list_with_subjects,
    merge_column=None,
    save=False,
    added_columns_name=['labels'],
    list=True,
    downsample_config={
            'out_path' : '../data/temp/merged/resampled_test.csv',
            'discrete_columns_list' : ['label'],
            'source_hz': 100,
            'target_hz': 50,
            'window_size': 20000,
            'add_timestamps': True
        }
)



_, run_history = pipObj.train_lstm_model(
    training_dataframe=dataframes,
    back_cols=['bx','by','bz'],
    thigh_cols=['tx','ty','tz'],
    config_path='../params/config.yml',
    label_col='label',
    save_to_path="trained_models/both_sensors_" + str(now.day) + "_" + str(now.month),
    save_weights=True,
    shuffle=False
)

print("---------------------------------------------")

# Plot each leave one out validation pass, history;
# 2 rows 1 column

# num rows * num cols >= len(train_list_with_subject) 5 * 3 = 15 >= 15
num_rows, num_cols = 4, 4
pipObj.plot_run_history(run_history, num_rows, num_cols, train_list_with_subjects, img_title="LOO_BOTH_ADAMAX_HISTORY.png")

pipObj.calculate_avg_prec_recall_f1(run_history, add_to_history=True)

# write RUN_HISTORY to JSON FILE
pipObj.save_run_history_to_file(run_history, "LOO_BOTH_ADAMAX_HISTORY.json")