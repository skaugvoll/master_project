import sys, os
from datetime import time

try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src import models
import pickle, math
import pandas as pd


# Create pipeline object
pipObj = Pipeline()
# Define how many cpus we can paralell meta classification on
cpus = os.cpu_count()


#define training data
list_with_subjects_to_classify = [
    '../data/input/4000181.7z',
    '../data/input/4001058.7z',
    '../data/input/4002734.7z',
    '../data/input/4003601.7z',
    '../data/input/4004141.7z'
]



for path in list_with_subjects_to_classify:
    list_with_subjects_to_classify = [path]
    subjectID = path.split("/")[-1].split(".")[0]

    # Unzipp all the data
    unzipped_paths = pipObj.unzip_multiple_directories(list_with_subjects_to_classify, zip_to="../data/temp/")
    # print(unzipped_paths)

    dataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
        unzipped_paths,
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
        added_columns_name=[],
        files=False,
        list=False
    )

    # Define the Meta classifier path
    RFC_PATH = './trained_models/LOO_RFC_11_5_ACC_0.991.h5'
    # Do some magic numbering for the Meta classifier, since temperature is recorded at a different speed
    sampling_frequency = 50
    window_length = 250
    samples_pr_window = 250


    # Define the Activity classifier paths
    lstm_models_path = {
        "1": {
            "config": "../params/config.yml",
            "saved_model": "trained_models/test_model_two_sensors.h5",
            "weights": "./trained_models/both_sensors_adadelta32T19F_19_5_ACC_0.939_weights.h5"
        },
        "2": {
            "config": "../params/thigh_sensor_config.yml",
            "saved_model": "trained_models/test_model_thigh_sensor.h5",
            "weights": "./trained_models/thigh_sensors_adadelta32T19F_19_5_ACC_0.938_weights.h5"
        },
        "3": {
            "config": "../params/back_sensor_config.yml",
            "saved_model": "trained_models/test_model_back_sensor.h5",
            "weights": "./trained_models/back_sensors_adadelta32T19F_19_5_ACC_0.875_weights.h5"
        }
    }


    dataframe_columns = {
            'back_features': ['bx','by','bz'],
            'thigh_features': ['tx', 'ty', 'tz'],
            'back_temp': ['btemp'],
            'thigh_temp': ['ttemp'],
            'label_column': None,
            'time': []
    }

    _, _, _, result_df = pipObj.parallel_pipeline_classification_run(
        dataframe=dataframe,
        dataframe_columns=dataframe_columns,
        rfc_model_path=RFC_PATH,
        lstm_models_paths=lstm_models_path,
        samples_pr_window=samples_pr_window,
        sampling_freq=50,
        train_overlap=0.8,
        seq_lenght=250,
        num_proc_mod=cpus,
        lstm_model_mapping={"both": '1', "thigh": '2', "back": '3', 'none': '4'},
        minimize_result=False
    )


    plotting_df = result_df.loc[:, ["timestart", "target"]]
    pipObj.plotter.plot_weekly_view(plotting_df, subjectID)
