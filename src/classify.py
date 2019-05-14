import sys, os
from datetime import time

try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src import models
import pickle, math
import pandas as pd



######                              ######
#                                        #
#         CONFIGURE THE PIPELINE         #
#                                        #
######                              ######


# Create pipeline object
pipObj = Pipeline()
# Define how many cpus we can paralell meta classification on
cpus = os.cpu_count()
# cpus = 1



######                              ######
#                                        #
#           CONFIGURE THE DATA           #
#                                        #
######                              ######




# define training data
# list_with_subjects_to_classify = [
#     '../data/temp/4003601.7z/4003601/'
# ]
#
#
# # Unzipp all the data
# unzipped_paths = pipObj.unzip_multiple_directories(list_with_subjects_to_classify, zip_to="../data/temp/")
# # print(unzipped_paths)

data = [
    '../data/temp/4003601.7z/4003601/'
    # '../data/temp/Thomas3.7z/Thomas3/'
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
    added_columns_name=[],
    files=True,
    list=False
)

######                              ######
#                                        #
#      CONFIGURE THE META CLASSIFIER     #
#                                        #
######                              ######


# Define the Meta classifier path
RFC_PATH = './trained_models/LOO_RFC_11_5_ACC_0.991.h5'
# Do some magic numbering for the Meta classifier, since temperature is recorded at a different speed
sampling_frequency = 50
window_length = 250
samples_pr_window = 250


######                              ######
#                                        #
#      CONFIGURE ACTIVITY CLASSIFIER     #
#                                        #
######                              ######


# Define the Activity classifier paths
lstm_models_path = {
    "1": {
        "config": "../params/config.yml",
        "saved_model": "trained_models/test_model_two_sensors.h5",
        "weights": "trained_models/both_sensors_13_5_ACC_0.914_weights.h5"
    },
    "2": {
        "config": "../params/one_sensor_config.yml",
        "saved_model": "trained_models/test_model_thigh_sensor.h5",
        "weights": "trained_models/thigh_sensors_13_5_ACC_0.842_weights.h5"
    },
    "3": {
        "config": "../params/one_sensor_config.yml",
        "saved_model": "trained_models/test_model_back_sensor.h5",
        "weights": "trained_models/back_sensors_13_5_ACC_0.857_weights.h5"
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


######                              ######
#                                        #
#           RUN THE CLASSIFICATION       #
#                                        #
######                              ######


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
    lstm_model_mapping={"both": '1', "thigh": '2', "back": '3'},
    minimize_result=False
)

##CSV
start_time = time.time()
result_df.to_csv("./trained_models/4003601.csv")
print("--- CSV savetime: {} seconds ---".format(time.time() - start_time))

##PICKLE
start_time = time.time()
result_df.to_pickle("./trained_models/4003601.pkl")
print("--- Pickle savetime: {} seconds ---".format(time.time() - start_time))

##HDF
start_time = time.time()
result_df.to_hdf('./trained_models/4003601.h5', key='df', mode='w')
print("--- HDF savetime: {} seconds ---".format(time.time() - start_time))

##FEATHER
result_df = result_df.reset_index()
start_time = time.time()
result_df.to_feather('./trained_models/4003601.feather')
print("--- Feather savetime: {} seconds ---".format(time.time() - start_time))

print(result_df)

