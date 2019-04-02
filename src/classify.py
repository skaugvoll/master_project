import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src import models
import pickle, math



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
list_with_subjects_to_classify = [
    '../data/input/shower_atle.7z',
    # '../data/input/nonshower_paul.7z',
    # '../data/input/Thomas.7z',
    # '../data/input/Thomas2.7z',
    # '../data/input/Sigve.7z'
]


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
    added_columns_name=['labels']
)


######                              ######
#                                        #
#      CONFIGURE THE META CLASSIFIER     #
#                                        #
######                              ######


# Define the Meta classifier path
RFC_PATH = './trained_rfc_shower_atle.save'
# Do some magic numbering for the Meta classifier, since temperature is recorded at a different speed
sampling_frequency = 50
window_length = 250
tempearture_reading_rate = 120
samples_pr_second = 1/(tempearture_reading_rate/sampling_frequency)
# samples_pr_window = int(window_length*samples_pr_second)
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
        "weights": "trained_models/test_model_two_sensors_weights.h5"
    },
    "2": {
        "config": "../params/one_sensor_config.yml",
        "saved_model": "trained_models/test_model_thigh_sensor.h5",
        "weights": "trained_models/test_model_thigh_sensor_weights.h5"
    },
    "3": {
        "config": "../params/one_sensor_config.yml",
        "saved_model": "trained_models/test_model_back_sensor.h5",
        "weights": "trained_models/test_model_back_sensor_weights.h5"
    }
}


dataframe_columns = {
        'back_features': ['bx','by','bz'],
        'thigh_features': ['tx', 'ty', 'tz'],
        'back_temp': ['btemp'],
        'thigh_temp': ['ttemp'],
        'label_column': ['labels'],
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
    train_overlap=0.8,
    seq_lenght=250,
    num_proc_mod=cpus,
    lstm_model_mapping={"both": '1', "thigh": '2', "back": '3'}
)

print(result_df)