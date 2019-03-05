import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from pipeline.PipeLineRFCWindow import Pipeline
from pipeline.DataHandler import DataHandler
from src import models
import pickle, math




# #########################################################
# ##
# # TESTING
# ##
# #########################################################
#
# GET DATA
dh3 = DataHandler()
dh3.load_dataframe_from_csv(
    input_directory_path='/app/data/temp/4000181.7z/4000181/',
    filename='4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv',
    header=0,
    columns=['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'btemp', 'ttemp']
)

dh3.convert_column_from_str_to_datetime_test(column_name='timestamp')
dh3.set_column_as_index("timestamp")

dh3.add_new_column()
dh3.add_labels_file_based_on_intervals(
    intervals={
        '1': [
            [
                '2017-09-19',
                '18:31:09',
                '23:59:59'
            ],
            [
                '2017-09-20',
                '00:00:00',
                '08:23:08'
            ],
            [
                '2017-09-20',
                '08:35:13',
                '16:03:58'
            ],
            [
                '2017-09-20',
                '16:20:21',
                '23:59:59'
            ],
            [
                '2017-09-21',
                '00:00:00',
                '09:23:07'
            ],
            [
                '2017-09-21',
                '09:35:40',
                '23:59:59'
            ],
            [
                '2017-09-22',
                '00:00:00',
                '09:54:29'
            ]
	    ],
        '3': [
            [
                '2017-09-20',
                '08:23:09',
                '08:35:12'
            ],
            [
                '2017-09-20',
                '16:03:59',
                '16:20:20'
            ],
            [
                '2017-09-21',
                '09:23:08',
                '09:35:39'
            ]
        ]
    }
)

dataframe_test = dh3.get_dataframe_iterator()
dataframe_test.dropna(subset=['label'], inplace=True)




###############
# RUN PIPELINE PARALLELL CODE building queues for model classification and activity classification
###############


# Do some magic numbering
sampling_frequency = 50
window_length = 120
tempearture_reading_rate = 120
samples_pr_second = 1/(tempearture_reading_rate/sampling_frequency)
samples_pr_window = int(window_length*samples_pr_second)


RFC = models.get("RFC", {})

#### save the model so that the parallelized code can lode it for each new process
# s = input("save the trained RFC model? [y/n]: ")
s = 'y'
if s == 'y':
    # TODO: fix where the file is saved
    rfc_model_path = "./trained_rfc.sav"
    # has to start on one, and be incremental
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


    model_cpus = math.floor(os.cpu_count() // 2)
    class_cpus = math.floor(os.cpu_count() // 2)
    if model_cpus == 0 or class_cpus == 0:
        model_cpus, class_cpus = 1, 1


    p = Pipeline()
    p.parallel_pipeline_classification_run(
        dataframe=dataframe_test,
        rfc_model_path=rfc_model_path,
        lstm_models_paths=lstm_models_path,
        samples_pr_window=samples_pr_window,
        train_overlap=0.8,
        seq_lenght=250,
        # num_proc_mod=model_cpus,
        # num_proc_clas=class_cpus
        num_proc_mod = model_cpus
        # num_proc_clas = 3
    )




