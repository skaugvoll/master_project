import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from pipeline.Pipeline import Pipeline
from pipeline.DataHandler import DataHandler
from src import models
import pickle




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
    model_path = "./trained_rfc.sav"
    print("MAX CPU CORES: ", os.cpu_count())
    p = Pipeline()
    p.parallel_pipeline_classification_run(
        dataframe=dataframe_test,
        model_path=model_path,
        samples_pr_window=samples_pr_window,
        train_overlap=0.8,
        seq_lenght=250,
        num_proc_mod=1,
        num_proc_clas=1
    )




