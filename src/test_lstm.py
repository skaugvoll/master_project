import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: pass

import pandas as pd

from pipeline.DataHandler import DataHandler

full_path = '/Users/sigveskaugvoll/Documents/Master/data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv'

datahandler = DataHandler()

datahandler.load_dataframe_from_csv('../data/temp/4000181.7z/4000181/',
                                '4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv',
                                header=0,
                                columns=['timestamp', 'bx', 'by', 'bz','tx','ty','tz','btemp','ttemp'],
                                whole_days=False,
                                chunk_size=20000,
                                max_days=6)

# print(datahandler.get_dataframe_iterator().describe)

# print(datahandler.head_dataframe())

datahandler.convert_column_from_str_to_datetime_test(column_name='timestamp')
datahandler.set_column_as_index('timestamp')
# print(datahandler.head_dataframe())

# ADD THE LABELS
datahandler.add_new_column()
datahandler.add_labels_file_based_on_intervals(
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

dataframe = datahandler.get_dataframe_iterator()
print("DESCRIBE0 : \n", dataframe.describe())
dataframe.dropna(subset=['label'], inplace=True)
print("DTYPES0 : \n", dataframe.dtypes)
dataframe['label'] = pd.to_numeric(dataframe['label']) #, downcast='integer')
print("DTYPES1 : \n", dataframe.dtypes)
input("...")

# from src import models

# lstm = models.get('LSTMTEST', {})
# #  lstm.train(dataframe, epochs=10, batch_size=512, sequence_lenght=250, split=0.8)
# lstm.trainBatch(dataframe, epochs=10, batch_size=512, sequence_lenght=250, split=0.8)









####
# Trying to make the HaakonLSTM run
####
from src import models
from src.config import Config

model_arguments = None

model_dir = '../data/outout/' # where the model should be stored

input_dir =  '../data/temp/4000181.7z/4000181/' # Input, relevant for training


WEIGHTS_PATH = '' # Where trained weights should be stored

DATASET_PATH = '' # Where the training dataset can be found

# read in configurations from yml config file
config = Config.from_yaml( '../params/config.yml', override_variables={
    'MODEL_DIR': model_dir,
    'INPUT_DIR': input_dir
})

config.pretty_print()

model_name = config.MODEL['name']
model_args = dict( config.MODEL['args'].items(), **config.INFERENCE.get( 'extra_model_args', {} ))

print()
print(model_name)
print(model_args)
print()
for k, v in model_args.items():
    print(k, v)

print()
model = models.get(model_name, model_args)
# model.summary() # for some reason does not work

print()
# print("TRYING TO TRAIN")
'''
__init__.py states:
    Train the model. Usually, we like to split the data into training and validation
    by producing a dichotomy over the subjects. This means that 

    Inputs:
      - train_data: list<pd.DataFrame>
        A list of dataframes, intended to be used for model fitting. It's columns are:
          <back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label> 
        Where the first 6 corresponds to sensor data as floats and the last one is an 
        integer corresponding to the annotated class
      - valid_data: list<pd.DataFrame>
        Same as train_data, execpt usually a much shorter list.
      - **kwargs:
        Extra arguments found in the model's config
    

'''

print("DESCRIBE2 : \n", dataframe.describe())
dataframe.drop(columns=['btemp', 'ttemp'], inplace=True)
print(dataframe.describe())
print(dataframe.head(2))
input("...")

model.train([dataframe])
