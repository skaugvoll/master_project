import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: pass


import pandas as pd

from pipeline.DataHandler import DataHandler
from pipeline.Pipeline import  Pipeline

# full_path = '/Users/sigveskaugvoll/Documents/Master/data/temp/4000181.7z/4000181/4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv'
#
# datahandler = DataHandler()
#
# datahandler.load_dataframe_from_csv('../data/temp/4000181.7z/4000181/',
#                                 '4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv',
#                                 header=0,
#                                 columns=['timestamp', 'bx', 'by', 'bz','tx','ty','tz','btemp','ttemp'],
#                                 whole_days=False,
#                                 chunk_size=20000,
#                                 max_days=6)
#
# # print(datahandler.get_dataframe_iterator().describe)
#
# # print(datahandler.head_dataframe())
#
# datahandler.convert_column_from_str_to_datetime_test(column_name='timestamp')
# datahandler.set_column_as_index('timestamp')
# # print(datahandler.head_dataframe())
#
# # ADD THE LABELS
# datahandler.add_new_column()
# datahandler.add_labels_file_based_on_intervals(
#     intervals={
#         '1': [
#             [
#                 '2017-09-19',
#                 '18:31:09',
#                 '23:59:59'
#             ],
#             [
#                 '2017-09-20',
#                 '00:00:00',
#                 '08:23:08'
#             ],
#             [
#                 '2017-09-20',
#                 '08:35:13',
#                 '16:03:58'
#             ],
#             [
#                 '2017-09-20',
#                 '16:20:21',
#                 '23:59:59'
#             ],
#             [
#                 '2017-09-21',
#                 '00:00:00',
#                 '09:23:07'
#             ],
#             [
#                 '2017-09-21',
#                 '09:35:40',
#                 '23:59:59'
#             ],
#             [
#                 '2017-09-22',
#                 '00:00:00',
#                 '09:54:29'
#             ]
# 	    ],
#         '3': [
#             [
#                 '2017-09-20',
#                 '08:23:09',
#                 '08:35:12'
#             ],
#             [
#                 '2017-09-20',
#                 '16:03:59',
#                 '16:20:20'
#             ],
#             [
#                 '2017-09-21',
#                 '09:23:08',
#                 '09:35:39'
#             ]
#         ]
#     }
# )
#
# dataframe = datahandler.get_dataframe_iterator()
# print("DESCRIBE0 : \n", dataframe.describe())
# dataframe.dropna(subset=['label'], inplace=True)
# print("DTYPES0 : \n", dataframe.dtypes)
# dataframe['label'] = pd.to_numeric(dataframe['label']) #, downcast='integer')
# print("DTYPES1 : \n", dataframe.dtypes)


# from src import models

# lstm = models.get('LSTMTEST', {})
# #  lstm.train(dataframe, epochs=10, batch_size=512, sequence_lenght=250, split=0.8)
# lstm.trainBatch(dataframe, epochs=10, batch_size=512, sequence_lenght=250, split=0.8)








####
# Create the dataset
####

p = Pipeline()

list_with_subjects = [
            '../data/input/006',
            '../data/input/008'
        ]

dataframe = p.create_large_dafatframe_from_multiple_input_directories(
    list_with_subjects,
    back_keywords=['Back'],
    thigh_keywords = ['Thigh'],
    label_keywords = ['GoPro', "Labels"],
    out_path=None,
    merge_column = None,
    master_columns = ['bx', 'by', 'bz'],
    slave_columns = ['tx', 'ty', 'tz'],
    rearrange_columns_to = None,
    save=False,
    added_columns_name=["label"]
)

# randomize aka shuffle the dataframe
# The frac keyword argument specifies the fraction of rows to return in the random sample,
# so frac=1 means return all rows (in random order).

# Burde kansje ikke shuffle dataframen her, da vi blander subject sensor readings.
# TODO: flytte shuffle til get features i selve modelen
# print("SHUFFLEROO")
# dataframe = dataframe.sample(frac=1)


how_much_is_training_data = 0.8
training_data = int(dataframe.shape[0] * how_much_is_training_data)
training_dataframe = dataframe.iloc[: training_data]
validation_dataframe = dataframe.iloc[training_data:] # 20 % of data



print('Training data: {}\nValidation data: {}'.format(training_dataframe.shape, validation_dataframe.shape))


from src import models
from src.config import Config

model_arguments = None

model_dir = '../data/outout/' # where the model should be stored

input_dir =  '../data/input/006/' # Input, relevant for training


WEIGHTS_PATH = '../params/two_sensor_weights.h5' # Where trained weights should be stored

DATASET_PATH = '' # Where the training dataset can be found


# config = Config.from_yaml( '../params/one_sensor_config.yml', override_variables={
config = Config.from_yaml( '../params/config.yml', override_variables={

    'MODEL_DIR': model_dir,
    'INPUT_DIR': input_dir,
    'WEIGHTS_PATH': WEIGHTS_PATH
})

# config.pretty_print()


model_name = config.MODEL['name']
model_args = dict( config.MODEL['args'].items(), **config.INFERENCE.get( 'extra_model_args', {} ))

model = models.get(model_name, model_args)

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

########################### single_sensor_lstm
# model.train(
#     train_data=[training_dataframe],
#     cols=['bx', 'by', 'bz'],
#     label_col='label',
#     valid_data=[validation_dataframe]
# )
#############


########################### dual_sensor_lstm
model.train(
    train_data=[training_dataframe],
    valid_data=[validation_dataframe],
    back_cols=['bx', 'by', 'bz'],
    thigh_cols=['tx', 'ty', 'tz'],
    label_col='label',
    epochs=config.TRAINING['args']['epochs']
)


#####
# Save the weights
#####


# model.model.save_weights(config.WEIGHTS_PATH)

input("\nTRAINING DONE\n [PRESS ENTER]")

#################
# EVALUATE MODEL
#################

eval_df = p.create_large_dafatframe_from_multiple_input_directories(
    ["../data/input/009"],
    back_keywords=['B'],
    thigh_keywords = ['T'],
    label_keywords = ['GoPro', "Labels"],
    out_path=None,
    merge_column = None,
    master_columns = ['bx', 'by', 'bz'],
    slave_columns = ['tx', 'ty', 'tz'],
    rearrange_columns_to = None,
    save=False,
    added_columns_name=["label"]
)

res = model.evaluate(
    dataframes=[eval_df],
    back_cols=['bx', 'by', 'bz'],
    thigh_cols=['tx', 'ty', 'tz'],
    label_col='label',
)

print(model.model.metrics_names, "\n", res)

input("\nEVALUATE DONE\n [PRESS ENTER]")

#################
# Predict on one window
#################
print("PREDICTION ON ONE WINDOW! in this case, one row of the dataset "\
      "so the shape is now (1, 1, 3) for each input "\
      "should change to (1, seq_lenght, 3)")


res = model.predict_on_one_window(eval_df.iloc[0])
res = res[0] # there will always be an array with one element
print("RES: ", res)
print("")
indx_of_most_conf = res.argmax(axis=0)
print("CLASS", " --> ", "CONFIDENCE")
print(indx_of_most_conf, " --> ", res[indx_of_most_conf])

input("\nONE WINDOW CLASSIFICATION DONE\n [PRESS ENTER]")

#################
# CLASSIFY W/ MODEL
#################

datahandler = DataHandler()

# csv has column names as first row
datahandler.load_dataframe_from_csv('../data/temp/4000181.7z/4000181/',
                                '4000181-34566_2017-09-19_B_TEMP_SYNCHED_BT.csv',
                                whole_days=True,
                                chunk_size=20000,
                                max_days=6)


#cols =  time,bx,by,bz,tx,ty,tz,btemp,ttemp

predictions = model.inference(
    dataframe_iterator=datahandler.get_dataframe_iterator(),
    batch_size=512,
    sequence_length=250,
    weights_path=config.WEIGHTS_PATH,
    timestamp_col="time",
    back_cols=['bx', 'by', 'bz'],
    thigh_cols=['tx', 'ty', 'tz']
)


input("\nENTIRE DATASET CLASSIFICATION DONE\n [PRESS ENTER]")

