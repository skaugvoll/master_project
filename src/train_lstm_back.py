import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
import datetime
now = datetime.datetime.now()

pipObj = Pipeline()

list_with_subjects = [
    '../data/input/training_data/'
]

trainDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    list_with_subjects,
    merge_column=None,
    save=False,
    added_columns_name=['labels'],
    list=True
)

train, validation = DataHandler.split_df_into_training_and_test(trainDataframe, split_rate=.2, shuffle=False)
validation, test = DataHandler.split_df_into_training_and_test(validation, split_rate=.5, shuffle=False)

####
# Train the model
####

pipObj.train_lstm_model(
    training_dataframe=train,
    back_cols=['bx','by','bz'],
    thigh_cols=None,
    config_path='../params/one_sensor_config.yml',
    label_col='label',
    validation_dataframe=validation,
    save_to_path="trained_models/back_sensor_" + str(now.day) + "_" + str(now.month),
    save_weights=False,
    shuffle=False
)

res = pipObj.evaluate_lstm_model(
    dataframe=test,
    label_col='label',
    num_sensors=None,
    model=None,
    back_cols=None,
    thigh_cols=None,
    cols=None,
    batch_size=None,
    sequence_length=None
)

print("Evaluation result: {}".format(res))