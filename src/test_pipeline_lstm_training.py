import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: pass


import pandas as pd

from pipeline.DataHandler import DataHandler
from pipeline.Pipeline import Pipeline



p = Pipeline()

list_with_subjects = [
    '../data/input/006',
    '../data/input/008',
    '../data/input/009'
]

dataframe = p.create_large_dafatframe_from_multiple_input_directories(
    list_with_subjects,
    back_keywords=['Back', "b"],
    thigh_keywords=['Thigh', "t"],
    label_keywords=['GoPro', "Labels"],
    out_path=None,
    merge_column=None,
    master_columns=['bx', 'by', 'bz'],
    slave_columns=['tx', 'ty', 'tz'],
    rearrange_columns_to=None,
    save=False,
    added_columns_name=["label"]
)


train, validation = DataHandler.split_df_into_training_and_test(dataframe, split_rate=.8)
validation, test = DataHandler.split_df_into_training_and_test(validation, split_rate=.8)

p.train_lstm_model(
    training_dataframe=train,
    back_cols=['bx','by','bz'],
    # back_cols=None,
    thigh_cols=['tx','ty','tz'],
    # thigh_cols=None,
    config_path='../params/config.yml',
    # config_path='../params/one_sensor_config.yml',
    label_col='label',
    validation_dataframe=validation,
    save_to_path="trained_models/test_le_both",
    save_weights=True
)


p.evaluate_lstm_model(
    test,
    label_col='label',
    num_sensors=None,
    model=None,
    back_cols=None,
    thigh_cols=None,
    cols=None,
    batch_size=None,
    sequence_length=None
)