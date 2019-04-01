import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler


pipObj = Pipeline()

list_with_subjects = [
    '../data/input/training_data/'
]

# os.system("rm -rf {}".format("trained_models/"))

trainDataframe = pipObj.create_large_dataframe_from_multiple_training_directories(
    list_with_subjects,
    save=False,
    added_columns_name=['labels']
)

train, validation = DataHandler.split_df_into_training_and_test(trainDataframe, split_rate=.2, shuffle=False)
validation, test = DataHandler.split_df_into_training_and_test(validation, split_rate=.5, shuffle=False)

####
# Train the model
####

#Train one sensor: remove back or thigh col and change config???
# TODO: GJØR CONCAT GENERELL OG LAG TO # SCRIPTS Som TRENER HVER SIN LSTM MODELL


pipObj.train_lstm_model(
    training_dataframe=train,
    back_cols=['bx','by','bz'],
    thigh_cols=None,
    # thigh_cols=['tx','ty','tz'],
    # config_path='../params/config.yml',
    config_path='../params/one_sensor_config.yml',
    label_col='label',
    validation_dataframe=validation,
    # save_to_path="trained_models/both_sensors_11_03",
    save_to_path="trained_models/back_sensors_11_03",
    save_weights=True,
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


# unzipped_paths += unzipped_test_paths
# paths = [ "/".join(p.split("/")[:-1]) for p in unzipped_paths]
# pipObj.remove_files_or_dirs_from(paths)