import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler


pipObj = Pipeline()

list_with_subjects = [
    '../data/input/006',
    '../data/input/008',
    '../data/input/009',
]



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

pipObj.train_lstm_model(
    training_dataframe=train,
    back_cols=['bx','by','bz'],
    thigh_cols=['tx','ty','tz'],
    config_path='../params/config.yml',
    # config_path='../params/one_senso_config.yml',
    label_col='label',
    validation_dataframe=validation,
    save_to_path="trained_models/both_sensors_11_03",
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


#####
# TEST THE MODEL
####

# unzipped_test_paths = pipObj.unzip_multiple_directories(['../data/input/nonshower_paul.7z'], zip_to="../data/temp/")
# testDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
#     list_with_subjects=unzipped_test_paths,
#     merge_column='time',
#     master_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz'],
#     slave_columns=['time', 'bx1', 'by1', 'bz1', 'btemp'],
#     slave2_columns=['time', 'tx1', 'ty1', 'tz1', 'ttemp'],
#     rearrange_columns_to=[
#                         'time',
#                         'bx',
#                         'by',
#                         'bz',
#                         'tx',
#                         'ty',
#                         'tz',
#                         'btemp',
#                         'ttemp'
#                     ],
#     save=False,
#     added_columns_name=['labels']
#
# )
#
# back, thigh, labels = pipObj.get_features_and_labels(testDataframe)
#
# RFC.test(back, thigh, labels, samples_pr_window, train_overlap)
#
# acc = RFC.calculate_accuracy()
# print("ACC: ", acc)
#
# unzipped_paths += unzipped_test_paths
# paths = [ "/".join(p.split("/")[:-1]) for p in unzipped_paths]
# pipObj.remove_files_or_dirs_from(paths)