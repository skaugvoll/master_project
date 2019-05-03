import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
import datetime
now = datetime.datetime.now()

pipObj = Pipeline()

train_list_with_subjects = [
    '../data/input/training_data/006',
    '../data/input/training_data/008',
    '../data/input/training_data/009',
    '../data/input/training_data/010',
    '../data/input/training_data/011',
    '../data/input/training_data/012',
    '../data/input/training_data/013',
    '../data/input/training_data/014',
    '../data/input/training_data/015',
    '../data/input/training_data/016',
    '../data/input/training_data/017',
    '../data/input/training_data/018',
    '../data/input/training_data/019',
    '../data/input/training_data/020',
    '../data/input/training_data/021'
]




trainDataframes = pipObj.create_large_dataframe_from_multiple_input_directories(
    train_list_with_subjects,
    merge_column=None,
    save=False,
    added_columns_name=['labels'],
    list=False,
    downsample_config={
            'out_path' : '../data/temp/merged/resampled_test.csv',
            'discrete_columns_list' : ['label'],
            'source_hz': 100,
            'target_hz': 50,
            'window_size': 20000,
            'add_timestamps': True
        }
)

# test_list_with_subjects = [
#     '../data/input/training_data/019',
#     '../data/input/training_data/020',
#     '../data/input/training_data/021'
# ]

# testDataframes= pipObj.create_large_dataframe_from_multiple_input_directories(
#     test_list_with_subjects,
#     merge_column=None,
#     save=False,
#     added_columns_name=['labels'],
#     list=True,
#     downsample_config={
#             'out_path' : '../data/temp/merged/resampled_test.csv',
#             'discrete_columns_list' : ['label'],
#             'source_hz': 100,
#             'target_hz': 50,
#             'window_size': 20000,
#             'add_timestamps': True
#         }
# )


####
# Train the model
####


train, validation = DataHandler.split_df_into_training_and_test(trainDataframes, split_rate=.2, shuffle=False)
validation, test = DataHandler.split_df_into_training_and_test(validation, split_rate=.5, shuffle=False)


_, History = pipObj.train_lstm_model(
    training_dataframe=train,
    back_cols=['bx','by','bz'],
    thigh_cols=['tx','ty','tz'],
    config_path='../params/config.yml',
    label_col='label',
    validation_dataframe=validation,
    save_to_path="trained_models/both_sensors_" + str(now.day) + "_" + str(now.month),
    save_weights=False,
    shuffle=False
)


from matplotlib import pyplot as plt
plt.plot(History.history['acc'], label="Trn")
plt.plot(History.history['val_acc'], label="Tst")
plt.plot(History.history['loss'], label="Ltrn")
plt.plot(History.history['val_loss'], label="Ltst")
plt.legend()
plt.savefig('Training History')


# print(History.history)
#
#
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

res, gt, cfm = pipObj.predict_lstm_model(
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

# print("CONFUSION MATRIX: \n", cfm)

#LEAVE ONE OUT X_VAL
# import numpy as np
# from sklearn.model_selection import LeaveOneOut
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
#
# loo = LeaveOneOut()
#
# for train_index, test_index in loo.split(X):
#   print("TRAIN:", train_index, "TEST:", test_index)
#   X_train, X_test = X[train_index], X[test_index]
#   print(X_train, X_test)