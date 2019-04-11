import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src import models

pipObj = Pipeline()

# Create a TRAINING dataframe

train = [
    '../data/temp/shower_atle.7z/shower_atle',
    '../data/temp/nonshower_paul.7z/nonshower_paul',
    # '../data/temp/Thomas.7z/Thomas',
    '../data/temp/Thomas2.7z/Thomas2',
    '../data/temp/Sigve.7z/Sigve'
]

trainDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    train,
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

# extract the features
back, thigh, labels = pipObj.get_features_and_labels_as_np_array(
    df=trainDataframe,
    back_columns=[0, 1, 2],
    thigh_columns=[3, 4, 5],
    label_column=[8]
)

btemp, ttemp, _ = pipObj.get_features_and_labels_as_np_array(
    df=trainDataframe,
    back_columns=[6],
    thigh_columns=[7],
    label_column=None
)



# Create a TEST dataframe
test = [
    '../data/temp/Thomas.7z/Thomas',
    # '../data/temp/Sigve.7z/Sigve'
    # '../data/temp/nonshower_paul.7z/nonshower_paul',
]

testDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    test,
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

# extract the features
testback, testthigh, testlabels = pipObj.get_features_and_labels_as_np_array(
    df=testDataframe,
    back_columns=[0, 1, 2],
    thigh_columns=[3, 4, 5],
    label_column=[8]
)

testbtemp, testttemp, _ = pipObj.get_features_and_labels_as_np_array(
    df=testDataframe,
    back_columns=[6],
    thigh_columns=[7],
    label_column=None
)

evaluation = [testback, testthigh, testbtemp, testttemp, testlabels]

####
# Train the model
####
# Get the model
XGB = models.get("XGB", {})
XGB.train(back, thigh, btemp, ttemp, labels, evaluation=evaluation)


