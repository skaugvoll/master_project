import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src import models


pipObj = Pipeline()


# list_with_subjects = [
#     '../data/input/shower_atle.7z',
#     '../data/input/nonshower_paul.7z',
#     '../data/input/Thomas.7z',
#     '../data/input/Thomas2.7z',
#     '../data/input/Sigve.7z'
# ]

# unzip all data
# unzipped_paths = pipObj.unzip_multiple_directories(list_with_subjects, zip_to="../data/temp/")
# print(unzipped_paths)


train = ['../data/temp/shower_atle.7z/shower_atle',
    '../data/temp/nonshower_paul.7z/nonshower_paul',
    '../data/temp/Thomas.7z/Thomas',
    '../data/temp/Thomas2.7z/Thomas2',
    # '../data/temp/Sigve.7z/Sigve'
    ]

test = [
    # '../data/temp/Thomas.7z/Thomas',
    '../data/temp/Sigve.7z/Sigve'
    # '../data/temp/nonshower_paul.7z/nonshower_paul',
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
#
# # Do some magic numbering since the temperature is recorded at a different speed then accelerometer
# sampling_frequency = 50
# window_length = 250
# tempearture_reading_rate = 120
# samples_pr_second = 1/(tempearture_reading_rate/sampling_frequency)
# samples_pr_window = int(window_length*samples_pr_second)
# train_overlap = .8
# number_of_trees_in_forest = 200


# extract the features
back, thigh, labels = pipObj.get_features_and_labels_as_np_array(
    df=trainDataframe,
    back_columns=[0,1,2],
    thigh_columns=[3,4,5],
    label_column=[8]
)

btemp, ttemp, _ = pipObj.get_features_and_labels_as_np_array(
    df=trainDataframe,
    back_columns=[6],
    thigh_columns=[7],
    label_column=None
)

####
# Train the model
####
# Get the model
RFC = models.get("RFC", {})

RFC = pipObj.train_rfc_model(back,thigh,btemp,ttemp,labels)


#####
# TEST THE MODEL
####
#
# unzipped_test_paths = pipObj.unzip_multiple_directories(test, zip_to="../data/temp/")
testDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    list_with_subjects=test,
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

back, thigh, labels = pipObj.get_features_and_labels_as_np_array(
    df=testDataframe,
    back_columns=[0,1,2],
    thigh_columns=[3,4,5],
    label_column=[8]
)

btemp, ttemp, _ = pipObj.get_features_and_labels_as_np_array(
    df=testDataframe,
    back_columns=[6],
    thigh_columns=[7],
    label_column=None
)


acc = pipObj.evaluate_rfc_model(back, thigh, btemp, ttemp, labels)

# RFC.test(back, thigh,[btemp, ttemp], labels, samples_pr_window, train_overlap)
#
# acc = RFC.calculate_accuracy()
print("ACC: ", acc)

pipObj.save_model(RFC, "./trained_jaeveligBra_rfc.save")


# unzipped_paths += unzipped_test_paths
# paths = [ "/".join(p.split("/")[:-1]) for p in unzipped_paths]
# pipObj.remove_files_or_dirs_from(paths)