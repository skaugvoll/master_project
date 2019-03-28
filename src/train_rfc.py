import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src import models


pipObj = Pipeline()


list_with_subjects = [
    '../data/input/shower_atle.7z',
    # '../data/input/nonshower_paul.7z',
    # '../data/input/Thomas.7z',
    # '../data/input/Thomas2.7z',  # mangler labels fil
]


# unzip all data
unzipped_paths = pipObj.unzip_multiple_directories(list_with_subjects, zip_to="../data/temp/")
print(unzipped_paths)


trainDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    unzipped_paths,
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

# # Do some magic numbering
sampling_frequency = 50
window_length = 120
tempearture_reading_rate = 120
samples_pr_second = 1/(tempearture_reading_rate/sampling_frequency)
samples_pr_window = int(window_length*samples_pr_second)
train_overlap = .8
number_of_trees_in_forest = 100


# extract the features
back, thigh, labels = pipObj.get_features_and_labels_as_np_array(trainDataframe)


# Get the model
RFC = models.get("RFC", {})

####
# Train the model
####
RFC.train(
    back_training_feat=back,
    thigh_training_feat=thigh,
    labels=labels,
    samples_pr_window=samples_pr_window,
    train_overlap=train_overlap,
    number_of_trees=number_of_trees_in_forest
)


#####
# TEST THE MODEL
####

unzipped_test_paths = pipObj.unzip_multiple_directories(['../data/input/nonshower_paul.7z'], zip_to="../data/temp/")
testDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    list_with_subjects=unzipped_test_paths,
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

back, thigh, labels = pipObj.get_features_and_labels(testDataframe)

RFC.test(back, thigh, labels, samples_pr_window, train_overlap)

acc = RFC.calculate_accuracy()
print("ACC: ", acc)

unzipped_paths += unzipped_test_paths
paths = [ "/".join(p.split("/")[:-1]) for p in unzipped_paths]
# pipObj.remove_files_or_dirs_from(paths)