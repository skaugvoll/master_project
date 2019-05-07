import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataHandler import DataHandler
from src.config import Config
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from src import models

import numpy as np
import pandas as pd


###########
# DATA IMPORT
###########


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
    list=True,
    downsample_config={
            'out_path' : '../data/temp/merged/resampled_test.csv',
            'discrete_columns_list' : ['label'],
            'source_hz': 100,
            'target_hz': 50,
            'window_size': 20000,
            'add_timestamps': True
        }
)

# print(type(trainDataframes))
# print()
# print(type(trainDataframes[0]))



########
# MODLE CONFIGURATION
########
config_path = "../params/config.yml"

config = Config.from_yaml(config_path, override_variables={})
# config.pretty_print()

model_name = config.MODEL['name']
model_args = dict(config.MODEL['args'].items(), **config.INFERENCE.get('extra_model_args', {}))
model = models.get(model_name, model_args)

batch_size = config.TRAINING['args']['batch_size']
sequence_length = config.TRAINING['args']['sequence_length']
callbacks = config.TRAINING['args']['callbacks'] or None

back_cols = ['bx','by','bz']
thigh_cols=['tx','ty','tz']
label_col='label'

#######
# LEAVE ONE OUT CONFIGURATION
#######

X = np.array(train_list_with_subjects)

loo = LeaveOneOut()
datahandler = DataHandler()


RUNS_HISTORY = {}

for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    trainingset = []
    testset = trainDataframes[test_index[0]]
    for idx in train_index:
        trainingset.append(trainDataframes[ idx ])

    model_history = model.train(
        train_data=trainingset,
        valid_data=None,
        epochs=config.TRAINING['args']['epochs'],
        batch_size=batch_size,  # gets this from config file when init model
        sequence_length=sequence_length,  # gets this from config file when init model
        back_cols=back_cols,
        thigh_cols=thigh_cols,
        label_col=label_col,
        shuffle=False,
        # callbacks=callbacks
    )

    preds,gt,  cm = model.predict(
        dataframes=[testset],
        batch_size=batch_size,
        sequence_length=sequence_length,
        back_cols=back_cols,
        thigh_cols=thigh_cols,
        label_col=label_col)


    gt = gt.argmax(axis=1)
    preds = preds.argmax(axis=1)

    precision, recall, fscore, support = precision_recall_fscore_support(gt, preds)
    print("P \n", precision)
    print("R \n",recall)
    print("F \n",fscore)
    print("S \n",support)
    print()

    labels = []
    label_values = list(set(gt)) + list(set(preds))
    label_values = list(set(label_values))
    label_values.sort()

    for i in label_values:
        # print("I: ", i)
        shift_up_from_OHE_downshift = i + 1
        labels.append(model.encoder.name_lookup[shift_up_from_OHE_downshift])

    # print(labels)
    # input("...")

    report = classification_report(gt, preds, target_names=labels, output_dict=True)

    acc = accuracy_score(gt, preds)
    report['Accuracy'] = acc
    RUNS_HISTORY[ train_list_with_subjects[test_index[0]].split("/")[-1] ] = report






import pprint
pprint.pprint(RUNS_HISTORY)





avg_acc = 0
for key in RUNS_HISTORY:
    avg_acc += RUNS_HISTORY[key]['Accuracy']

avg_acc /= len(RUNS_HISTORY.keys())
print("AVG ACCURACY : ", avg_acc)

