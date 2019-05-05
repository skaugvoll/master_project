import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")


import datetime
import numpy as np
from matplotlib import pyplot as plt
from src.pipeline.DataHandler import DataHandler
from src.pipeline.Pipeline import Pipeline

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




dataframes = pipObj.create_large_dataframe_from_multiple_input_directories(
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



_, run_history = pipObj.train_lstm_model(
    training_dataframe=dataframes,
    back_cols=['bx','by','bz'],
    thigh_cols=['tx','ty','tz'],
    config_path='../params/config.yml',
    label_col='label',
    save_to_path="trained_models/both_sensors_" + str(now.day) + "_" + str(now.month),
    save_weights=False,
    shuffle=False
)

print("---------------------------------------------")

# Plot each leave one out validation pass, history;
# 2 rows 1 column

num_rows, num_cols = 5, 3
row_height, col_height = 10, 10
figsize= (num_rows * row_height, num_cols * col_height)
fig, axis = pipObj.plotter.start_multiple_plots(num_rows, num_cols, figsize=figsize)

row = 0
col= 0

for k in run_history:
    # print("K: ", k)
    if k == 'AVG_ACCURACY':
        continue

    run = run_history[k]
    try:
        labels = np.array(run['Labels'])
        y_true = np.array(run['Ground_truth'])
        y_pred = np.array(run['Predictions'])

        # if num_cols >= 2, use col index when get axis at row column, else column = None and use row as index
        ax = pipObj.plotter.get_axis_at_row_column(row, col)
        ax.set_yscale('linear')
        ax.set_title('linear')
        ax.grid(True)

        # if no more columns, and there is a new row
        if col + 1 >= num_cols and row + 1 < num_rows:
            row += 1


        # write out the row
        if col + 1 < num_cols:
            col += 1
        else:
            col = 0




        ds = train_list_with_subjects[k - 1]
        title = str(ds).split("/")[-1] + " :: AVG ACC: " + str(run_history[k]['Accuracy'])
        pipObj.plot_confusion_matrix(y_true, y_pred, labels, figure=fig, axis=ax, title=title)
    except Exception as e:
        print("Woopsises; ", e)
        continue
    finally:
        pass
        # input("....")

# pipObj.plotter.plotter_show()
pipObj.plotter.plotter_save()