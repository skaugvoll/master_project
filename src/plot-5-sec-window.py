import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")


import datetime
import numpy as np
from matplotlib import pyplot as plt
from src.pipeline.DataHandler import DataHandler
from src.pipeline.Pipeline import Pipeline


pipObj = Pipeline()

train_list_with_subjects = [
    '../data/temp/Sigve2.7z/Sigve2/',
]


timestamps = [
    [
        [
            "2019-04-01 20:00:00",
            "2019-04-01 20:00:05"
        ],
        [
            "2019-04-01 22:00:00",
            "2019-04-01 22:00:05"
        ],
        [
            "2019-04-01 22:16:00",
            "2019-04-01 22:16:05"
        ],

    ]
]


dataframes = pipObj.create_large_dataframe_from_multiple_input_directories(
    train_list_with_subjects,
    merge_column=None,
    save=False,
    added_columns_name=['labels'],
    list=True
)


dh = DataHandler()
for idx, df in enumerate(dataframes):
        for tidx, times in enumerate(timestamps[idx]):

            start = times[0]
            end = times[1]

            # print(start, end)
            # input("...")

            res = df.loc[start:end, ['ttemp']]
            vals = res['ttemp'].values
            # print(vals, type(vals))
            # input("...")
            min = np.amin(vals) - 1
            max = np.amax(vals) + 1
            # print(" \t ")
            print(res, min, max)
            # input("...")
            styles_to_print = ["r-"]

            fig = plt.figure()
            res['ttemp'].plot(style=styles_to_print)
            try:
                plt.plot(res['ttemp'])
                plt.savefig("plot_{}_{}-{}.png".format(idx, tidx, tidx))
            except Exception as e:
                print("PLT . PLOT DID NOT WORK!!")
            # plt.ylim([min, max])
            plt.savefig("plot_{}_{}.png".format(idx, tidx))

