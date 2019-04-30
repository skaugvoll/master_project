import sys, os
try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
except: print("SAdsadsadhsa;hkldasjkd")

from src.pipeline.Pipeline import Pipeline

pipObj = Pipeline()


# 1. Fist unzipp if .7z

# 2. then concat / merge two sensors and synch with time and temperature
## for instance; pipelineObject.create_large_dataframe_from_multiple_input_directories does all of this!

# For each dataset:
    # extract_temperature using cwa_converter.convert_cwas_to_csv_with_temp
    # merge_multiple_csvs
    # concat_dataframes
    # optional add_labels
    # save to a specific output path


# outpath, res_df = pipObj.downsampleData(
#     input_csv_path="../data/temp/merged/res006.csv",
#     out_csv_path="../data/temp/merged/resampled006.csv",
#     discrete_columns=['label']
# )


trainDataframe = pipObj.create_large_dataframe_from_multiple_input_directories(
    ['../data/input/training_data/006'],
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

# ha en funksjon for a lese inn csv som dataframe, som saa blir da trining eller testing dataframe equals to return of pipObj.create_large_dataframe_from_multiple_input_directories
#dataframe =  datahandler.load_dataframe_from_csv()

