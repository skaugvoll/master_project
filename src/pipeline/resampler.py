import os
import sys
import argparse
import contextlib
import src.utils.resamplers as resamplers
import pandas as pd
import subprocess
from src.utils.cmdline_input import cmd_input


def read_sensor_data( file, chunksize=None, index_col=0, parse_dates=[0] ):
  '''
  Convenience function for reading sensor data in chunks
  which defaults to a datetime index at column-position 0
  '''
  return pd.read_csv( file, index_col=index_col, parse_dates=parse_dates, chunksize=chunksize )


def write_chunked_dataframe_to_file( file, dataframe_iterator ):
    '''
    Write a chunked dataframe, meaning a dataframe that
    comes as an iterator of dataframe, to csv
    '''

    result_df = None
    for i, df in enumerate( dataframe_iterator ):
        if i == 0:
            result_df = df
        # print(df.describe())
        # input("....")
        df.to_csv( file, mode='a', header=(i==0))
        result_df = result_df.append(df)

    return result_df



def main(resampler, source_rate, target_rate, window_size, inputD, output, discrete_columns):
    # Default input/output to stdin/stdout
    if inputD is None:
        print('Using stdin for input')
        inputD = sys.__stdin__

    if output is None:
        print('Using stdout for output')
        output = sys.__stdout__

    elif os.path.exists(output):
        question = "There already exists a resampled file. Do you want to delete it ? "
        funcYes = lambda: os.system("rm {}".format(output))
        funcNo = lambda: sys.exit(-1)
        cmd_input(question, funcYes, funcNo, yesPrint="Done. file is removed.", noPrint="Ok. Exiting system now.")


    # Compute downsample factor
    downsample_factor = source_rate / target_rate
    print('Source rate {source_rate}, target rate {target_rate} => downsample_factor {downsample_factor}'.format(source_rate=source_rate, target_rate=target_rate, downsample_factor=downsample_factor))

    # Get resampler
    resampler_f = resamplers.get_resampler(resampler)
    print('Using resampler: {}'.format(resampler))

    discrete_columns = discrete_columns or []
    print('Using discrete_columns: ', discrete_columns, "\n")


    # Read data in chunks
    dataframe_iterator = read_sensor_data(inputD, chunksize=window_size) # pandas.io.parsers.TextFileReader

    # Apply resampler
    resampled_stream = resamplers.resample_stream(
        resampler=resampler_f,
        dataframe_iterator=dataframe_iterator,
        downsample_factor=downsample_factor,
        discrete_columns=discrete_columns
    )
    # Then write it to file
    result_df = write_chunked_dataframe_to_file(output, resampled_stream)

    return result_df

#
# if __name__ == '__main__':
#     with contextlib.redirect_stdout(sys.stderr):
#         main("fourier", 100, 50, window_size=20000, "inputfile.csv", "outputfile.csv", "labelcolumnnavn")