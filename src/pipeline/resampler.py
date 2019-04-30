import os
import sys
import argparse
import contextlib
import src.utils.resamplers as resamplers
import pandas as pd
import subprocess


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


    for i, df in enumerate( dataframe_iterator ):
        # print(df.describe())
        # input("....")
        df.to_csv( file, mode='a', header=(i==0))



def copy_stream( stream ):
  '''
  Copies a stream so that it can be read by two differnt consumers
  '''
  # Leverage the unix tee-tool to copy stdout to stderr
  p = subprocess.Popen(
    ['tee', '/dev/stderr'],
    stdin=stream,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
  )
  return p.stdout, p.stderr

# parser = argparse.ArgumentParser('Read subject data in one or more filepaths and write as csv')
# parser.add_argument('-r', '--resampler',
#                     help='Which resampling method to use',
#                     required=True,
#                     choices=src.resamplers.get_resampler_names()
#                     )
# parser.add_argument('-s', '--source-rate',
#                     help='Sample rate of the input file',
#                     required=True,
#                     type=float
#                     )
# parser.add_argument('-t', '--target-rate',
#                     help='Target sample rate after resampling',
#                     required=True,
#                     type=float
#                     )
# parser.add_argument('-w', '--window-size',
#                     help='If set, resample data with a rolling window. Useful for large files',
#                     type=int,
#                     default=20000
#                     )
# parser.add_argument('-i', '--input',
#                     help='Filepath to csv data that will be resampled. Defaults to standard in.'
#                     )
# parser.add_argument('-o', '--output',
#                     help='Where to write the result. Defaults to standard out.',
#                     )
# parser.add_argument('--discrete-columns',
#                     help='Columns that will just be resampling by getting the closest value',
#                     nargs='+',
#                     )

def main(resampler, source_rate, target_rate, window_size, inputD, output, discrete_columns):
    # Default input/output to stdin/stdout
    if inputD is None:
        print('Using stdin for input')
        inputD = sys.__stdin__

    if output is None:
        print('Using stdout for output')
        output = sys.__stdout__

    elif os.path.exists(output):
        asw = ''
        acceptable_yes_answers = ['yes', 'y']
        acceptable_no_answers = ['no', 'n']
        while asw not in acceptable_yes_answers + acceptable_no_answers:
            asw = input("There already exists a resampled file. Do you want to delete it ? ")

        if asw in acceptable_yes_answers:
            os.system("rm {}".format(output))
            print("Done, file is removed")
        elif asw in acceptable_no_answers:
            print("Ok. Exiting program now.")
            sys.exit(-1)


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
    write_chunked_dataframe_to_file(output, resampled_stream)

#
# if __name__ == '__main__':
#     with contextlib.redirect_stdout(sys.stderr):
#         main("fourier", 100, 50, window_size=20000, "inputfile.csv", "outputfile.csv", "labelcolumnnavn")