
import os, sys
import axivity
import pandas as pd
from utils import zip_utils
from utils import csv_loader


class DataHandler():
    def __init__(self):
        self.data_input_folder = os.getcwd() + '../../data/input'
        self.data_output_folder = os.getcwd() + '../../data/output'

    def _get_csv_file(self, args):
        '''
        Returns a context for getting csv file, either by a directly specified one
        or through unzipping a provided zip file and synching it with axivity's software
        '''

        # If a precomputed, timesynched file is available -> use it
        try:
            if not os.path.exists(args.file):
                raise RuntimeError('Provided presynched file "%s" does not exist' % args.file)
            job_name = args.name or os.path.splitext(os.path.basename(args.file))[0]
            return job_name, args.file

        except Exception:
            print("Could not get the csv_file")

    def _get_cwa_files(self, filepath='filepath', temp_dir='working_dir'):
        print(">>>>>>>>: ", 3, filepath, temp_dir)
        try:
            # Make sure zipfile exists
            if not os.path.exists(filepath):
                print(">>>>>>>>: ", 3.1)
                raise RuntimeError('Provided zip file "%s" does not exist' % filepath)
                # Make sure that a working directory for unzipping and time synching also exists
            if not temp_dir:
                print(">>>>>>>>: ", 3.2)
                raise RuntimeError('A working directory ("-w <directoy name.") must be specified when using --zip-file')
            if not os.path.exists(temp_dir):
                print(">>>>>>>>: ", 3.3)
                raise RuntimeError('Provided working directory "%s" does not exist' % temp_dir)

            print(">>>>>>>>: ", 4)

            # Unzip contents of zipfile
            job_name = filepath.split('/')[-1].split('.')[0]
            unzip_to_path = os.path.join(temp_dir, os.path.basename(filepath))
            print(">>>>>>>>: ", job_name, unzip_to_path)

            # with zip_utils.zip_to_working_dir(filepath, unzip_to_path) as subject_dir:
            subject_dir = zip_utils.zip_to_working_dir(filepath, unzip_to_path)
                # Apply omconvert and timesync to join thigh & back .cwa files into a single .csv
                # with axivity.timesynched_csv(subject_dir) as synched_csv:
            synched_csv = axivity.timesynched_csv(subject_dir)
            return job_name, synched_csv
        except Exception as e:
            print("could not unzipp 7z arhcive and synch it", e)


    def load_dataframe_from_7z(self, input_arhcive_path, whole_days=False, chunk_size=20000, max_days=6):
        current_directory = os.getcwd()


        # Create output directory if it does not exist
        if not os.path.exists(self.data_output_folder + '/test'):
            os.makedirs(self.data_output_folder + '/test' )

        print(">>>>>>>>: ", 1)

        name, synched_csv = self._get_cwa_files(filepath=input_arhcive_path, temp_dir=current_directory)

        print(">>>>>>>>: ", 2, name, synched_csv)
        # Return if no csv file was found
        if synched_csv is None:
            return print('Got synched csv file:', synched_csv)

        print(">>>>>>>>: ", 5)

        # Read csv files in chunks
        columns = ['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
        if whole_days:
            # Use custom loader that scans to first midnight if --whole-days is enabled
            print(">>>>>>>>: ", 6)
            self.dataframe_iterator = csv_loader.csv_chunker(synched_csv, chunk_size, ts_index=0,
                                                        columns=columns, n_days=max_days)
        else:
            print(">>>>>>>>: ", 7)
            # Otherwise, just load with pandas
            self.dataframe_iterator = pd.read_csv(synched_csv, header=None, chunksize=chunk_size,
                                             names=columns, parse_dates=[0])


    def get_dataframe_iterator(self):
        return self.dataframe_iterator

if __name__ == '__main__':
    pass


