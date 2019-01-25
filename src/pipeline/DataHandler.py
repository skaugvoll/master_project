
import os
import axivity
import pandas as pd
from utils import zip_utils
from utils import csv_loader


class DataHandler():
    def __init__(self):
        self.data_input_folder = os.getcwd() + '../../data/input'
        self.data_output_folder = os.getcwd() + '../../data/output'
        self.data_temp_folder = os.getcwd() + '../../data/temp'
        self.data_cleanup_path = None
        self.data_synched_csv_path = None
        self.job_name = None

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
                # print(">>>>>>>>: ", 3.1)
                raise RuntimeError('Provided zip file "%s" does not exist' % filepath)
                # Make sure that a working directory for unzipping and time synching also exists
            if not temp_dir:
                # print(">>>>>>>>: ", 3.2)
                raise RuntimeError('A working directory ("-w <directoy name.") must be specified when using --zip-file')
            if not os.path.exists(temp_dir):
                # print(">>>>>>>>: ", 3.3)
                raise RuntimeError('Provided working directory "%s" does not exist' % temp_dir)

            # print(">>>>>>>>: ", 4)

            # Unzip contents of zipfile
            self.name = filepath.split('/')[-1].split('.')[0]
            unzip_to_path = os.path.join(temp_dir, os.path.basename(filepath))
            self.data_cleanup_path = unzip_to_path # store the path to the unzipped folder for easy cleanup

            unzipped_dir = zip_utils.unzip_subject_data(
                subject_zip_path=filepath,
                unzip_to_path=unzip_to_path,
                return_inner_dir=True
            )

            # print(">>>>>>: UNZIPPED_DIR: ", unzipped_dir)

            self.data_synched_csv_path = axivity.convert_cwas_to_csv(
                unzipped_dir,
                out_dir=None
            )


        except Exception as e:
            print("could not unzipp 7z arhcive and synch it", e)


    def load_dataframe_from_7z(self, input_arhcive_path, whole_days=False, chunk_size=20000, max_days=6):
        current_directory = os.getcwd()

        # print(">>>>>>>>: ", 1)

        # Unzipp and synch
        self._get_cwa_files(filepath=input_arhcive_path, temp_dir=self.data_temp_folder)
        # print(">>>>>>>>>>>: ", synched_csv)

        # Create output directory if it does not exist
        self.data_output_folder = os.path.join(self.data_output_folder, self.name)
        if not os.path.exists(self.data_output_folder):
            os.makedirs(self.data_output_folder)


        # print(">>>>>>>>: ", 2)
        # Return if no csv file was found
        if self.data_synched_csv_path is None:
            raise Exception("Synched_csv is none")

        print('Got synched csv file:', self.data_synched_csv_path)

        # print(">>>>>>>>: ", 5)

        # Read csv files in chunks
        columns = ['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
        if whole_days:
            # Use custom loader that scans to first midnight if --whole-days is enabled
            # print(">>>>>>>>: ", 6)
            self.dataframe_iterator = csv_loader.csv_chunker(self.data_synched_csv_path, chunk_size, ts_index=0,
                                                        columns=columns, n_days=max_days)
        else:
            # print(">>>>>>>>: ", 7)
            # Otherwise, just load with pandas
            self.dataframe_iterator = pd.read_csv(self.data_synched_csv_path, header=None, chunksize=chunk_size,
                                             names=columns, parse_dates=[0])


    def get_dataframe_iterator(self):
        return self.dataframe_iterator

    def cleanup_temp_folder(self):
        print("Cleaning {}".format(self.data_cleanup_path))
        try:
            zip_utils.clean_up_working_dir(self.data_cleanup_path)
            print("Cleanup SUCCESS")
        except:
            print("Cleanup FAILED")

if __name__ == '__main__':
    pass


