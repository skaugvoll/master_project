
import os
import axivity
import pandas as pd
import numpy as np
from utils import zip_utils
from utils import csv_loader



class DataHandler():
    def __init__(self):
        self.name = None
        self.dataframe_iterator = None
        self.data_synched_csv_path = None
        self.data_cleanup_path = None
        self.data_input_folder = os.getcwd() + '../../data/input'
        self.data_output_folder = os.getcwd() + '../../data/output'
        self.data_temp_folder = os.getcwd() + '/../data/temp'



    def _check_paths(self, filepath, temp_dir):
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

    def unzip_7z_archive(self, filepath, unzip_to_path='../data/temp', return_inner_dir=True, cleanup=True):
        self._check_paths(filepath, unzip_to_path)
        unzip_to_path = os.path.join(unzip_to_path, os.path.basename(filepath))
        print("UNZIP to PATH inside y7a: ", unzip_to_path)

        unzipped_dir_path = zip_utils.unzip_subject_data(
            subject_zip_path=filepath,
            unzip_to_path=unzip_to_path,
            return_inner_dir=return_inner_dir
        )

        self.data_cleanup_path = unzip_to_path  # store the path to the unzipped folder for easy cleanup
        if cleanup:
            self.cleanup_temp_folder()

        return unzipped_dir_path

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
        try:
            # # Make sure zipfile exists
            # if not os.path.exists(filepath):
            #     # print(">>>>>>>>: ", 3.1)
            #     raise RuntimeError('Provided zip file "%s" does not exist' % filepath)
            #     # Make sure that a working directory for unzipping and time synching also exists
            # if not temp_dir:
            #     # print(">>>>>>>>: ", 3.2)
            #     raise RuntimeError('A working directory ("-w <directoy name.") must be specified when using --zip-file')
            # if not os.path.exists(temp_dir):
            #     # print(">>>>>>>>: ", 3.3)
            #     raise RuntimeError('Provided working directory "%s" does not exist' % temp_dir)

            self._check_paths(filepath, temp_dir)

            # Unzip contents of zipfile
            self.name = filepath.split('/')[-1].split('.')[0]
            unzip_to_path = os.path.join(temp_dir, os.path.basename(filepath))
            self.data_cleanup_path = unzip_to_path # store the path to the unzipped folder for easy cleanup

            # unzipped_dir = zip_utils.unzip_subject_data(
            #     subject_zip_path=filepath,
            #     unzip_to_path=unzip_to_path,
            #     return_inner_dir=True
            # )
            unzipped_dir = self.unzip_7z_archive(filepath, unzip_to_path)


            self.data_synched_csv_path = axivity.convert_cwas_to_csv(
                unzipped_dir,
                out_dir=None
            )


        except Exception as e:
            print("could not unzipp 7z arhcive and synch it", e)

    def load_dataframe_from_7z(self, input_arhcive_path, whole_days=False, chunk_size=20000, max_days=6):

        # Unzipp and synch
        self._get_cwa_files(filepath=input_arhcive_path, temp_dir=self.data_temp_folder)


        # Create output directory if it does not exist
        self.data_output_folder = os.path.join(self.data_output_folder, self.name)
        if not os.path.exists(self.data_output_folder):
            os.makedirs(self.data_output_folder)

        # Return if no csv file was found
        if self.data_synched_csv_path is None:
            raise Exception("Synched_csv is none")

        print('Got synched csv file:', self.data_synched_csv_path)

        # Read csv files in chunks
        columns = ['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
        if whole_days:
            # Use custom loader that scans to first midnight if --whole-days is enabled
            self.dataframe_iterator = csv_loader.csv_chunker(self.data_synched_csv_path, chunk_size, ts_index=0,
                                                        columns=columns, n_days=max_days)
        else:
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

    def merge_csvs_on_first_time_overlap(self, master_csv_path, slave_csv_path, out_path=None, rearrange_columns_to=None):
        '''
        Master_csv is the csv that the first recording is used as starting point

        :param master_csv_path:
        :param slave_csv_path:
        :param out_path:
        :param rearrange_columns_to:
        :return: None
        '''

        print("READING MASTER CSV")
        master_df = pd.read_csv(master_csv_path)
        master_df.columns = ['time', 'bx', 'by', 'bz', 'btemp']

        print("READING SLAVE CSV")
        slave_df = pd.read_csv(slave_csv_path)
        slave_df.columns = ['time', 'tx', 'ty', 'tz', 'ttemp']

        # Merge the csvs
        print("MERGING MASTER AND SLAVE CSV")
        merged_df = master_df.merge(slave_df, on='time')

        ## Rearrange the columns
        if not rearrange_columns_to is None:
            print("REARRANGING CSV COLUMNS")
            merged_df = merged_df[rearrange_columns_to]

        if out_path is None:
            master_file_dir, master_filename_w_format = os.path.split(master_csv_path)
            out_path = os.path.join(master_file_dir, master_filename_w_format.split('.')[0] + '_TEMP_SYNCHED_BT.csv')

        else:
            out_path_dir, out_path_filename= os.path.split(out_path)
            if out_path_filename == '':
                out_path_filename = os.path.basename(master_csv_path).split('.')[0] + '_TEMP_SYNCHED_BT.csv'

            if not os.path.exists(out_path_dir):
                print('Creating output directory... ', out_path_dir)
                os.makedirs(out_path_dir)

            out_path = os.path.join(out_path_dir, out_path_filename)


        print("SAVING MERGED CSV")
        merged_df.to_csv(out_path, index=False)
        print("Saved synched and merged as csv to : ", os.path.abspath(out_path))

        self.dataframe_iterator = merged_df

        self.data_cleanup_path = os.path.abspath(out_path[:out_path.find('.7z/') + 4])
        self.data_synched_csv_path = os.path.abspath(out_path)
        self.name = os.path.basename(out_path)
        self.data_temp_folder = os.path.abspath(os.path.split(out_path)[0])

    def _adc_to_c(self, row, normalize=False):
        temperature_celsius_b = (row['btemp'] * 300 / 1024) - 50
        temperature_celsius_t = (row['btemp'] * 300 / 1024) - 50

        if normalize:
            print("NORAMLIZATION NOT IMPLEMENTED YET")
            # TODO IMPLEMENT NORMALIZATION

        row['btemp'] = temperature_celsius_b
        row['ttemp'] = temperature_celsius_t

        return row

    def convert_ADC_temp_to_C(self, dataframe=None, dataframe_path=None, normalize=False, save=False):
        df = None

        # 10
        if dataframe:
            df = dataframe
        # 01
        elif dataframe is None and not dataframe_path is None:
            try:
                df = pd.read_csv(dataframe_path)
            except Exception as e:
                print("Did not give a valid csv_path")
                raise e
        # 00
        elif dataframe_path is None and dataframe_path is None:
            print("Need to pass either dataframe or csv_path")
            raise Exception("Need to pass either dataframe or csv_path")
        # 11 save memory and performance
        elif dataframe and dataframe_path:
            df = dataframe

        print("STARTING converting adc to celcius...")
        self.dataframe_iterator = df.apply(self._adc_to_c, axis=1, raw=False, normalize=normalize)

        print(self.dataframe_iterator.describe(), "\n")
        print ()
        print(self.dataframe_iterator.dtypes)
        print()
        print("DONE, here is a sneak peak:\n", self.dataframe_iterator.head(5))

        if (dataframe_path or self.data_synched_csv_path) and save:
            path = dataframe_path or self.data_synched_csv_path
            self.dataframe_iterator.to_csv(path, index=False)

    def convert_column_from_str_to_datetime_test(self, dataframe, column_name="time"):
        if isinstance(dataframe, str):
            self.dataframe_iterator = pd.read_csv(dataframe)
            print(self.dataframe_iterator.head(5))
            print()
            self.dataframe_iterator.columns = ['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz', 'btemp', 'ttemp']

        self.dataframe_iterator[column_name] = pd.to_datetime(self.dataframe_iterator[column_name])
        print(self.dataframe_iterator.dtypes)

    def convert_column_from_str_to_datetime(self, column_name="time"):
        self.dataframe_iterator[column_name] = pd.to_datetime(self.dataframe_iterator[column_name])
        print(self.dataframe_iterator.dtypes)

    def set_column_as_index(self, column_name):
        self.dataframe_iterator.set_index(column_name, inplace=True)
        print("The dataframe index is now: ", self.dataframe_iterator.index.name)


    def add_new_column(self, name='label', default_value=np.nan):
        self.dataframe_iterator.insert(len(self.dataframe_iterator.columns), name, value=default_value)
        print(self.dataframe_iterator.describe())


    def add_labels_file_based_on_intervals(self, intervals={}, label_mapping={}):
        '''
        intervals = {
            'Label' : [
                        date:YYYY-MM-DD
                        start: HH:MM:SS
                        stop: HH:MM:SS
                    ]
        }

        :param dataframe:
        :param intervals:
        :return:
        '''

        if not intervals:
            print("Faak off")

        for label in intervals:
            print("label", label)
            for interval in intervals[label]:
                print("INTERVAL", interval)
                date = interval[0]
                start = interval[1]
                end = interval[2]

                start_string = '{} {}'.format(date, start)
                end_string = '{} {}'.format(date, end)
                # get indexes to add label to
                self.dataframe_iterator.loc[start_string:end_string, 'label'] = label

        print(self.dataframe_iterator)







if __name__ == '__main__':
    pass


