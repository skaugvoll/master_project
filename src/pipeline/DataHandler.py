
import os
import re
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

    def create_output_dir(self, output_dir_path, name):
        # Create output directory if it does not exist
        self.data_output_folder = os.path.join(output_dir_path, name)
        if not os.path.exists(self.data_output_folder):
            os.makedirs(self.data_output_folder)

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

    def _get_csv_file(self, filepath):
        '''
        Returns a context for getting csv file, either by a directly specified one
        or through unzipping a provided zip file and synching it with axivity's software
        '''

        # TODO : Rewrite this to support the DataHandler class, and not the old style as below
        # # If a precomputed, timesynched file is available -> use it
        try:
            if not os.path.exists(filepath):
                raise RuntimeError('Provided presynched file "%s" does not exist' % filepath)
            self.name = os.path.splitext(os.path.basename(filepath))[0]
            self.data_synched_csv_path = filepath

        except Exception:
            print("Could not get the csv_file. Check the INPUT DIRECTORY PATH and FILENAME")

    def load_dataframe_from_csv(self, input_directory_path,
                                filename,
                                header=None,
                                columns=['timestamp', 'x', 'y', 'z'],
                                whole_days=False,
                                chunk_size=20000,
                                max_days=6):


        filepath = os.path.join(input_directory_path, filename)
        self._get_csv_file(filepath)

        print("NAME:", self.name)
        print("DSCP: ", self.data_synched_csv_path)

        # Create output directory if it does not exist
        self.create_output_dir(self.data_output_folder, self.name)

        self.dataframe_iterator = pd.read_csv(self.data_synched_csv_path, header=header, names=columns )

        # # Read csv files in chunks
        # if whole_days:
        #     # Use custom loader that scans to first midnight if --whole-days is enabled
        #     self.dataframe_iterator = csv_loader.csv_chunker(self.data_synched_csv_path, chunk_size, ts_index=0,
        #                                                      columns=columns, n_days=max_days)
        #     print("DATAFRAME ITERATOR SET 1")
        # else:
        #     # Otherwise, just load with pandas
        #     self.dataframe_iterator = pd.read_csv(self.data_synched_csv_path, header=None, chunksize=chunk_size,
        #                                           names=columns, parse_dates=[0])
        #     print("DATAFRAME ITERATOR SET 2")

    def _get_cwa_files(self, filepath='filepath', temp_dir='working_dir'):
        '''
        #TODO rename to unzip, synch and return cwa files

        :param filepath:
        :param temp_dir:
        :return:
        '''
        try:
            self._check_paths(filepath, temp_dir)

            # Unzip contents of zipfile
            self.name = filepath.split('/')[-1].split('.')[0]
            unzip_to_path = os.path.join(temp_dir, os.path.basename(filepath))
            self.data_cleanup_path = unzip_to_path # store the path to the unzipped folder for easy cleanup

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
        self.create_output_dir(self.data_output_folder, self.name)

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
        temperature_celsius_t = (row['ttemp'] * 300 / 1024) - 50

        if normalize:
            print("NORAMLIZATION NOT IMPLEMENTED YET")
            # TODO IMPLEMENT NORMALIZATION

        row['btemp'] = temperature_celsius_b
        row['ttemp'] = temperature_celsius_t

        return row

    def convert_ADC_temp_to_C(self, dataframe=None, dataframe_path=None, normalize=False, save=False):
        '''
        IF passed in dataframe, sets dh objects dataframe to the converted, not inplace change on the parameter
        The check of path and dataframe should be upgradet, but works for now.
        Perhaps make the apply function be inplace

        :param dataframe:
        :param dataframe_path:
        :param normalize:
        :param save:
        :return:
        '''

        df = None

        # 10
        if not dataframe is None:
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
            # Todo this will never happen, i think because of if
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

        return self.get_dataframe_iterator()

    def convert_column_from_str_to_datetime_test(self, dataframe, column_name="time"):
        # TODO if dataframe is actually dataframe object, self.dataframe_iterator = dataframe
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

    def convert_column_from_str_to_numeric(self, column_name="ttemp"):
        self.dataframe_iterator[column_name] = pd.to_numeric(self.dataframe_iterator[column_name])
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
            print("Faak off, datahandler add_labels_file_based_on_intervals")

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

    def read_and_return_multiple_csv_iterators(self, dir_path,
                                               filenames=['back', 'thigh', "labels"],
                                               format='csv',
                                               header=None,
                                               asNumpyArray=True):
        if not filenames:
            raise Exception('Filenames for csv to read cannot be empty')

        csvs = []
        error = []
        regex_base = "[A-Za-z_\-.]*{}[A-Za-z_\-.]*.{}"

        for name in filenames:
            print("Trying to read file: ", name, " @ ", dir_path)
            try:
                filename = [f for f in os.listdir(dir_path) if re.match(regex_base.format(name, format), f, re.IGNORECASE)][0]
                if filename:
                    if asNumpyArray:
                        csvs.append(pd.read_csv(os.path.join(dir_path, filename), header=header).to_numpy)
                    else:
                        csvs.append(pd.read_csv(os.path.join(dir_path, filename), header=header))
            except Exception as e:
                error.append( (name, e) )
            finally:
                for e, err in error:
                    print("Could not find or read file: {} --> {}".format(e, err))
                raise Exception("Something went wrong when reading csvs ")
        print("DONE reading csv's!")
        return csvs

    def get_rows_and_columns(self, dataframe=None, rows=None, columns=None):
        '''
        
        :param dataframe: 
        :param rows: 
        :param columns: 
        :return: 
        '''

        if dataframe is None:
            print("Faak off, datahandler get_rows_and_columns")
            # TODO fix exception

        if rows is None and columns is None:
            return dataframe
        elif rows is None:
            return dataframe.iloc[:, columns]
        elif columns is None:
            return dataframe.iloc[rows, :]
        else:
            return dataframe.iloc[rows, columns]

    def show_dataframe(self):
        print(self.dataframe_iterator)

    def head_dataframe(self, n=5):
        print(self.dataframe_iterator.head(n))

    def set_active_dataframe(self, dataframe):
        self.dataframe_iterator = dataframe

    def save_dataframe_to_path(self, path, dataframe=None):
        if dataframe is None:
            dataframe = self.get_dataframe_iterator()

        dataframe.to_csv(path)

    def remove_rows_where_columns_have_NaN(self, dataframe=None, columns=[]):
        df = dataframe or self.get_dataframe_iterator()
        if df is None:
            raise Exception("No dataframe detected")

        df.dropna(subset=columns, inplace=True)
        self.set_active_dataframe(df)

    def vertical_stack_dataframes(self, df1, df2, set_as_current_df=True):
        # TODO : CHECK IF THER IS MORE PATHS THAT NEEDS TO BE SET, THERE ARE!
        union = pd.merge(df1, df2, how='outer')
        if set_as_current_df:
            self.set_active_dataframe(union)

        return union

    def vertical_stack_csvs(self, csv_path_one,
                            csv_path_two,
                            column_names_df1=[],
                            column_names_df2=[],
                            rearranged_columns_after_merge=[],
                            index_column_name=None):
        # TODO : CHECK IF THER IS MORE PATHS THAT NEEDS TO BE SET, THERE ARE!

        df1 = pd.read_csv(csv_path_one)
        df2 = pd.read_csv(csv_path_two)

        if column_names_df1:
            df1.columns = column_names_df1
        if column_names_df2:
            df2.columns = column_names_df2

        self.vertical_stack_dataframes(df1, df2)

        if rearranged_columns_after_merge:
            self.rearrange_columns(rearranged_columns_after_merge)


        if index_column_name:
            self.set_column_as_index(column_name=index_column_name)

        return self.get_dataframe_iterator()


    def rearrange_columns(self, rearranged_columns):
        self.dataframe_iterator = self.dataframe_iterator[rearranged_columns]

    def tail_dataframe(self, n=5):
        print(self.dataframe_iterator.tail(n))





if __name__ == '__main__':
    pass


