import os
import re
import axivity
import pandas as pd
import numpy as np
import time
import json
from utils import zip_utils
from utils import csv_loader
from utils import progressbar
from sklearn.model_selection import train_test_split

class DataHandler():
    # TODO: change all places pd.read_csv is called to self.load_dataframe_from_csv(...)

    def __init__(self):
        self.name = None
        self.data_unzipped_path = None
        self.data_synched_csv_path = None
        self.data_cleanup_path = None
        self.data_input_folder = os.getcwd() + '../../data/input'
        self.data_output_folder = os.getcwd() + '../../data/output'
        self.data_temp_folder = os.getcwd() + '/../data/temp'
        self.dataframe_iterator = None

    def unzip_synch_cwa(self, filepath='filepath', temp_dir='working_dir', unzip_cleanup=False, timeSynchedName=None):
        self.data_unzipped_path = self.unzip_7z_archive(
            filepath=os.path.join(os.getcwd(), filepath),
            unzip_to_path=self.data_temp_folder,
            cleanup=unzip_cleanup
        )
        self.data_unzipped_path = os.path.relpath(self.data_unzipped_path)

        with axivity.timesynched_csv(self.data_unzipped_path, clean_up=False, synched_filename=timeSynchedName) as synch_csv:
            print("Saved timesynched csv")

        synched_csv = list(filter(lambda x: 'timesync' in x, os.listdir(self.data_unzipped_path)))
        self.data_synched_csv_path = (self.data_unzipped_path + '/' + synched_csv[0])

        return self.data_unzipped_path


    def unzip_7z_archive(self, filepath, unzip_to_path='../data/temp', return_inner_dir=True, cleanup=True):
        try:
            self._check_paths(filepath, unzip_to_path)
        except NotADirectoryError as e:
            print("Temp directory not found.. Crating it now")
            self.create_output_dir(unzip_to_path, '')

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
        else:
            # TODO: change this to elif and pass in a parameter with default strict or something...
            try:
                p = "/".join(unzipped_dir_path.split("/")[:-1])
                pwd = os.getcwd()
                p = os.path.join(pwd, p)
                os.system("chmod 777 -R {}".format(p))
            except Exception as e:
                print("Could not give easy access rights")

        return unzipped_dir_path

    def merge_multiple_csvs(self, master_csv_path, slave_csv_path, slave2_csv_path, out_path=None,
                            master_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz'],
                            slave_columns=['time', 'bx1', 'by1', 'bz1', 'btemp'],
                            slave2_columns=['time', 'tx1', 'ty1', 'tz1', 'ttemp'],
                            rearrange_columns_to=None,
                            merge_how='inner',
                            merge_on='time',
                            header_value=None,
                            save=False):

        print("READING MASTER CSV")
        master_df = pd.read_csv(master_csv_path, header=header_value)
        master_df.columns = master_columns

        print("READING BACK CSV")
        slave_df = pd.read_csv(slave_csv_path, header=header_value)
        slave_df.columns = slave_columns

        print("READING THIGH CSV")
        slave2_df = pd.read_csv(slave2_csv_path, header=header_value)
        slave2_df.columns = slave2_columns

        # Merge the csvs
        print("MERGING CSVS WITH MASTER")
        merged_df = master_df.merge(slave_df, on=merge_on, how=merge_how).merge(slave2_df, on=merge_on, how=merge_how)

        ## Rearrange the columns
        if not rearrange_columns_to is None:
            print("REARRANGING CSV COLUMNS")
            merged_df = merged_df[rearrange_columns_to]

        if out_path is None:
            master_file_dir, master_filename_w_format = os.path.split(master_csv_path)
            out_path = os.path.join(master_file_dir,
                                    master_filename_w_format.split('.')[0] + '_TEMP_SYNCHED_BT.csv')

        else:
            out_path_dir, out_path_filename = os.path.split(out_path)
            if out_path_filename == '':
                out_path_filename = os.path.basename(master_csv_path).split('.')[0] + '_TEMP_SYNCHED_BT.csv'

            if not os.path.exists(out_path_dir):
                print('Creating output directory... ', out_path_dir)
                os.makedirs(out_path_dir)

            out_path = os.path.join(out_path_dir, out_path_filename)

        print("DONE, here is a sneak peak:\n", merged_df.head(5))
        if save:
            print("SAVING MERGED CSV")
            merged_df.to_csv(out_path, index=False, float_format='%.6f')
            print("Saved synched and merged as csv to : ", os.path.abspath(out_path))

        self.dataframe_iterator = merged_df

        # self.data_cleanup_path = os.path.abspath(out_path[:out_path.find('.7z/') + 4])
        # self.data_synched_csv_path = os.path.abspath(out_path)
        # self.name = os.path.basename(out_path)
        self.data_temp_folder = os.path.abspath(os.path.split(out_path)[0])

        #Make two temp txt files based on new merged DF
        self.write_temp_to_txt(
            dataframe=self.dataframe_iterator
        )

    def write_temp_to_txt(self, dataframe=None, dataframe_path=None):

        df = None

        if not dataframe is None:
            df = dataframe

        elif dataframe is None and not dataframe_path is None:
            try:
                df = pd.read_csv(dataframe_path)
            except Exception as e:
                print("Did not give a valid csv_path")
                raise e

        elif dataframe is None and dataframe_path is None:
            print("Need to pass either dataframe or csv_path")
            raise Exception("Need to pass either dataframe or csv_path")

        elif dataframe and dataframe_path:
            df = dataframe

        print("STARTING writing temp to file")

        #Static list for now. maybe change later?
        for i in ['btemp', 'ttemp']:
            start_time = time.time()
            print('Creating %s txt file' % i)
            nanIndex = list(df[i].index[df[i].apply(np.isnan)])
            valildIndex = list(df[i].index[df[i].notnull()])
            firstLastIndex = [nanIndex[0], nanIndex[-1]]

            file = open(self.data_temp_folder + '/' + i + '.txt', 'w')
            if firstLastIndex[0] < valildIndex[0]:
                file.write(str(str((float(df.loc[valildIndex[0], i]) * 300 / 1024) - 50) + '\n') * (valildIndex[0] - firstLastIndex[0]))

            for j in range(len(valildIndex) - 1):
                progressbar.printProgressBar(j, len(valildIndex), 10)
                if valildIndex[j] + 1 == valildIndex[j + 1]:
                    file.write(str((float(df.loc[valildIndex[j], i]) * 300 / 1024) - 50) + '\n')
                else:
                    file.write(str((float(df.loc[valildIndex[j], i]) * 300 / 1024) - 50) + '\n')
                    file.write(str(str((float(df.loc[valildIndex[j+1], i]) * 300 / 1024) - 50) + '\n') * ((valildIndex[j + 1]) - (valildIndex[j] + 1)))
            if firstLastIndex[1] > valildIndex[-1]:
                file.write(str(str((float(df.loc[valildIndex[-1], i]) * 300 / 1024) - 50) + '\n') * ((firstLastIndex[-1]+1) - valildIndex[-1]))
            else:
                file.write(str((float(df.loc[valildIndex[-1], i]) * 300 / 1024) - 50) + '\n')


            file.close()
            print("---- %s seconds ---" % (time.time() - start_time))

        return self.get_dataframe_iterator()

    def concat_timesynch_and_temp(self,
                                  master_csv_path,
                                  btemp_txt_path,
                                  ttemp_txt_path,
                                  master_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz'],
                                  back_temp_column=['btemp'],
                                  thigh_temp_column=['ttemp'],
                                  header_value=None,
                                  save=False):

        print("READING MASTER CSV")
        master_df = pd.read_csv(master_csv_path, header=header_value)
        master_df.columns = master_columns

        print("READING BACK TXT")
        btemp_df = pd.read_csv(btemp_txt_path, header=header_value)
        btemp_df.columns = back_temp_column

        print("READING THIGH TXT")
        ttemp_df = pd.read_csv(ttemp_txt_path, header=header_value)
        ttemp_df.columns = thigh_temp_column

        # Merge the csvs
        print("MERGING MASTER AND CSVS")
        merged_df = pd.concat([master_df, btemp_df, ttemp_df], axis=1,)


        master_file_dir, master_filename_w_format = os.path.split(master_csv_path)
        out_path = os.path.join(master_file_dir, master_filename_w_format.split('.')[0] + '_TEMP_BT.csv')

        self.dataframe_iterator = merged_df

        print("SAVING MERGED CSV")
        print("DONE, here is a sneak peak:\n", merged_df.head(5))
        if save:
            print("Saving")
            merged_df.to_csv(out_path, index=False, float_format='%.6f')
            print("Saved synched and merged as csv to : ", os.path.abspath(out_path))

        return self.dataframe_iterator


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
            raise NotADirectoryError('Provided temp directory "%s" does not exist' % temp_dir)

    def create_output_dir(self, output_dir_path, name):
        # Create output directory if it does not exist
        self.data_output_folder = os.path.join(output_dir_path, name)
        if not os.path.exists(self.data_output_folder):
            os.makedirs(self.data_output_folder)



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

        # self.dataframe_iterator = pd.read_csv(self.data_synched_csv_path, header=header, names=columns)

        # # Read csv files in chunks
        if whole_days:
            # Use custom loader that scans to first midnight if --whole-days is enabled
            '''
            if columns is None, the column names must be the first row in the csv!
            else if columns is specifiec, the pd.read_csv will set header param to None, meaning use the first row as data row
            '''
            self.dataframe_iterator = csv_loader.csv_chunker(self.data_synched_csv_path, chunk_size, ts_index=0,
                                                             columns=None, n_days=max_days)

        else:
            # TODO : read up on what the chunksize and parse_dates parameter does
            # Otherwise, just load with pandas
            # self.dataframe_iterator = pd.read_csv(self.data_synched_csv_path, header=None, chunksize=chunk_size,
            #                                       names=columns, parse_dates=[0])
            self.dataframe_iterator = pd.read_csv(self.data_synched_csv_path, header=header, names=columns)


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

    def get_unzipped_path(self):
        return self.data_unzipped_path

    def get_synched_csv_path(self):
        return self.data_synched_csv_path

    def cleanup_temp_folder(self):
        print("Cleaning {}".format(self.data_cleanup_path))
        try:
            zip_utils.clean_up_working_dir(self.data_cleanup_path)
            print("Cleanup SUCCESS")
        except:
            print("Cleanup FAILED")

    def merge_csvs_on_first_time_overlap(self,
                                         master_csv_path,
                                         slave_csv_path,
                                         out_path=None,
                                         merge_column="time",
                                         master_columns=['time', 'bx', 'by', 'bz', 'btemp'],
                                         slave_columns=['time', 'tx', 'ty', 'tz', 'ttemp'],
                                         rearrange_columns_to=None,
                                         save=True,
                                         verbose=False,
                                         **kwargs
                                         ):
        '''
        Master_csv is the csv that the first recording is used as starting point

        :param master_csv_path:
        :param slave_csv_path:
        :param out_path:
        :param merge_column: the column name or default Time || can be None
        :param master_columns: the name to give each column in master csv
        :param slave_columns: the name to give each column in master csv
        :param rearrange_columns_to:
        :param save: default True
        :param verbose: default False, prints out what it does
        :param **kwargs: additional keyword=value arguments, that can be used with the pandas.merge() function
        :return: None
        '''

        # TODO: PASS IN MASTER AND SLAVE COLUMN NAMES
        # TODO change all the pd.read_csv s to use datahandlers own load_from_csv function
        if verbose:
            print("READING MASTER CSV")
        master_df = pd.read_csv(master_csv_path)
        master_df.columns = master_columns

        if verbose:
            print("READING SLAVE CSV")
        slave_df = pd.read_csv(slave_csv_path)
        slave_df.columns = slave_columns

        # Merge the csvs
        if verbose:
            print("MERGING MASTER AND SLAVE CSV")
        merged_df = master_df.merge(slave_df, on=merge_column, **kwargs)

        # print("MASTER SHAPE: {} \n SLAVE SHAPE: {} \n MERGED SHAPE: {}".format(master_df.shape, slave_df.shape, merged_df.shape))

        ## Rearrange the columns
        if not rearrange_columns_to is None:
            if verbose:
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

        if save:
            print("SAVING MERGED CSV")
            merged_df.to_csv(out_path, index=False)
            # merged_df.to_csv(out_path, index=False, float_format='%.6f')
            print("Saved synched and merged as csv to : ", os.path.abspath(out_path))


        self.dataframe_iterator = merged_df

        self.data_cleanup_path = os.path.abspath(out_path[:out_path.find('.7z/') + 4])
        self.data_synched_csv_path = os.path.abspath(out_path)
        self.name = os.path.basename(out_path)
        self.data_temp_folder = os.path.abspath(os.path.split(out_path)[0])
        return self.get_dataframe_iterator()



    def convert_column_from_str_to_datetime(self,
                                                 dataframe=None,
                                                 dataframe_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz', 'btemp', 'ttemp'],
                                                 column_name="time",
                                                 verbose=False):
        # TODO if dataframe is actually dataframe object, self.dataframe_iterator = dataframe
        # TODO remove this and change all places it it called to call convert_column_from_str_to_datetime

        # TODO: PERHAPS MAYBE WHO KNOWS, is this the one we should use and just pass in column names????

        if isinstance(dataframe, str):
            self.dataframe_iterator = pd.read_csv(dataframe)
            if verbose: print(self.dataframe_iterator.head(5)); print()


            self.dataframe_iterator.columns = dataframe_columns
        else:
            if verbose: print("USING THE Datahandlers own dataframe-Instance")
            else: pass

        self.dataframe_iterator[column_name] = pd.to_datetime(self.dataframe_iterator[column_name])
        if verbose: print(self.dataframe_iterator.dtypes)

    def convert_column_from_str_to_numeric(self, column_name="ttemp", verbose=False):
        self.dataframe_iterator[column_name] = pd.to_numeric(self.dataframe_iterator[column_name])
        if verbose: print(self.dataframe_iterator.dtypes)

    def set_column_as_index(self, column_name, verbose=False):
        self.dataframe_iterator.set_index(column_name, inplace=True)
        if verbose: print("The dataframe index is now: ", self.dataframe_iterator.index.name)

    def add_new_column(self, name='label', default_value=np.nan, verbose=False):
        self.dataframe_iterator.insert(len(self.dataframe_iterator.columns), name, value=default_value)
        if verbose: print(self.dataframe_iterator.describe())

    def add_columns_based_on_csv(self, path, columns_name=['label'], join_type='outer', **kwargs_read_csv):
        df_label = pd.read_csv(path, **kwargs_read_csv)
        df_label.columns = columns_name

        self.dataframe_iterator = pd.concat(objs=[self.dataframe_iterator, df_label], join=join_type, axis=1, sort=False)

    def read_labels_from_json(self, filepath):
        with open(filepath) as json_file:
            return json.load(json_file)

    def add_labels_file_based_on_intervals(self, intervals={}, label_col_name="label", label_mapping={}, verbose=False):
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
            if verbose: print("Faak off, datahandler add_labels_file_based_on_intervals")

        for label in intervals:
            if verbose: print("label", label)
            for interval in intervals[label]:
                if verbose: print("INTERVAL", interval)

                # there are noe intervals with this label
                if len(interval) == 0:
                    continue

                date = interval[0]
                start = interval[1]
                end = interval[2]

                start_string = '{} {}'.format(date, start)
                end_string = '{} {}'.format(date, end)
                # get indexes to add label to
                self.dataframe_iterator.loc[start_string:end_string, label_col_name] = label

        if verbose: print(self.dataframe_iterator)

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
            if type(columns[0]) == str:
                return dataframe.loc[:, columns]
            return dataframe.iloc[:, columns]
        elif columns is None:
            if type(rows[0]) == str:
                return dataframe.loc[rows, :]
            return dataframe.iloc[rows, :]
        else:
            if type(rows[0] == str) and type(columns[0]) == str:
                return dataframe.loc[rows, columns]
            return dataframe.iloc[rows, columns]



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


    def show_dataframe(self):
        print(self.dataframe_iterator)

    def head_dataframe(self, n=5):
        print(self.dataframe_iterator.head(n))

    def tail_dataframe(self, n=5):
        print(self.dataframe_iterator.tail(n))


    def _adc_to_c(self, row, normalize=False):
        row['btemp'] = (row['btemp'] * 300 / 1024) - 50
        row['ttemp'] = (row['ttemp'] * 300 / 1024) - 50

        if normalize:
            print("NORAMLIZATION NOT IMPLEMENTED YET")
            # TODO IMPLEMENT NORMALIZATION

            # temperature_celsius_b = (row['btemp'] * 300 / 1024) - 50
            # temperature_celsius_t = (row['ttemp'] * 300 / 1024) - 50

        return row

    def convert_ADC_temp_to_C(self, dataframe=None, dataframe_path=None, normalize=False, save=False, verbose=False):
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

        if verbose: print("STARTING converting adc to celcius...")
        self.dataframe_iterator = df.apply(self._adc_to_c, axis=1, raw=False, normalize=normalize)

        print(self.dataframe_iterator.describe(), "\n")
        print()
        print(self.dataframe_iterator.dtypes)
        print()
        print("DONE, here is a sneak peak:\n", self.dataframe_iterator.head(5))

        if (dataframe_path or self.data_synched_csv_path) and save:
            path = dataframe_path or self.data_synched_csv_path
            self.dataframe_iterator.to_csv(path, index=False, float_format='%.6f')

        return self.get_dataframe_iterator()



    def vertical_stack_dataframes(self, df1, df2, set_as_current_df=True):
        # TODO : CHECK IF THER IS MORE PATHS THAT NEEDS TO BE SET, THERE ARE!
        union = pd.concat([df1, df2], join='outer')
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


    @staticmethod
    def split_df_into_training_and_test(data, label_col=None, split_rate=.2, shuffle=False):
        '''

        :param data:
        :param label_col:
        :param split_rate: how many percent hould be in the test TEST set!
        :return: IF label_col is given, returns x_train, x_test, y_train, y_test ELSE: train, test
        '''
        # how_much_is_training_data = 0.8
        # split_idx = int(dataframe.shape[0] * how_much_is_training_data)
        # df1 = dataframe.iloc[: split_idx]
        # df2 = dataframe.iloc[split_idx:]  # 1 - how_musch_is_training_data % of data
        # return df1, df2

        if label_col:
            x = data.iloc[:, :label_col]
            y = data.iloc[: , label_col]
            return train_test_split(x, y, test_size=split_rate, shuffle=shuffle)
        else:
            return train_test_split(data, test_size=split_rate, shuffle=shuffle)

    @staticmethod
    def create_batches_with_seq_length(dataframes, columns, sequence_length, stateful=None, batch_size=None):
        '''

        :param self:
        :param dataframes: list with dataframes
        :param columns: list with columns (features) to keep in the new batch
        :param batch_size:
        :param sequence_length:
        :param stateful: Only fill X with full batches
        :return:
        '''


        X = np.concatenate([
            dataframe[columns].values[: (len(dataframe) - len(dataframe) % sequence_length)] for dataframe in dataframes
        ]).reshape( -1, sequence_length, len(columns) )

        if stateful and batch_size:
            # No half-batches are allowed if using stateful. TODO: This should probably be done very differently
            batch_size = batch_size
            X = X[: (len(X) - len(X) % batch_size)]

        return X

    @staticmethod
    def findFilesInDirectoriesAndSubDirs(list_with_subjects, back_keywords, thigh_keywords, label_keywords, synched_keywords, verbose=False):
        '''
        takes in a list with path to directories containing files to look for based on keywords,
        the files must be at root level in the directory.

        NOTE: Files must use _ and or . for delimiter in filename
        Note: case Insensitive

        :param list_with_subjects:
        :param back_keywords:
        :param thigh_keywords:
        :param label_keywords:
        :return: a dictionary {subject_dir_name: {backCSV : x, thighCSV : y, labelCSVL : z}}
        '''
        # check if directory contains files or another directory:
        sub_dirs = []
        for sub in list_with_subjects:
            for dirpath, dirnames, filenames in os.walk(sub):
                # if not leaf directory, but has files
                if filenames and not dirnames == []:
                    sub_dirs.append(dirpath)

                # if leaf directory and has files, not empty directory
                if dirnames == [] and filenames:
                    sub_dirs.append(dirpath)

        subjects = {}
        for subject in sub_dirs:
            if not os.path.exists(subject):
                print("Could not find Subject at path: ", subject)

            files = {}
            for sub_files_and_dirs in os.listdir(subject):
                # print(sub_files_and_dirs)
                words = re.split("[_ .]", sub_files_and_dirs)
                if "cwa" in words:
                    continue

                words = list(map(lambda x: x.lower(), words))

                check_for_matching_word = lambda words, keywords: [True if keyword.lower() == word.lower() else False
                                                                   for word in words for keyword in keywords]

                if any(check_for_matching_word(words, back_keywords)):
                    files["backCSV"] = sub_files_and_dirs

                elif any(check_for_matching_word(words, thigh_keywords)):
                    files["thighCSV"] = sub_files_and_dirs

                elif any(check_for_matching_word(words, label_keywords)):
                    files["labelCSV"] = sub_files_and_dirs

                elif any(check_for_matching_word(words, synched_keywords)):
                    files["synchedCSV"] = sub_files_and_dirs

            subjects[subject] = files
        if verbose:
            print("Found following subjects")
            for k, _ in subjects.items():
                print("Subject: {}".format(k))
        return subjects


    @staticmethod
    def getAttributeOrReturnDefault(dict, name, default=None):
        return dict.get(name, default)


if __name__ == '__main__':
    pass


