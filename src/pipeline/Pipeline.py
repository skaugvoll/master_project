import sys, os
# try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
# except: print("SAdsadsadhsa;hkldasjkd")
import re, math, random, time
import numpy as np
import cwa_converter
import pickle
from multiprocessing import Process, Queue, current_process, freeze_support, Manager
from pipeline.DataHandler import DataHandler
from keras.models import load_model
from layers.normalize import Normalize
import utils.temperature_segmentation_and_calculation as temp_feature_util
from src.config import Config
from src import models
from keras.models import load_model
from tensorflow.keras.backend import clear_session



class Pipeline:
    def __init__(self):
        self.dh = DataHandler()
        self.dataframe = None
        self.model = None

    def printProgressBar(self, current, totalOperations, sizeProgressBarInChars, explenation=""):
        # try:
        #     import sys, time
        # except Exception as e:
        #     print("Could not import sys and time")

        fraction_completed = current / totalOperations
        filled_bar = round(fraction_completed * sizeProgressBarInChars)

        # \r means start from the beginning of the line
        fillerChars = "#" * filled_bar
        remains = "-" * (sizeProgressBarInChars - filled_bar)

        color = '\033[94m' # blue
        reset  = "\u001b[0m" # reset (turn of color)
        sys.stdout.write('\r{} {} {} [{:>7.2%}]'.format(
            color + explenation + reset,
            fillerChars,
            remains,
            fraction_completed
        ))

        sys.stdout.flush()


    def unzip_extractNconvert_temp_merge_dataset(self, rel_filepath, label_interval, label_mapping, unzip_path='../data/temp', unzip_cleanup=False, cwa_paralell_convert=True):
        # unzip cwas from 7z arhcive
        unzipped_path = self.dh.unzip_7z_archive(
            filepath=os.path.join(os.getcwd(), rel_filepath),
            unzip_to_path=unzip_path,
            cleanup=unzip_cleanup
        )

        print('UNZIPPED PATH RETURNED', unzipped_path)

        ##########################
        #
        #
        ##########################

        # convert the cwas to independent csv containing timestamp xyz and temp
        back_csv, thigh_csv = cwa_converter.convert_cwas_to_csv_with_temp(
            subject_dir=unzipped_path,
            out_dir=unzipped_path,
            paralell=cwa_paralell_convert
        )

        ##########################
        #
        #
        ##########################


        # Timesynch and concate csvs
        self.dh.merge_csvs_on_first_time_overlap(
            master_csv_path=back_csv,
            slave_csv_path=thigh_csv,
            rearrange_columns_to=[
                'time',
                'bx',
                'by',
                'bz',
                'tx',
                'ty',
                'tz',
                'btemp',
                'ttemp'
            ]
        )

        df = self.dh.get_dataframe_iterator()
        print(df.head(5))
        # input("looks ok ? \n")


        ##########################
        #
        #
        ##########################

        self.dh.convert_ADC_temp_to_C(
            dataframe=df,
            dataframe_path=None,
            normalize=False,
            save=True
        )

        df = self.dh.get_dataframe_iterator()
        print(df.head(5))
        # input("looks ok ? \n")

        ##########################
        #
        #
        ##########################


        print('SET INDEX TO TIMESTAMP')
        #test that this works with a dataframe and not only path to csv
        # thus pre-loaded and makes it run a little faster
        self.dh.convert_column_from_str_to_datetime_test(
                dataframe=df,
        )

        self.dh.set_column_as_index("time")
        print('DONE')

        ##########################
        #
        #
        ##########################


        print('MAKE NUMERIC')
        self.dh.convert_column_from_str_to_numeric(column_name="btemp")

        self.dh.convert_column_from_str_to_numeric(column_name="ttemp")
        print('DONE')

        ##########################
        #
        #
        ##########################


        print('ADDING LABELS')
        self.dh.add_new_column()
        print('DONE')

        self.dh.add_labels_file_based_on_intervals(
            intervals=label_interval,
            label_mapping=label_mapping
        )


        # ##########################
        # #
        # #
        # ##########################

        # dh.show_dataframe()
        df = self.dh.get_dataframe_iterator()

        return df, self.dh

    # TODO create method for unzip_extractNconvert_temp_stack_dataset() or adopt the above def..

    def get_features_and_labels(self, df, dh=None, columns_back=[0,1,2,6], columns_thigh=[3,4,5,7], column_label=[8]):
        if dh is None:
            dh = DataHandler()

        back_feat, thigh_feat, labels = None, None, None

        print(columns_back, columns_thigh, column_label)

        if columns_back:
            back_feat = dh.get_rows_and_columns(dataframe=df, columns=columns_back).values
        if columns_thigh:
            thigh_feat = dh.get_rows_and_columns(dataframe=df, columns=columns_thigh).values
        if column_label:
            labels = dh.get_rows_and_columns(dataframe=df, columns=column_label).values

        return back_feat, thigh_feat, labels


    ####################################################################################################################
    #                                   vPARALLELL PIPELINE EXECUTE WINDOW BY WINDOW CODEv                             #
    ####################################################################################################################

    # MODEL Klassifisering
    def model_classification_worker(self, input_q, output, model):
        for idx, window in iter(input_q.get, 'STOP'):
            # print("model_classification_worker executing")
            # print("MODLE CLASSIFICATION: \n", "IDX: ", idx, "\n","WINDOW: \n", window, "\n", "SHAPE: ", window.shape, "\n", "DIMS: ", window.ndim)

            # Vil ha in ett ferdig window, aka windows maa lages utenfor her og addes til queue
            # TODO: enten velge ut x antall av window, predikere de og ta avg result som LSTM (den med flest forekomster)
            # TODO: eller bare ett random row in window og predikerer den og bruker res som LSTM
            res = model.window_classification(window[0])[0]
            # print("RESSS: >>>>>> :: ", res, type(res))

            # output_queue.put((idx, window, res))
            # TODO remove window from output tuple, we do not need the temperature window anymore
            output.append((idx, res))




    def parallel_pipeline_classification_run(self, dataframe, rfc_model_path, lstm_models_paths, samples_pr_window, train_overlap=0.8, num_proc_mod=1, seq_lenght=None):
        '''

        :param dataframe: Pandas DataFrame
        :param rfc_model_path: str with path to saved RFC
        :param lstm_models_paths: dictionary containing lstm_mapping and path {rfc_result_number : model_path}
        :param samples_pr_window:
        :param train_overlap:
        :param num_proc_mod:
        :param num_proc_clas:
        :param seq_lenght:
        :return:
        '''

        # TODO: burde passe inn back, thigh og label columns til methoden også videre inn i get_features_and_labels

        self.dataframe = dataframe
        NUMBER_OF_PROCESSES_models = num_proc_mod

        # Create queues
        model_queue = Queue()
        # output_classification_queue = Queue()
        manager = Manager()
        output_classification_windows = manager.list()


        # Submit tasks for model klassifisering
        back_feat, thigh_feat, label = self.get_features_and_labels(self.dataframe) # returns numpy arrays
        back_feat = temp_feature_util.segment_acceleration_and_calculate_features(back_feat,
                                                                    samples_pr_window=samples_pr_window,
                                                                    overlap=train_overlap)

        thigh_feat = temp_feature_util.segment_acceleration_and_calculate_features(thigh_feat,
                                                                    samples_pr_window=samples_pr_window,
                                                                    overlap=train_overlap)

        # concatinates example : [[1,2,3],[4,5,6]] og [[a,b,c], [d,e,f]] --> [[1,2,3,a,b,c], [4,5,6,d,e,f]]
        # akas rebuild the dataframe shape
        both_features = np.hstack((back_feat, thigh_feat))


        # print("BOTH FEATURES SHAPE : ", both_features.shape)
        # TODO: EXTRACT THE CONVERTION INTO WINDOWS INTO OWN FUNC
        num_rows_in_window = 1
        if seq_lenght:
            num_rows = both_features.shape[0]
            num_rows_in_window = int( num_rows / seq_lenght)


        feature_windows = []
        last_index = 0

        for _ in range(num_rows_in_window):
            feature_windows.append(both_features[last_index:last_index + seq_lenght])
            last_index = last_index + seq_lenght

        both_features = np.array(feature_windows)
        # print(both_features.shape)

        number_of_tasks = both_features.shape[0]

        for idx, window in enumerate(both_features):
            model_queue.put((idx, window))


        # Lists to maintain processes
        processes_model = []

        # CREATE a worker processes on model klassifisering
        for _ in range(NUMBER_OF_PROCESSES_models):
            RFC = pickle.load(open(rfc_model_path, 'rb'))
            processes_model.append(Process(target=self.model_classification_worker,
                                           args=(model_queue,
                                                 output_classification_windows,
                                                 RFC)
                                           ))

        # START the worker processes
        for process in processes_model:
            process.start()

        # waith for tasks_queue to become empty before sending stop signal to workers
        while not model_queue.empty():
            # print("CURRENT: {}\nQUEUE SIZE: {}".format(number_of_tasks - model_queue.qsize(), model_queue.qsize()))
            self.printProgressBar(
                current=int(number_of_tasks - model_queue.qsize()),
                totalOperations=number_of_tasks,
                sizeProgressBarInChars=30,
                explenation="Model classification :: ")

        
        self.printProgressBar(
            current=int(number_of_tasks - model_queue.qsize()),
            totalOperations=number_of_tasks,
            sizeProgressBarInChars=30,
            explenation="Model classification :: ")
        print("DONE")

        # Tell child processes to stop waiting for more jobs
        for _ in range(NUMBER_OF_PROCESSES_models):
            model_queue.put('STOP')

        # print(">>>>>>>>>>>>>>>>> EUREKA <<<<<<<<<<<<<<<<<")

        # LET ALL PROCESSES ACTUALLY TERMINATE AKA FINISH THE JOB THEIR DOING
        while any([p.is_alive() for p in processes_model]):
            pass

        # print(">>>>>>>>>>>>>>>>> POT OF GOLD <<<<<<<<<<<<<<<<<")

        # join the processes aka block the threads, do not let them take on any more jobs
        for process in processes_model:
            process.join()

        # print("\nALL PROCESSES STATUS BEFORE TERMINATING:\n{}".format(processes_model))

        # Kill all the processes to release memory or process space or something. its 1.15am right now
        for process in processes_model:
            process.terminate()

        print(">>>>>>>>>>>>>>>>> ||||||||| <<<<<<<<<<<<<<<<<")

        # continue the pipeline work
        # ...
        # ...
        # ...


        # See results
        print("OUTPUT/Activities windows to classify : ", len(output_classification_windows))

        both_sensors_windows_queue = list(filter(lambda x: x[1] == '1', output_classification_windows))
        thigh_sensors_windows_queue = list(filter(lambda x: x[1] == '2', output_classification_windows))
        back_sensors_windows_queue = list(filter(lambda x: x[1] == '3', output_classification_windows))
        del output_classification_windows

        back_colums = ['back_x', 'back_y', 'back_z']
        thigh_colums = ['thigh_x', 'thigh_y', 'thigh_z']

        # x1 = model.get_features([dataframe], ['back_x', 'back_y', 'back_z'], batch_size=1, sequence_length=seq_lenght)
        xBack = np.concatenate([dataframe[back_colums].values[ : (len(dataframe) - len(dataframe) % seq_lenght) ] for dataframe in [dataframe]])
        xThigh = np.concatenate([dataframe[thigh_colums].values[: (len(dataframe) - len(dataframe) % seq_lenght)] for dataframe in [dataframe]])

        xBack = xBack.reshape(-1, seq_lenght, len(back_colums))
        xThigh = xThigh.reshape(-1, seq_lenght, len(thigh_colums))

        print("XBACK: ", xBack.shape)

        # BOTH
        bth_class = self.predict_on_one_window("1", lstm_models_paths, both_sensors_windows_queue, xBack, xThigh, seq_lenght)

        # THIGH
        thigh_class = self.predict_on_one_window('2', lstm_models_paths, thigh_sensors_windows_queue, xBack, xThigh, seq_lenght)

        # BACK
        back_class = self.predict_on_one_window('3', lstm_models_paths, back_sensors_windows_queue, xBack, xThigh, seq_lenght)

        return bth_class, thigh_class, back_class

        # classifiers = {}
        # for key in lstm_models_paths.keys():
        #     config = Config.from_yaml(lstm_models_paths[key]['config'], override_variables={})
        #     model_name = config.MODEL['name']
        #     model_args = dict(config.MODEL['args'].items(), **config.INFERENCE.get('extra_model_args', {}))
        #     model_args['batch_size'] = 1
        #
        #     model = models.get(model_name, model_args)
        #     # model.compile()
        #     model.model.load_weights(lstm_models_paths[key]['weights'])
        #     # model.compile()
        #
        #     classifiers[key] = {"model": model, "weights": lstm_models_paths[key]["weights"]}
        #
        #
        # start = 0
        # end = len(output_classification_windows) // 5
        # while start < end:
        #     meta = output_classification_windows.pop()
        #     wndo_idx, mod_clas = meta[0], meta[1]
        #     model = classifiers[mod_clas]['model']
        #     # model.compile() # with this as the only compile it started to run agian...
        #     # weights_path = classifiers[mod_clas]['weights']
        #
        #     # get the correct features from the dataframe, and not the temperature feature
        #     x1 = model.get_features([dataframe], ['back_x', 'back_y', 'back_z'], batch_size=1, sequence_length=seq_lenght)
        #     x2 = model.get_features([dataframe], ['thigh_x', 'thigh_y', 'thigh_z'], batch_size=1, sequence_length=seq_lenght)
        #     x1 = x1[wndo_idx].reshape(1, seq_lenght, x1.shape[2])
        #     x2 = x2[wndo_idx].reshape(1, seq_lenght, x2.shape[2])
        #     if mod_clas == "1":  # both sensors
        #         target, prob = model.predict_on_one_window(window=[x1, x2])
        #     elif mod_clas == '2':  # just thigh sensor
        #         target, prob = model.predict_on_one_window(window=x2)
        #     elif mod_clas == '3':  # just back sensor
        #         target, prob = model.predict_on_one_window(window=x1)
        #     print(target, prob)
        #     # print(res)
        #     self.printProgressBar(start, end, 20, explenation="Activity classification prog. :: ")
        #     start += 1
        #
        # self.printProgressBar(start, end, 20, explenation="Activity classification prog. :: ")

    def predict_on_one_window(self, model_num, lstm_models_paths, sensors_windows_queue, xBack, xThigh, seq_lenght, time_col='time'):
        model = None
        config = Config.from_yaml(lstm_models_paths[model_num]['config'], override_variables={})
        model_name = config.MODEL['name']
        model_args = dict(config.MODEL['args'].items(), **config.INFERENCE.get('extra_model_args', {}))
        model_args['batch_size'] = 1

        model = models.get(model_name, model_args)
        model.compile()
        model.model.load_weights(lstm_models_paths[model_num]['weights'])
        model.compile()
        start = 0
        end = len(sensors_windows_queue)

        classifications = []

        for meta in sensors_windows_queue:
            wndo_idx, mod = meta[0], meta[1]
            task = None
            if mod == "1":
                task = "Both"
                if time_col in xThigh:
                    timestamp = xThigh[time_col]
                else:
                    timestamp = "NA"
                x1 = xBack[wndo_idx].reshape(1, seq_lenght, xBack.shape[2])
                x2 = xThigh[wndo_idx].reshape(1, seq_lenght, xThigh.shape[2])
                target, prob = model.predict_on_one_window(window=[x1, x2])
                classifications.append((timestamp, prob, target))

            elif mod == '2':
                task = "Thigh"
                if time_col in xThigh:
                    timestamp = xThigh[time_col]
                else:
                    timestamp = "NA"
                x = xThigh[wndo_idx].reshape(1, seq_lenght, xThigh.shape[2])
                target, prob = model.predict_on_one_window(window=x)
                classifications.append((timestamp, prob, target))

            elif mod == '3':
                task = "Back"
                if time_col in xBack:
                    timestamp = xBack[time_col]
                else:
                    timestamp = "NA"
                x = xBack[wndo_idx].reshape(1, seq_lenght, xBack.shape[2])
                target, prob = model.predict_on_one_window(window=x)
                classifications.append((timestamp, prob, target))

            # print("<<<<>>>>><<<>>>: \n", ":: " + model_num +" ::", target, prob)
            self.printProgressBar(start, end, 20, explenation=task + " activity classification prog. :: ")
            start += 1

        self.printProgressBar(start, end, 20, explenation=task + " activity classification prog. :: ")
        print() # create new line
        try:
            del model  # remove the model
            clear_session()
        except Exception as e:
            print("Could not remove model from memory.")
        finally:
            # RETURN TIMESTAMP, CONFIDENCE/PROB and TARGET
            return np.array(classifications)

    @staticmethod
    def load_model_weights(model, weights_path):
        model.load_weights(weights_path)

    ####################################################################################################################
    #                                   ^PARALLELL PIPELINE EXECUTE WINDOW BY WINDOW CODE^                             #
    ####################################################################################################################

    def create_large_dafatframe_from_multiple_input_directories(self,
                                                                list_with_subjects,
                                                                back_keywords=['Back'],
                                                                thigh_keywords = ['Thigh'],
                                                                label_keywords = ['GoPro', "Labels"],
                                                                out_path=None,
                                                                merge_column = None,
                                                                master_columns = ['bx', 'by', 'bz'],
                                                                slave_columns = ['tx', 'ty', 'tz'],
                                                                rearrange_columns_to = None,
                                                                save=False,
                                                                added_columns_name=["new_col"]
                                                                ):


        subjects = DataHandler.findFilesInDirectoriesAndSubDirs(list_with_subjects,
                                                         back_keywords,
                                                         thigh_keywords,
                                                         label_keywords,
                                                         verbose=True)

        # print(subjects)
        merged_df = None
        dh = DataHandler()
        dh_stacker = DataHandler()
        for idx, root_dir in enumerate(subjects):
            subject = subjects[root_dir]
            # print("SUBJECT: \n", subject)

            master = os.path.join(root_dir, subject['backCSV'])
            slave = os.path.join(root_dir, subject['thighCSV'])
            label = os.path.join(root_dir, subject['labelCSV'])

            # dh = DataHandler()
            dh.merge_csvs_on_first_time_overlap(
                master,
                slave,
                out_path=out_path,
                merge_column=merge_column,
                master_columns=master_columns,
                slave_columns=slave_columns,
                rearrange_columns_to=rearrange_columns_to,
                save=save,
                left_index=True,
                right_index=True
            )

            dh.add_columns_based_on_csv(label, columns_name=added_columns_name, join_type="inner")

            if idx == 0:
                merged_df = dh.get_dataframe_iterator()
                continue

            merged_old_shape = merged_df.shape
            # vertically stack the dataframes aka add the rows from dataframe2 as rows to the dataframe1
            merged_df = dh_stacker.vertical_stack_dataframes(merged_df, dh.get_dataframe_iterator(),
                                                             set_as_current_df=False)

        #     print(
        #     "shape merged df: ", merged_df.shape, "should be ", dh.get_dataframe_iterator().shape, "  more than old  ",
        #     merged_old_shape)
        #
        # print("Final merge form: ", merged_df.shape)
        return merged_df


    def train_lstm_model(self,
                         training_dataframe,
                         back_cols,
                         thigh_cols,
                         config_path,
                         label_col=None,
                         validation_dataframe=None,
                         batch_size=None,
                         sequence_length=None,
                         save_to_path=None,
                         save_model=False,
                         save_weights=False
                         ):
        '''
        :param training_dataframe: Pandas Dataframe
        :param back_cols: list containing the labels that identify back feature columns
        :param thigh_cols: list containing the labels that identify thigh feature columns
        :param config_path: relative path to the configuration of LSTM
        :param label_col: the name of the column in dataframe that identifies Class/Target
        :param validation_dataframe: Pandas Dataframe with same columns as training_dataframe
        :param batch_size: Batch_size, should be given in config file, but this will over overwrite
        :param sequence_length: sequence_length, should be given in config file, but this will over overwrite
        :param save_to_path: if given, saves the trained weights and/or model to the given path
        :param save_model: if path and save_model [True | False] saves the model to the path
        :param save_weights: if path and save_weights [True | False] saves the weight to the path with suffix: _weights
        :return: the trained model object
        '''
        '''
        src/models/__init__.py states:
            Train the model. Usually, we like to split the data into training and validation
            by producing a dichotomy over the subjects. This means that

            Inputs:
              - train_data: list<pd.DataFrame>
                A list of dataframes, intended to be used for model fitting. It's columns are:
                  <back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label>
                Where the first 6 corresponds to sensor data as floats and the last one is an
                integer corresponding to the annotated class
              - valid_data: list<pd.DataFrame>
                Same as train_data, execpt usually a much shorter list.
              - **kwargs:
                Extra arguments found in the model's config
        '''

        config = Config.from_yaml(config_path, override_variables={})
        # config.pretty_print()

        model_name = config.MODEL['name']
        model_args = dict(config.MODEL['args'].items(), **config.INFERENCE.get('extra_model_args', {}))
        model = models.get(model_name, model_args)

        # if passed in validation dataframe, give it its propper format.
        if not validation_dataframe is None:
            validation_dataframe = [validation_dataframe]


        # potentially overwrite config variables
        batch_size = batch_size or config.TRAINING['args']['batch_size']
        sequence_length = sequence_length or config.TRAINING['args']['sequence_length']
        callbacks = config.TRAINING['args']['callbacks'] or None

        cols = None
        if back_cols and thigh_cols:
            self.num_sensors = 2
            cols = [back_cols, thigh_cols]
            model.train(
                train_data=[training_dataframe],
                valid_data=validation_dataframe,
                epochs=config.TRAINING['args']['epochs'],
                batch_size=batch_size, # gets this from config file when init model
                sequence_length=sequence_length, # gets this from config file when init model
                back_cols=back_cols,
                thigh_cols=thigh_cols,
                label_col=label_col,
            )
        else:
            cols = back_cols or thigh_cols
            self.num_sensors = 1
            model.train(
                train_data=[training_dataframe],
                valid_data=validation_dataframe,
                callbacks=[],
                epochs=config.TRAINING['args']['epochs'],
                batch_size=batch_size,
                sequence_length=sequence_length,
                cols=cols,
                label_col=label_col
            )

        #####
        # Save the model / weights
        #####
        if save_to_path and (save_weights or save_model):
            print("Done saving: {}".format(
                    model.save_model_andOr_weights(path=save_to_path, model=save_model, weight=save_weights)
                )
            )

        self.config = config
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.cols = cols
        self.model = model
        return self.model


    def evaluate_lstm_model(self, dataframe, label_col, num_sensors=None, model=None, back_cols=None, thigh_cols=None, cols=None, batch_size=None, sequence_length=None):
        model = model or self.model
        num_sensors = num_sensors or self.num_sensors

        if num_sensors == 2:
            return model.evaluate(dataframes=[dataframe],
                          batch_size=batch_size or self.config.TRAINING['args']['batch_size'],
                          sequence_length=sequence_length or self.config.TRAINING['args']['sequence_length'],
                          back_cols=self.cols[0] or back_cols,
                          thigh_cols=self.cols[1] or thigh_cols,
                          label_col=label_col)
        elif num_sensors == 1:
            return model.evaluate(dataframes=[dataframe],
                          batch_size=batch_size or self.config.TRAINING['args']['batch_size'],
                          sequence_length=sequence_length or self.config.TRAINING['args']['sequence_length'],
                          cols=self.cols or cols,
                          label_col=label_col)
        else:
            print("Pipeline.py :: evaluate_lstm_model ::")
            raise NotImplementedError()



if __name__ == '__main__':
    p = Pipeline()