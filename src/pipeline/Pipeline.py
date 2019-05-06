import sys, os
import numpy as np
import cwa_converter
import pickle
import pprint
import pandas as pd
import utils.temperature_segmentation_and_calculation as temp_feature_util
from multiprocessing import Process, Queue, Manager
from pipeline.DataHandler import DataHandler
from pipeline.Plotter import Plotter
from utils import progressbar
from src.config import Config
from src import models
from src.utils.WindowMemory import WindowMemory
from src.utils.ColorPrint import ColorPrinter
from src.utils.cmdline_input import cmd_input
from tensorflow.keras.backend import clear_session
from pipeline.resampler import main as resampler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score


class Pipeline:
    def __init__(self):
        self.dh = DataHandler()
        self.colorPrinter = ColorPrinter()
        self.plotter = Plotter()
        self.dataframe = None
        self.model = None

    def unzipNsynch(self, rel_filepath, unzip_path='../../data/temp', cwa_paralell_convert=True, save=False):
        # unzip cwas from 7z arhcive
        self.dh.unzip_synch_cwa(rel_filepath)

        back_csv, thigh_csv = cwa_converter.convert_cwas_to_csv_with_temp(
            subject_dir=self.dh.get_unzipped_path(),
            out_dir=self.dh.get_unzipped_path(),
            paralell=cwa_paralell_convert
        )

        self.dh.merge_multiple_csvs(
            master_csv_path=self.dh.get_synched_csv_path(),
            slave_csv_path=back_csv,
            slave2_csv_path=thigh_csv,
            merge_how='left',
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

        self.dh.concat_dataframes(
            master_path=self.dh.get_synched_csv_path(),
            slave_path=self.dh.get_unzipped_path() + '/btemp.txt',
            slave2_path=self.dh.get_unzipped_path() + '/ttemp.txt',
            save=save
        )


        print('SET INDEX TO TIMESTAMP')
        # test that this works with a dataframe and not only path to csv
        # thus pre-loaded and makes it run a little faster
        self.dh.convert_column_from_str_to_datetime(
            dataframe=self.dh.get_dataframe_iterator(),
        )

        self.dh.set_column_as_index("time")
        print('DONE')

        ##########################

        print('MAKE NUMERIC')
        self.dh.convert_column_from_str_to_numeric(column_name="btemp")

        self.dh.convert_column_from_str_to_numeric(column_name="ttemp")
        print('DONE')
        return self.dh


    def get_features_and_labels_as_np_array(self, df, dh=None, back_columns=[0, 1, 2, 6], thigh_columns=[3, 4, 5, 7], label_column=[8]):
        if dh is None:
            dh = DataHandler()

        back_feat, thigh_feat, labels = None, None, None

        # print(back_columns, thigh_columns, label_column)

        if back_columns:
            back_feat = dh.get_rows_and_columns(dataframe=df, columns=back_columns).values
        if thigh_columns:
            thigh_feat = dh.get_rows_and_columns(dataframe=df, columns=thigh_columns).values
        if label_column:
            labels = dh.get_rows_and_columns(dataframe=df, columns=label_column).values

        return back_feat, thigh_feat, labels


    def addLables(self, intervals, column_name, datahandler=None):
        '''
        This needs to make sure that the pipeline Object has been used, or has been given a datahandler
        :param intervals:
        :param column_name:
        :param datahandler:
        :return:
        '''
        dh = None
        intervals = intervals
        if datahandler is None:
            dh = self.dh
        else:
            dh = datahandler
        dh.add_new_column(column_name)
        if isinstance(intervals, str):
            # use datahandler function for reading json file with labels as dict
            intervals = dh.read_labels_from_json(filepath=intervals)

        dh.add_labels_file_based_on_intervals(intervals=intervals, label_col_name=column_name)


    @staticmethod
    def remove_files_or_dirs_from(list_with_paths):
        for f in list_with_paths:
            try:
                os.system("rm -rf {}".format(f))
            except:
                print("Could not remove file {}".format(f))


    def downsampleData(self,input_csv_path, out_csv_path, resampler_method='fourier', source_hz=100, target_hz=50, window_size=20000, discrete_columns=[], save=False):
        '''

        :param input_csv_path: A csv file with TIMESTAMP INDEX AND where COLUMNS are xyz from all sensors are merged into one file
        :param out_csv_path: the relative path where to save the resampled csv file
        :param resampler_method: DEFAULT FOURIER
        :param source_hz: DEFAULT 100
        :param target_hz: DEFAULT 50
        :param window_size: DEFAULT 20 000
        :param discrete_columns:list with column names in dataframe not to downsample. IF TRANING DATA, give ["label(s)"]
        :return: Path to saved downsampled csv file
        '''

        result_df = resampler(
            resampler=resampler_method,
            source_rate=source_hz,
            target_rate=target_hz,
            window_size=window_size,
            inputD=input_csv_path,
            output=out_csv_path,
            discrete_columns=discrete_columns,
            save=save
        )

        return out_csv_path, result_df



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
            res = model.window_classification(window)[0]
            # print("RESSS: >>>>>> :: ", res, type(res))

            # output_queue.put((idx, window, res))
            # TODO remove window from output tuple, we do not need the temperature window anymore
            output.append((idx, res))




    def parallel_pipeline_classification_run(self,
                                             dataframe,
                                             dataframe_columns,
                                             rfc_model_path,
                                             lstm_models_paths,
                                             samples_pr_window,
                                             sampling_freq=50,
                                             train_overlap=0.8,
                                             num_proc_mod=1,
                                             seq_lenght=None,
                                             lstm_model_mapping={"both": '1', "thigh": '2', "back": '3'},
                                             minimize_result=True
                                             ):
        '''
        :param dataframe: Pandas DataFrame
        :param dataframe_columns: dictionary with keys
                ['back_features', 'thigh_features', 'back_temp', 'thigh_temp', 'label_column']
        :param rfc_model_path: str with path to saved RFC
        :param lstm_models_paths: dictionary containing lstm_mapping and path {rfc_result_number : model_path}
        :param samples_pr_window: 250 frames = 50second
        :param sampling_freq: Hz the dataset was recorded at :: 50hz DEFAULT ::
        :param train_overlap: :: DEFAULT 0.8 ::
        :param num_proc_mod: :: DEFAULT 1 ::
        :param seq_lenght: :: DEFAULT None ::
        :param lstm_model_mapping: :: DEFAULT {"both": '1', "thigh": '2', "back": '3'} ::
                dict with keys ["both" : <str>, "thigh" : <str>, "back": <str>]
        :param minimize_result: :: DEFAULT TRUE ::reduces the output size by taking the starttime of the first window in
                a sequence with same target, calculates avg of confidence over all sequential windows and
                takes the time of last window with same target, as the endtime of activity sequence
        :return: list<both>, list<thigh>, list<back>, each list has a new list with tuples (time, conf/prog, class)
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

        # Build arguments for get_features_and_labels function

        b_clm = DataHandler.getAttributeOrReturnDefault(dataframe_columns, 'back_features')
        t_clm = DataHandler.getAttributeOrReturnDefault(dataframe_columns, 'thigh_features')
        l_clm = DataHandler.getAttributeOrReturnDefault(dataframe_columns, 'label_column')

        btemp = DataHandler.getAttributeOrReturnDefault(dataframe_columns, 'back_temp')
        ttemp = DataHandler.getAttributeOrReturnDefault(dataframe_columns, 'thigh_temp')

        args = {
            'back_columns': b_clm,
            'thigh_columns': t_clm,
            'label_column': l_clm,
        }

        # extract back, thigh and labels
        back_feat, thigh_feat, labels = self.get_features_and_labels_as_np_array(
            df=self.dataframe,
            **args
        )


        btemp, ttemp, _ = self.get_features_and_labels_as_np_array(
            df=self.dataframe,
            back_columns=btemp,
            thigh_columns=ttemp,
            label_column=None
        )


        # calculate temperature features
        back_feat = temp_feature_util.segment_acceleration_and_calculate_features(back_feat,
                                                                  temp=btemp,
                                                                  samples_pr_window=samples_pr_window,
                                                                  sampling_frequency=sampling_freq,
                                                                  overlap=train_overlap)

        thigh_feat = temp_feature_util.segment_acceleration_and_calculate_features(thigh_feat,
                                                                   temp=ttemp,
                                                                   samples_pr_window=samples_pr_window,
                                                                   sampling_frequency=sampling_freq,
                                                                   overlap=train_overlap)


        # concatinates example : [[1,2,3],[4,5,6]] og [[a,b,c], [d,e,f]] --> [[1,2,3,a,b,c], [4,5,6,d,e,f]]
        # akas rebuild the dataframe shape
        both_features = np.hstack((back_feat, thigh_feat))

        number_of_tasks = both_features.shape[0]

        for idx, window in enumerate(both_features):
            model_queue.put((idx, window))


        # Lists to maintain processes
        processes_model = []

        # CREATE a worker processes on model klassifisering
        for _ in range(NUMBER_OF_PROCESSES_models):
            # RFC = pickle.load(open(rfc_model_path, 'rb'))
            RFC = self.instansiate_model("RFC", {})
            RFC.load_model(rfc_model_path)
            # print(RFC, dir(RFC))
            # input("....")

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
            progressbar.printProgressBar(
                current=int(number_of_tasks - model_queue.qsize()),
                totalOperations=number_of_tasks,
                sizeProgressBarInChars=30,
                explenation="Model classification :: ")


        # Tell child processes to stop waiting for more jobs
        for _ in range(NUMBER_OF_PROCESSES_models):
            model_queue.put('STOP')

        # print(">>>>>>>>>>>>>>>>> EUREKA <<<<<<<<<<<<<<<<<")

        # LET ALL PROCESSES ACTUALLY TERMINATE AKA FINISH THE JOB THEIR DOING
        while any([p.is_alive() for p in processes_model]):
            progressbar.printProgressBar(
                current=int(number_of_tasks - model_queue.qsize()),
                totalOperations=number_of_tasks,
                sizeProgressBarInChars=30,
                explenation="Model classification :: ")
            pass

        print("DONE")
        # print(">>>>>>>>>>>>>>>>> POT OF GOLD <<<<<<<<<<<<<<<<<")

        # join the processes aka block the threads, do not let them take on any more jobs
        for process in processes_model:
            process.join()

        # print("\nALL PROCESSES STATUS BEFORE TERMINATING:\n{}".format(processes_model))

        # Kill all the processes to release memory or process space or something. its 1.15am right now
        for process in processes_model:
            process.terminate()

        # print(">>>>>>>>>>>>>>>>> ||||||||| <<<<<<<<<<<<<<<<<")

        # continue the pipeline work
        # ...
        # ...
        # ...


        # print("OUTPUT/Activities windows to classify : ", len(output_classification_windows))

        both_sensors_windows_queue = list(
            filter(lambda x: x[1] == lstm_model_mapping['both'], output_classification_windows)
        )
        thigh_sensors_windows_queue = list(
            filter(lambda x: x[1] == lstm_model_mapping['thigh'], output_classification_windows)
        )
        back_sensors_windows_queue = list(
            filter(lambda x: x[1] == lstm_model_mapping['back'], output_classification_windows)
        )

        del output_classification_windows # save memory! GC can clean this now

        back_colums = DataHandler.getAttributeOrReturnDefault(dataframe_columns, 'back_features')
        thigh_colums = DataHandler.getAttributeOrReturnDefault(dataframe_columns, 'thigh_features')

        xBack = np.concatenate([dataframe[back_colums].values[ : (len(dataframe) - len(dataframe) % seq_lenght) ] for dataframe in [dataframe]])
        xThigh = np.concatenate([dataframe[thigh_colums].values[: (len(dataframe) - len(dataframe) % seq_lenght)] for dataframe in [dataframe]])

        xBack = xBack.reshape(-1, seq_lenght, len(back_colums))
        xThigh = xThigh.reshape(-1, seq_lenght, len(thigh_colums))

        # BOTH
        bth_class = self.predict_on_one_window(DataHandler.getAttributeOrReturnDefault(lstm_model_mapping, 'both'), lstm_models_paths, both_sensors_windows_queue, xBack, xThigh, seq_lenght)

        # THIGH
        thigh_class = self.predict_on_one_window(DataHandler.getAttributeOrReturnDefault(lstm_model_mapping, 'thigh'), lstm_models_paths, thigh_sensors_windows_queue, xBack, xThigh, seq_lenght)

        # BACK
        back_class = self.predict_on_one_window(DataHandler.getAttributeOrReturnDefault(lstm_model_mapping, 'back'), lstm_models_paths, back_sensors_windows_queue, xBack, xThigh, seq_lenght)


        # Get timestamps (TODO: extract to own function)
        indexes = np.array(self.dataframe.index.tolist())

        num_windows = int(indexes.shape[0] / seq_lenght)
        prev_window = 0
        next_window = prev_window + seq_lenght
        timestap_windows = []
        while len(timestap_windows) < num_windows:
            indexes_window = [ indexes[prev_window : next_window] ]
            timestap_windows.append( indexes_window )
            prev_window = next_window
            next_window += seq_lenght

        timestap_windows = np.array(timestap_windows)

        # Todo extract to static method in DataHandler
        #build dataframe [timestart, timeend, confidence, target]
        import pandas as pd
        result_df = pd.DataFrame(columns=['timestart', 'timeend', 'confidence', 'target'])
        result_df['timestart'] = pd.to_datetime(result_df['timestart']).sort_values() # sort the dataframe on starttime
        result_df['timeend'] = pd.to_datetime(result_df['timeend'])
        result_df['confidence'] = pd.to_numeric(result_df['confidence'])
        result_df['target'] = pd.to_numeric(result_df['target'])

        # combine all windows for saving
        # classifications = np.concatenate((bth_class, thigh_class, back_class))

        classifications = np.array([]).reshape(-1, 3)
        if bth_class.shape[0] > 0 and bth_class.shape[1] == 3:
            classifications = np.vstack((classifications, bth_class))
        if thigh_class.shape[0] > 0 and thigh_class.shape[1] == 3:
            classifications = np.vstack((classifications, thigh_class))
        if back_class.shape[0] > 0 and back_class.shape[1] == 3:
            classifications = np.vstack((classifications, back_class))

        if not minimize_result:  # do not minimize result
            i = 1
            for idx, conf, target in classifications:
                timestart = timestap_windows[idx][0][0]
                timeend = timestap_windows[idx][0][-1]
                conf = conf[0]
                target = target[0]

                row = {
                    'timestart': timestart,
                    'timeend': timeend,
                    'confidence': conf,
                    'target': target
                }
                result_df.loc[len(result_df)] = row
                progressbar.printProgressBar(
                    current=i,
                    totalOperations=len(classifications),
                    sizeProgressBarInChars=20,
                    explenation='Creating result dataframe')
        else:  # minizime result
            windowMemory = WindowMemory()
            counter = 0
            counter_target = len(classifications)
            for idx, conf, target in classifications:

                timestart = timestap_windows[idx][0][0]
                timeend = timestap_windows[idx][0][-1]
                conf = conf[0]
                target = target[0]

                # TODO: do some logic that finds timestart and timeend for sequences of same target, take avg, conf and then thats the row we want to add to dataframe, not all windows!
                if windowMemory.get_last_target() is None: # if the first window to classify
                    # this can be done outside the loop, to not check each time, use classifiaction.pop() on var init
                    # last_target = target
                    windowMemory.update_last_target(target)
                    # last_start = timestart
                    windowMemory.update_last_start(timestart)
                    # last_end = timeend
                    windowMemory.update_last_end(timeend)
                    # avg_conf += conf
                    windowMemory.update_avg_conf_nominator(conf)

                elif not windowMemory.check_targets(target): # not same target as last window, thus write last window to mem.
                    # add to result_df
                    row = {
                        'timestart': windowMemory.get_last_start(),
                        'timeend': windowMemory.get_last_end(),
                        'confidence': windowMemory.get_avg_conf(),
                        'target': windowMemory.get_last_target()
                    }

                    result_df.loc[len(result_df)] = row

                    # keep track of new windows with same result
                    windowMemory.reset_avg_conf()
                    windowMemory.update_avg_conf_nominator(conf)
                    windowMemory.update_last_target(target)
                    windowMemory.update_last_start(timestart)
                    windowMemory.update_last_end(timeend)
                    windowMemory.reset_divisor()

                elif counter == counter_target-1: # if last window to classify
                    row = {
                        'timestart': windowMemory.get_last_start(),
                        'timeend': windowMemory.get_last_end(),
                        'confidence': windowMemory.get_avg_conf(),
                        'target': windowMemory.get_last_target()
                    }
                    result_df.loc[len(result_df)] = row

                else: # same target as last window and just in the middle of classification
                    # upate memory_variables
                    windowMemory.update_last_end(timeend)
                    windowMemory.update_avg_conf_nominator(conf)
                    windowMemory.update_avg_conf_divisor()


                # Feedback to user
                progressbar.printProgressBar(
                    current=windowMemory.get_num_windows(),
                    totalOperations=len(classifications),
                    sizeProgressBarInChars=20,
                    explenation='Creating result dataframe')

                # Controll feedback to user
                windowMemory.update_num_windows()

                counter += 1

            print("DONE")

        return bth_class, thigh_class, back_class, result_df

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
                x1 = xBack[wndo_idx].reshape(1, seq_lenght, xBack.shape[2])
                x2 = xThigh[wndo_idx].reshape(1, seq_lenght, xThigh.shape[2])
                target, prob = model.predict_on_one_window(window=[x1, x2])
                classifications.append((wndo_idx, prob, target))

            elif mod == '2':
                task = "Thigh"
                x = xThigh[wndo_idx].reshape(1, seq_lenght, xThigh.shape[2])
                target, prob = model.predict_on_one_window(window=x)
                classifications.append((wndo_idx, prob, target))

            elif mod == '3':
                task = "Back"
                x = xBack[wndo_idx].reshape(1, seq_lenght, xBack.shape[2])
                target, prob = model.predict_on_one_window(window=x)
                classifications.append((wndo_idx, prob, target))

            # print("<<<<>>>>><<<>>>: \n", ":: " + model_num +" ::", target, prob)
            progressbar.printProgressBar(start, end, 20, explenation=task + " activity classification prog. :: ")
            start += 1

        progressbar.printProgressBar(start, end, 20, explenation="Activity classification prog. :: ")
        print("Done") # create new line
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


    def unzip_multiple_directories(self, list_with_dirs, zip_to="../data/temp/", synched_file_name="timesynched_csv.csv"):
        unzipped_paths = []
        for path in list_with_dirs:
            dir_name = path.split("/")[-1]
            rel_path = zip_to + dir_name
            if os.path.exists(rel_path):
                os.system("rm -rf {}".format(rel_path))
            # Now we know there is no directory with that name in the directory to
            unzipped_paths.append(self.dh.unzip_synch_cwa(path, temp_dir=zip_to, timeSynchedName=synched_file_name))

        return unzipped_paths


    def create_large_dataframe_from_multiple_input_directories(self,
                                                               list_with_subjects,
                                                               merge_column,
                                                               back_keywords=['Back', "B"],
                                                               thigh_keywords=['Thigh', "T"],
                                                               label_keywords=['GoPro', "Labels", "interval", "intervals", "json"],
                                                               synched_keywords=["timesynched"],
                                                               out_path=None,
                                                               master_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz'],
                                                               slave_columns=['time', 'bx1', 'by1', 'bz1', 'btemp'],
                                                               slave2_columns=['time', 'tx1', 'ty1', 'tz1', 'ttemp'],
                                                               rearrange_columns_to=None,
                                                               save=False,
                                                               added_columns_name=["labels"],
                                                               drop_non_labels=True,
                                                               verbose=True,
                                                               list=True,
                                                               downsample_config=None
                                                               ):
        '''

        Example of call:
        list_with_subjects=["../data/input/test.7z],
        back_keywords=['Back', "B"],
        thigh_keywords = ['Thigh', "T"],
        label_keywords = ['GoPro', "Labels", "intervals", "interval", "json"],
        synched_keywords=["timesynched"],
        out_path=None,
        merge_column='time',
        master_columns=['time', 'bx', 'by', 'bz', 'tx', 'ty', 'tz'],
        slave_columns=['time', 'bx1', 'by1', 'bz1', 'btemp'],
        slave2_columns=['time', 'tx1', 'ty1', 'tz1', 'ttemp'],
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
                    ],
        save=False,
        added_columns_name=['labels']

        :param list_with_subjects:
        :param back_keywords:
        :param thigh_keywords:
        :param label_keywords:
        :param synched_keywords:
        :param out_path:
        :param merge_column:
        :param master_columns:
        :param slave_columns:
        :param slave2_columns:
        :param rearrange_columns_to:
        :param save:
        :param added_columns_name:
        :param drop_non_labels:
        :param verbose:
        :param list:
        :return:
        '''


        for subj in list_with_subjects:
            if ".7z" in subj:
                cwa_converter.convert_cwas_to_csv_with_temp(
                    subject_dir=subj,
                    out_dir=subj,
                    paralell=True
                )


        subjects = DataHandler.findFilesInDirectoriesAndSubDirs(list_with_subjects,
                                                         back_keywords,
                                                         thigh_keywords,
                                                         label_keywords,
                                                         synched_keywords,
                                                         verbose=verbose)

        # print(subjects)
        merged_df = None
        if list:
            merged_df = []

        dh = DataHandler()
        dh_stacker = DataHandler()
        for idx, root_dir in enumerate(subjects):
            subject = subjects[root_dir]
            # print("SUBJECT: \n", subject, root_dir)
            back = os.path.join(root_dir, subject['backCSV'])
            thigh = os.path.join(root_dir, subject['thighCSV'])
            needSynchronization = ".7z" in root_dir
            print("Need Synchronization; ", needSynchronization)

            if needSynchronization:
                timesync = os.path.join(root_dir, subject['synchedCSV'])
                # dh = DataHandler()
                dh.merge_multiple_csvs(
                    timesync, back, thigh,
                    out_path=out_path,
                    master_columns=master_columns,
                    slave_columns=slave_columns,
                    slave2_columns=slave2_columns,
                    merge_how='left',
                    rearrange_columns_to=rearrange_columns_to,
                    merge_on=merge_column,
                    header_value=None,
                    # save=save
                )

                df = dh.concat_dataframes(
                    timesync,
                    root_dir + "/btemp.txt",
                    root_dir + "/ttemp.txt",
                    master_columns=master_columns,
                    slave_column=['btemp'],  # should probably be set snz we can specify temp_col_name
                    slave2_column=['ttemp'],  # should probably be set snz we can specify temp_cl_name
                    header_value=None,
                    # save=save
                )


                dh.convert_column_from_str_to_datetime(
                    dataframe=dh.get_dataframe_iterator(),
                )

                dh.set_column_as_index("time")

                for col_name in added_columns_name:
                    if col_name is "labels" or col_name is "label":
                        label = os.path.join(root_dir, subject['labelCSV'])
                        self.addLables(label, column_name=col_name, datahandler=dh)
                        if drop_non_labels:
                            dh.get_dataframe_iterator().dropna(subset=[col_name], inplace=True)
                    else:
                        dh.add_new_column(col_name)

            else:
                label = os.path.join(root_dir, subject['labelCSV'])
                df = dh.concat_dataframes(
                    back,
                    thigh,
                    label,
                    master_columns=['bx', 'by', 'bz'],
                    slave_column=['tx', 'ty', 'tz'],  # should probably be set snz we can specify temp_col_name
                    slave2_column=['label'],  # should probably be set snz we can specify temp_cl_name
                    header_value=None)


            # ALWAYS DO THIS, not dependent on file format.
            if downsample_config:
                # TODO: pass in downsample config as dictionary

                if downsample_config['add_timestamps']:
                    df.index = pd.date_range(
                        start=pd.Timestamp.now(),
                        periods=len(df),
                        freq=pd.Timedelta(seconds=1/downsample_config['source_hz'])
                    )


                outpath, res_df = self.downsampleData(
                    input_csv_path=df,
                    out_csv_path=downsample_config['out_path'],
                    discrete_columns=downsample_config['discrete_columns_list'],
                    source_hz=downsample_config['source_hz'],
                    target_hz=downsample_config['target_hz'],
                    window_size=downsample_config['window_size'],
                    save=save
                )

                print('Length {}Hz: {}\nLength {}Hz: {}'.format(
                    downsample_config['source_hz'],
                    len(df),
                    downsample_config['target_hz'],
                    len(res_df))
                )

                df = res_df

            if list:
                merged_df.append(df)
            else:
                if idx == 0:
                    merged_df = dh.get_dataframe_iterator()
                    continue

                # vertically stack the dataframes aka add the rows from dataframe2 as rows to the dataframe1
                merged_df = dh_stacker.vertical_stack_dataframes(merged_df, dh.get_dataframe_iterator(), set_as_current_df=False)

            progressbar.printProgressBar(idx, len(subjects), 20, explenation='Merging datasets prog.: ')

        progressbar.printProgressBar(len(subjects), len(subjects), 20, explenation='Merging datasets prog.: ')

        if save and not list:
            out_path_dir, out_path_filename = os.path.split(out_path)
            if out_path_filename == "":
                out_path_filename = "MERGED_CSVS_SYNCED_OR_NOT.csv"
            if not os.path.exists(out_path_dir):
                os.makedirs(out_path_dir)
            out_path = os.path.join(out_path_dir, out_path_filename)
            merged_df.to_csv(out_path, index=False)

        print("DONE")
        return merged_df

    ####################################################################################################################
    #                                            vPIPELINE CODE FOR RUNNING MODELSv                                    #
    ####################################################################################################################


    def instansiate_model(self, model_name, model_args):
        return models.get(model_name, model_args)


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
                         save_weights=False,
                         shuffle=False
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
        :param shuffle: if given, set numpy random seed to 47, then shuffle the windows and labels
        :return: model, leave one out histroy: the trained model object, a dictionary with history of leave one out passes
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
        if not validation_dataframe is None and type(validation_dataframe) == pd.DataFrame:
            validation_dataframe = [validation_dataframe]

        # potentially overwrite config variables
        batch_size = batch_size or config.TRAINING['args']['batch_size']
        sequence_length = sequence_length or config.TRAINING['args']['sequence_length']
        callbacks = config.TRAINING['args']['callbacks'] or None

        cols = None


        if type(training_dataframe) == pd.DataFrame:
           training_dataframe = [training_dataframe]

        model_history = None

        ######### DO TRAINING AND PREDICTION HERE ###########
        indexes = [i for i in range( 1, len( training_dataframe )+1) ]
        X = np.array(indexes)

        loo = LeaveOneOut()

        RUNS_HISTORY = {}
        previous_acc = 0.0

        for train_index, test_index in loo.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            trainingset = []
            testset = training_dataframe[test_index[0]]
            for idx in train_index:
                trainingset.append(training_dataframe[idx])

            if back_cols and thigh_cols:
                self.num_sensors = 2
                cols = [back_cols, thigh_cols]

                model_history = model.train(
                    train_data=trainingset,
                    valid_data=validation_dataframe,
                    epochs=config.TRAINING['args']['epochs'],
                    batch_size=batch_size,  # gets this from config file when init model
                    sequence_length=sequence_length,  # gets this from config file when init model
                    back_cols=back_cols,
                    thigh_cols=thigh_cols,
                    label_col=label_col,
                    shuffle=shuffle,
                    callbacks=callbacks
                )

                preds, gt, cm = model.predict(
                    dataframes=[testset],
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    back_cols=back_cols,
                    thigh_cols=thigh_cols,
                    label_col=label_col)
            else:
                cols = back_cols or thigh_cols
                self.num_sensors = 1

                model_history = model.train(
                    train_data=trainingset,
                    valid_data=validation_dataframe,
                    epochs=config.TRAINING['args']['epochs'],
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    cols=cols,
                    label_col=label_col,
                    shuffle=shuffle,
                    callbacks=callbacks,
                )

                preds, gt, cm = model.predict(
                    dataframes=[testset],
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    cols= back_cols or thigh_cols,
                    label_col=label_col)

            gt = gt.argmax(axis=1)
            preds = preds.argmax(axis=1)


            precision, recall, fscore, support = precision_recall_fscore_support(gt, preds)
            # print("P \n", precision)
            # print("R \n", recall)
            # print("F \n", fscore)
            # print("S \n", support)
            print()

            # only use labels present in the data
            labels = []
            label_values = list(set(gt)) + list(set(preds))
            label_values = list(set(label_values))
            label_values.sort()

            for i in label_values:
                # print("I: ", i)
                shift_up_from_OHE_downshift = i + 1
                labels.append(model.encoder.name_lookup[shift_up_from_OHE_downshift])

            # print(labels)
            # input("...")

            report = classification_report(gt, preds, target_names=labels, output_dict=True)

            acc = accuracy_score(gt, preds)
            print("PREV ACC {} -->  ACC {}".format(previous_acc, acc))
            print("Save: ", (save_to_path and (save_weights or save_model) and acc > previous_acc))

            #####
            # Save the model / weights
            #####
            prev_save = None
            if save_to_path and (save_weights or save_model) and acc > previous_acc:
                path = "{}_{}_{:.3f}".format(save_to_path, "ACC", acc)
                print("Done saving: {} \nSaved testmodel: {}\n Accuracy: {}".format(
                    model.save_model(path=path,
                                     model=save_model,
                                     weight=save_weights),
                    indexes[test_index[0]]-1,
                    acc
                ))
                previous_acc = acc

                try:
                    print("PREV_SAVE: ", prev_save)
                    if prev_save:
                        os.system('rm {}'.format(prev_save))
                        input("...")
                except:
                    print("Previous best saved weights could not be deleted")

                prev_save = path



            # Save the extra info to the report
            report['Accuracy'] = acc
            report['Confusion_matrix'] = cm
            report['Ground_truth'] = gt
            report['Predictions'] = preds
            report['Labels'] = labels
            # Add the current run report to the overall HISTORY report
            RUNS_HISTORY[indexes[test_index[0]]] = report

        ## print the RUN HISTROY dictionary
        pprint.pprint(RUNS_HISTORY)


        ## CALCULATE THE AVERAGE ACCURACY FOR THE LEAVE ONE OUT VALIDATION
        avg_acc = 0
        for key in RUNS_HISTORY:
            avg_acc += RUNS_HISTORY[key]['Accuracy']

        avg_acc /= len(RUNS_HISTORY.keys())
        print("AVG ACCURACY : ", avg_acc)
        RUNS_HISTORY['AVG_ACCURACY'] = avg_acc

        #####################################################

        # VARIABLE CONTROL
        self.config = config
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.cols = cols
        self.model = model

        # return the trained model (last run), leave one out history
        return self.model, RUNS_HISTORY


    def evaluate_lstm_model(self, dataframe, label_col, num_sensors=None, model=None, back_cols=None, thigh_cols=None, cols=None, batch_size=None, sequence_length=None):
        model = model or self.model
        num_sensors = num_sensors or self.num_sensors

        if type(dataframe) == pd.DataFrame:
           dataframe = [dataframe]

        if num_sensors == 2:
            return model.evaluate(
                          dataframes=dataframe,
                          batch_size=batch_size or self.config.TRAINING['args']['batch_size'],
                          sequence_length=sequence_length or self.config.TRAINING['args']['sequence_length'],
                          back_cols=self.cols[0] or back_cols,
                          thigh_cols=self.cols[1] or thigh_cols,
                          label_col=label_col)
        elif num_sensors == 1:
            return model.evaluate(
                          dataframes=dataframe,
                          batch_size=batch_size or self.config.TRAINING['args']['batch_size'],
                          sequence_length=sequence_length or self.config.TRAINING['args']['sequence_length'],
                          cols=self.cols or cols,
                          label_col=label_col)
        else:
            print("Pipeline.py :: evaluate_lstm_model ::")
            raise NotImplementedError()


    def predict_lstm_model(self, dataframe, label_col, num_sensors=None, model=None, back_cols=None, thigh_cols=None, cols=None, batch_size=None, sequence_length=None):
        model = model or self.model
        num_sensors = num_sensors or self.num_sensors

        if type(dataframe) == pd.DataFrame:
           dataframe = [dataframe]

        if num_sensors == 2:
            return model.predict(
                          dataframes=dataframe,
                          batch_size=batch_size or self.config.TRAINING['args']['batch_size'],
                          sequence_length=sequence_length or self.config.TRAINING['args']['sequence_length'],
                          back_cols=self.cols[0] or back_cols,
                          thigh_cols=self.cols[1] or thigh_cols,
                          label_col=label_col)
        elif num_sensors == 1:
            return model.predict(
                          dataframes=dataframe,
                          batch_size=batch_size or self.config.TRAINING['args']['batch_size'],
                          sequence_length=sequence_length or self.config.TRAINING['args']['sequence_length'],
                          cols=self.cols or cols,
                          label_col=label_col)
        else:
            print("Pipeline.py :: evaluate_lstm_model ::")
            raise NotImplementedError()


    def train_rfc_model(self,
                        back,
                        thigh,
                        btemp,
                        ttemp,
                        labels,
                        model=None,
                        window_length=250,
                        sampling_freq=50,
                        train_overlap=.8,
                        number_of_trees_in_forest=100,
                        snt_memory_seconds=600,
                        use_acc_data=True,
                        ):



        self.RFC = model or models.get("RFC", {})

        self.RFC.train(
            back_training_feat=back,
            thigh_training_feat=thigh,
            back_temp=btemp,
            thigh_temp=ttemp,
            labels=labels,
            samples_pr_window=window_length,
            sampling_freq=sampling_freq,
            train_overlap=train_overlap,
            number_of_trees=number_of_trees_in_forest,
            snt_memory_seconds=snt_memory_seconds,
            use_acc_data=use_acc_data

        )

        return self.RFC


    def evaluate_rfc_model(self,
                           back,
                           thigh,
                           btemp,
                           ttemp,
                           labels,
                           model=None,
                           sampling_frequency=50,
                           window_length=250,
                           snt_memory_seconds=600,
                           use_acc_data=True,
                           train_overlap=.8):

        RFC = model or self.RFC

        RFC.test(back,
                 thigh,
                 [btemp, ttemp],
                 labels,
                 samples_pr_window=window_length,
                 sampling_freq=sampling_frequency,
                 snt_memory_seconds=snt_memory_seconds,
                 use_acc_data=use_acc_data,
                 train_overlap=train_overlap)

        acc = RFC.calculate_accuracy()
        return acc


    def classify_rfc(self,
                     back,
                     thigh,
                     btemp,
                     ttemp,
                     labels,
                     model=None,
                     sampling_frequency=50,
                     window_lenght=250,
                     snt_memory_seconds=600,
                     use_acc_data=True,
                     train_overlap=.8):

        RFC = model or self.RFC


        return RFC.classify(
            back,
            thigh,
            [btemp, ttemp],
            labels,
            samples_pr_window=window_lenght,
            sampling_freq=sampling_frequency,
            snt_memory_seconds=snt_memory_seconds,
            use_acc_data=use_acc_data,
            train_overlap=train_overlap)




    def save_model(self, model, path):
        s = ""
        while s not in ["y", "n"]:
            try:
                s = input("save model ? [y | n]")
                if s == "y":
                    model.save_model(path=path)
            except:
                print("Something went wrong. Could not save")
        if s == "y":
            print("Done saving")
        else:
            print("Did not save")

    ####################################################################################################################
    #                                            ^PIPELINE CODE FOR RUNNING MODELS^                                    #
    ####################################################################################################################

    ####################################################################################################################
    #                                            ^PIPELINE CODE FOR PLOTTING^                                    #
    ####################################################################################################################

    def plot_confusion_matrix(self, y_true, y_pred, classes, figure=None, axis=None, normalize=False, title=None):
        plot = self.plotter.plot_confusion_matrix(y_true, y_pred, classes, normalize, title, figure=figure, axis=axis)



    ####################################################################################################################
    #                                            ^PIPELINE CODE FOR PLOTTING^                                    #
    ####################################################################################################################






if __name__ == '__main__':
    p = Pipeline()