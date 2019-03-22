import sys, os
# try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
# except: print("SAdsadsadhsa;hkldasjkd")
import re, math, random, time
import numpy as np
import cwa_converter
import pickle
from multiprocessing import Process, Queue, current_process, freeze_support
from pipeline.DataHandler import DataHandler
from keras.models import load_model
from layers.normalize import Normalize
import utils.temperature_segmentation_and_calculation as temp_feature_util
from src.config import Config
from src import models



class PipelinePAR:
    '''
    This script tried to parallalize the DIFFERENT LSTMS (different, taks) over one GPU, but it seems like that GPUs are SIMD instructions and
    thus does not like to be parallized the way we thought and want.

    Data-parallelism is applying the SAME operation to multiple data-items (SIMD)
    A GPU is designed for data-parallelism

    This code does not work, but is not deleted in case of multiple GPU installment in future


    NOTE: TODO, check available GPU's then set each LSTM model to a specific GPU
    '''
    def __init__(self):
        print("HELLO FROM PIPELINE")
        # Create a data handling object for importing and manipulating dataset ## PREPROCESSING
        print('CREATING datahandler')
        self.dh = DataHandler()
        print('CREATED datahandler')
        self.dataframe = None

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

        sys.stdout.write('\r{} {} {} [{:>7.2%}]'.format(
            explenation,
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
        self.dh.convert_column_from_str_to_datetime(
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
    def model_classification_worker(self, input_q, output_queues, model):
        for idx, window in iter(input_q.get, 'STOP'):
            # print("model_classification_worker executing")
            # print("MODLE CLASSIFICATION: \n", "IDX: ", idx, "\n","WINDOW: \n", window, "\n", "SHAPE: ", window.shape, "\n", "DIMS: ", window.ndim)

            # Vil ha in ett ferdig window, aka windows maa lages utenfor her og addes til queue
            # TODO: enten velge ut x antall av window, predikere de og ta avg result som LSTM (den med flest forekomster)
            # TODO: eller bare ett random row in window og predikerer den og bruker res som LSTM
            res = model.window_classification(window[0])[0]
            # print("RESSS: >>>>>> :: ", res, type(res))
            # print("OUTPUTQUEUE: ", output_queues[str(res)])
            output_queues[str(res)].put((idx, window, res))




    # AKTIVITET Klassifisering
    def activity_class_both_worker(self, input_q, model, output_queue):
        # todo implement this to do LSTM classifications!
        '''
        models: 1: both, 2:thigh, 3:back
        :param input_q:
        :return:
        '''
        for window in iter(input_q.get, 'STOP'):
            # print("ELLO")
            # window_idx, model = window[0], window[1]
            idx, window, mod_clas = window[0], window[1][0], window[2]
            # print(">>>>:: ", idx, window, mod_clas)
            try:
                res = model.predict_on_one_window(window)
            except Exception as e:
                print(">>>>>>>> ERROR <<<<<<<<<\n", e)
            output_queue.put((idx, res))
            # print("BAY")

    def activity_class_thigh_worker(self, input_q, model, output_queue):
        # todo implement this to do LSTM classifications!
        '''
        models: 1: both, 2:thigh, 3:back
        :param input_q:
        :return:
        '''
        for window in iter(input_q.get, 'STOP'):
            # window_idx, model = window[0], window[1]
            idx, window = window[0], window[1][0]
            res = model.predict_on_one_window(window)
            output_queue.put((idx, res))

    def activity_class_back_worker(self, input_q, model, output_queue):
        # todo implement this to do LSTM classifications!
        '''
        models: 1: both, 2:thigh, 3:back
        :param input_q:
        :return:
        '''
        for window in iter(input_q.get, 'STOP'):
            # window_idx, model = window[0], window[1]
            idx, window = window[0], window[1][0]
            res = model.predict_on_one_window(window)
            output_queue.put((idx, res))
            pass



    def parallel_pipeline_classification_run(self, dataframe, rfc_model_path, lstm_models_paths, samples_pr_window, train_overlap=0.8, num_proc_mod=1, num_proc_clas=1, seq_lenght=None):
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
        NUMBER_OF_PROCESSES_class = num_proc_clas

        # Create queues
        model_queue = Queue()
        both_sensors_queue = Queue()
        thigh_sensor_queue = Queue()
        back_sensor_queue = Queue()
        output_classification_queue = Queue()


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
        print("NUMBER OF TASKS: ", number_of_tasks, "  VS BOTH FETURES ", both_features.shape)
        input("...")

        for idx, window in enumerate(both_features):
            model_queue.put((idx, window))

        print("MODEL_QUEUE SIZE : ", model_queue.qsize())
        input("...")

        # Lists to maintain processes
        processes_model = []
        processes_both = []
        processes_thigh = []
        processes_back = []

        all_processes_act_class = [processes_both, processes_thigh, processes_back]
        all_processes_function = [
            self.activity_class_both_worker,
            self.activity_class_thigh_worker,
            self.activity_class_back_worker
        ]
        all_processes_queues = [both_sensors_queue, thigh_sensor_queue, back_sensor_queue]

        number_of_processes_pr_lstm = int(math.floor(NUMBER_OF_PROCESSES_class // len(lstm_models_paths)))



        # CREATE a worker processes on activity klassifisering
        for idx, process_list, process_func, activity_queue in zip(range(1, len(lstm_models_paths)+1),
                                                              all_processes_act_class,
                                                              all_processes_function,
                                                              all_processes_queues):
            idx = str(idx)
            config = Config.from_yaml(lstm_models_paths[idx]['config'], override_variables={})
            model_name = config.MODEL['name']
            model_args = dict(config.MODEL['args'].items(), **config.INFERENCE.get('extra_model_args', {}))
            # model = models.get(model_name, model_args)

            for _ in range(number_of_processes_pr_lstm):
                print("IDX {}\n process_list: {}\n, process_fun {}\n, activity_queue {}\n".format(idx,
                                                                                                  process_list,
                                                                                                  process_func,
                                                                                                  activity_queue))
                model = models.get(model_name, model_args)
                model.model.load_weights(lstm_models_paths[idx]['weights'])
                input("... think what it should be and what it is...")
                # LAG MODEL OBJECKT som tilhører process_list, process_func og activity_queue  # LSTM = models.get( TwoSensorLSTM(), {} )
                process_list.append(Process(
                    target=process_func,
                    args=(activity_queue, model, output_classification_queue)))


        # START the worker processes for LSTM Classification
        for process_list in all_processes_act_class:
            for process in process_list:
                process.start()


        output_queues = {
            "1": both_sensors_queue,
            "2": thigh_sensor_queue,
            "3": back_sensor_queue
        }
        # CREATE a worker processes on model klassifisering
        for _ in range(NUMBER_OF_PROCESSES_models):
            RFC = pickle.load(open(rfc_model_path, 'rb'))
            processes_model.append(Process(target=self.model_classification_worker,
                                           args=(model_queue,
                                                 output_queues,
                                                 RFC)
                                           ))

        # START the worker processes
        for process in processes_model:
            process.start()

        # waith for tasks_queue to become empty before sending stop signal to workers
        while not model_queue.empty():
            print("CURRENT: {}\nQUEUE SIZE: {}".format(number_of_tasks - model_queue.qsize(), model_queue.qsize()))
            self.printProgressBar(
                current=int(number_of_tasks - model_queue.qsize()),
                totalOperations=number_of_tasks,
                sizeProgressBarInChars=30,
                explenation="Model classification :: ")
            # pass

        self.printProgressBar(
            current=int(number_of_tasks - model_queue.qsize()),
            totalOperations=number_of_tasks,
            sizeProgressBarInChars=30,
            explenation="Model classification :: ")
        print("DONE \n MODLE Q: {} \n BTH Q: {}\n T Q: {} \n B Q {}".format(model_queue.qsize(), both_sensors_queue.qsize(), thigh_sensor_queue.qsize(), back_sensor_queue.qsize()))

        # Tell child processes to stop waiting for more jobs
        for _ in range(NUMBER_OF_PROCESSES_models):
            model_queue.put('STOP')

        print("1")

        # LET ALL PROCESSES ACTUALLY TERMINATE AKA FINISH THE JOB THEIR DOING
        while any([p.is_alive() for p in processes_model]):
            pass

        print("2")

        # waith for activity_queue to become empty before sending stop signal to workers
        while not both_sensors_queue.empty():
            pass

        while not thigh_sensor_queue.empty():
            pass

        while not back_sensor_queue.empty():
            pass

        # Tell child processes to stop waiting for more jobs
        # for _ in range(NUMBER_OF_PROCESSES_class):
        #     activity_queue.put('STOP')
        for queue in all_processes_queues:
            for _ in range(number_of_processes_pr_lstm):
                queue.put('STOP')

        # LET ALL PROCESSES ACTUALLY TERMINATE AKA FINISH THE JOB THEIR DOING
        while any([p.is_alive() for p in processes_both]):
            pass

        while any([p.is_alive() for p in processes_thigh]):
            pass

        while any([p.is_alive() for p in processes_back]):
            pass

        all_processes = processes_model + processes_both + processes_thigh + processes_back

        # join the processes aka block the threads, do not let them take on any more jobs
        for process in all_processes:
            process.join()

        print("\nALL PROCESSES STATUS BEFORE TERMINATING:\n{}".format(all_processes))

        # Kill all the processes to release memory or process space or something. its 1.15am right now
        for process in all_processes:
            process.terminate()

        # continue the pipeline work
        # ...
        # ...
        # ...

        # See results
        while not output_classification_queue.empty():
            print(output_classification_queue.get())


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



        subjects = {}
        for subject in list_with_subjects:
            if not os.path.exists(subject):
                print("Could not find Subject at path: ", subject)

            files = {}
            for sub_files_and_dirs in os.listdir(subject):
                # print(sub_files_and_dirs)
                words = re.split("[_ .]", sub_files_and_dirs)
                words = list(map(lambda x: x.lower(), words))

                check_for_matching_word = lambda words, keywords: [True if keyword.lower() == word.lower() else False
                                                                   for word in words for keyword in keywords]

                if any(check_for_matching_word(words, back_keywords)):
                    files["backCSV"] = sub_files_and_dirs

                elif any(check_for_matching_word(words, thigh_keywords)):
                    files["thighCSV"] = sub_files_and_dirs

                elif any(check_for_matching_word(words, label_keywords)):
                    files["labelCSV"] = sub_files_and_dirs

            subjects[subject] = files

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


    @staticmethod
    def load_model_weights(model, weights_path):
        model.load_weights(weights_path)


if __name__ == '__main__':
    p = Pipeline()