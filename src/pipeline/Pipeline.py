import sys, os
# try: sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__), '..')))
# except: print("SAdsadsadhsa;hkldasjkd")

import numpy as np
import cwa_converter
import time
import random
import pickle
from collections import Counter
from multiprocessing import Process, Queue, current_process, freeze_support
from pipeline.DataHandler import DataHandler
from src import models

class Pipeline:
    def __init__(self):
        print("HELLO FROM PIPELINE")


    def unzip_extractNconvert_temp_merge_dataset(self, rel_filepath, label_interval, label_mapping):
        # Create a data handling object for importing and manipulating dataset ## PREPROCESSING
        print('CREATING datahandler')
        dh = DataHandler()
        print('CREATED datahandler')

        ##########################
        #
        #
        ##########################


        # unzip cwas from 7z arhcive
        unzipped_path = dh.unzip_7z_archive(
            filepath=os.path.join(os.getcwd(), rel_filepath),
            unzip_to_path='../data/temp',
            cleanup=False
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
            paralell=True
        )

        ##########################
        #
        #
        ##########################

        # Timesynch and concate csvs
        dh.merge_csvs_on_first_time_overlap(
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

        df = dh.get_dataframe_iterator()
        print(df.head(5))
        # input("looks ok ? \n")


        ##########################
        #
        #
        ##########################

        dh.convert_ADC_temp_to_C(
            dataframe=df,
            dataframe_path=None,
            normalize=False,
            save=True
        )

        df = dh.get_dataframe_iterator()
        print(df.head(5))
        # input("looks ok ? \n")

        ##########################
        #
        #
        ##########################


        print('SET INDEX TO TIMESTAMP')
        #test that this works with a dataframe and not only path to csv
        # thus pre-loaded and makes it run a little faster
        dh.convert_column_from_str_to_datetime_test(
                dataframe=df,
        )

        dh.set_column_as_index("time")
        print('DONE')

        ##########################
        #
        #
        ##########################


        print('MAKE NUMERIC')
        dh.convert_column_from_str_to_numeric(column_name="btemp")

        dh.convert_column_from_str_to_numeric(column_name="ttemp")
        print('DONE')

        ##########################
        #
        #
        ##########################


        print('ADDING LABELS')
        dh.add_new_column()
        print('DONE')

        dh.add_labels_file_based_on_intervals(
            intervals=label_interval,
            label_mapping=label_mapping
        )


        # ##########################
        # #
        # #
        # ##########################

        # dh.show_dataframe()
        df = dh.get_dataframe_iterator()

        return df, dh

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

    def max(self, array):
        '''
        Finds and returns the highest value in the array
        :param array:
        :return:
        '''
        return np.max(array)

    def min(self, array):
        '''
        Finds and returns the lowest value in the array
        :param array:
        :return:
        '''
        return np.min(array)

    def first_last_delta(self, array):
        temperature_first_sample_in_window = array[0]
        temperature_last_sample_in_window = array[-1]
        return (temperature_last_sample_in_window - temperature_first_sample_in_window)

    def max_min_delta(self, array):
        '''
        # The variation in temperature from the warmest to the coldest
        :param array:
        :return:
        '''
        # max_temp = max(array)
        # min_temp = min(array)

        max_temp = np.amax(array)
        min_temp = np.amin(array)
        return (max_temp - min_temp)

    def segment_acceleration_and_calculate_features(self,
                                                    sensor_data,
                                                    samples_pr_window=50,
                                                    overlap=0.0,
                                                    remove_sign_after_calculation=True):
        '''

        :param sensor_data:
        :param samples_pr_window:
        :param overlap:
        :param remove_sign_after_calculation:
        :return:
        '''

        # print("len sensor data: ", sensor_data.shape)
        functions = [
            self.max,
            self.min,
            self.max_min_delta,
            self.first_last_delta,
        ]

        # window_samples = int(sampling_rate * window_length)
        window_samples = samples_pr_window

        # print("Windows samples ", window_samples)
        step_size = int(round(window_samples * (1.0 - overlap)))

        all_features = []

        for window_start in np.arange(0, sensor_data.shape[0], step_size):
            # print("Window start: ", window_start, "Sensor_data.shape[0]: ", sensor_data.shape[0], step_size)
            window_start = int(round(window_start))
            window_end = window_start + int(round(window_samples))
            if window_end > sensor_data.shape[0]:
                break
            window = sensor_data[window_start:window_end]

            # print("Window", window)
            extracted_features = []
            # print("Windows start: ")
            index_of_function = 0
            n = 0
            for func in functions:
                value = func(window)
                extracted_features.append(value)

            all_features.append(np.hstack(extracted_features))

        one_large_array = np.vstack(all_features)

        if remove_sign_after_calculation:
            np.absolute(one_large_array, one_large_array)

        print(one_large_array.shape)

        return one_large_array

    def segment_labels(self, label_data, overlap=0.0, samples_pr_window=50):
        # window_samples = int(sampling_rate * window_length)
        window_samples = samples_pr_window
        step_size = int(round(window_samples * (1.0 - overlap)))

        labels = []

        for window_start in np.arange(0, label_data.shape[0], step_size):
            window_start = int(round(window_start))
            window_end = window_start + int(round(window_samples))
            if window_end > label_data.shape[0]:
                break
            window = label_data[window_start:window_end]
            # print(window)
            top = self.find_majority_activity(window)
            labels.append(top)

        return np.array(labels)

    def find_majority_activity(self, window):
        sensor_labels_list = window.tolist()
        labels_without_list = []
        for sensor_label in sensor_labels_list:
            labels_without_list.append(sensor_label[0])
        counts = Counter(labels_without_list)
        top = counts.most_common(1)[0][0]
        return top


    ####################################################################################################################
    #                                   ^PARALLELL PIPELINE EXECUTE WINDOW BY WINDOW CODE^                             #
    ####################################################################################################################


    # MODEL Klassifisering
    def model_classification_worker(self, input_q, output, model):
        for window in iter(input_q.get, 'STOP'):
            print("model_classification_worker executing")
            # if window[0] <= 5:
            #     res = (1, window)
            #     time.sleep(5.5)
            # elif window[0] <= 13:
            #     res = (2, window)
            #     time.sleep(2.5)
            # else:
            #     res = (3, window)
            #     time.sleep(0.05)

            # Vil ha in ett ferdig window, aka windows maa lages utenfor her og addes til queue
            res = model.window_classification(window)

            # SUBMIT TASKS FOR ACTIVITY CLASSIFICATION
            output.put((res, window))
            print("worker done")

    # AKTIVITET Klassifisering
    def activity_classification_worker(self, input_q):
        # todo implement this to do LSTM classifications!
        for window in iter(input_q.get, 'STOP'):
            model, w = window[0], window[1]
            # time.sleep(0.5 * random.random())
            print("MODEL: {}\nWINDOW: {}".format(model, w))


    def paralle_run(self, dataframe, model_path, samples_pr_window, train_overlap):
        NUMBER_OF_PROCESSES_models = 3
        NUMBER_OF_PROCESSES_class = 1

        # Create queues
        model_queue = Queue()
        activity_queue = Queue()

        # Submit tasks for model klassifisering
        back_feat, thigh_feat, label = self.get_features_and_labels(dataframe) # returns numpy arrays
        back_feat = self.segment_acceleration_and_calculate_features(back_feat,
                                                                    samples_pr_window=samples_pr_window,
                                                                    overlap=train_overlap)

        thigh_feat = self.segment_acceleration_and_calculate_features(thigh_feat,
                                                                    samples_pr_window=samples_pr_window,
                                                                    overlap=train_overlap)
        both_features = np.hstack((back_feat, thigh_feat))

        for window in both_features:
            model_queue.put(window)

        # Lists to maintain processes
        processes_model = []
        processes_class = []

        # CREATE a worker processes on model klassifisering
        for i in range(NUMBER_OF_PROCESSES_class):
            processes_class.append(Process(target=self.activity_classification_worker, args=(activity_queue,)
                                           )
                                   )

        # START the worker processes
        for process in processes_class:
            process.start()

        # CREATE a worker processes on model klassifisering
        for i in range(NUMBER_OF_PROCESSES_models):
            # todo fix the path here, to be a input parameter
            RFC = pickle.load(open(model_path, 'rb'))
            processes_model.append(Process(target=self.model_classification_worker, args=(model_queue,
                                                                                          activity_queue,
                                                                                          RFC
                                                                                          )
                                           )
                                   )

        # START the worker processes
        for process in processes_model:
            process.start()

        # waith for tasks_queue to become empty before sending stop signal to workers
        while not model_queue.empty():
            pass

        # Tell child processes to stop waiting for more jobs
        for i in range(NUMBER_OF_PROCESSES_models):
            model_queue.put('STOP')

        # LET ALL PROCESSES ACTUALLY TERMINATE AKA FINISH THE JOB THEIR DOING
        while any([p.is_alive() for p in processes_model]):
            pass

        # waith for activity_queue to become empty before sending stop signal to workers
        while not activity_queue.empty():
            pass

        # Tell child processes to stop waiting for more jobs
        for i in range(NUMBER_OF_PROCESSES_class):
            activity_queue.put('STOP')

        # LET ALL PROCESSES ACTUALLY TERMINATE AKA FINISH THE JOB THEIR DOING
        while any([p.is_alive() for p in processes_class]):
            pass

        all_processes = processes_model + processes_class

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


    ####################################################################################################################
    #                                   ^PARALLELL PIPELINE EXECUTE WINDOW BY WINDOW CODE^                             #
    ####################################################################################################################


if __name__ == '__main__':
    p = Pipeline()